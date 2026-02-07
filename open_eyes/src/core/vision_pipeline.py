"""
Vision pipeline orchestrator.

Main pipeline that coordinates camera capture, change detection,
scene analysis, relevance filtering, and agent output through
threaded workers.

Thread architecture:
- Thread 1 (Capture): Camera frames -> capture_queue
- Thread 2 (Detection): capture_queue -> changed frames -> analysis_queue
- Thread 3 (Analysis): analysis_queue -> SceneDescription -> relevance_queue
- Thread 4 (Relevance): relevance_queue -> filtered -> output_queue
- Thread 5 (Output): output_queue -> visual_observations.txt
- Thread 6 (Display): Terminal visualization (optional)
"""

import json
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..config import settings
from ..models.vision_models import DetectedObject, ObservationType, RelevanceScore, SceneDescription
from ..services.analyzer_factory import create_analyzer
from ..services.camera_capture import CameraCapture
from ..services.change_detector import ChangeDetector
from ..services.context_reader import ConversationContextReader
from ..services.eyes_status_notifier import EyesStatusNotifier
from ..services.relevance_engine import RelevanceEngine
from ..services.vision_memory import VisionMemory
from .state_manager import StateManager

logger = logging.getLogger(__name__)


class VisionPipeline:
    """
    Main vision processing pipeline.

    Orchestrates camera capture, change detection, scene analysis,
    relevance filtering, and agent output through threaded workers.
    """

    def __init__(
        self,
        capture_fps: float = settings.CAPTURE_FPS,
        analyzer_mode: str = settings.ANALYZER_MODE,
        relevance_threshold: float = settings.RELEVANCE_THRESHOLD,
        camera_device: int = settings.CAMERA_DEVICE_INDEX,
        molt_speak_runtime: Optional[Path] = settings.MOLT_SPEAK_RUNTIME_DIR,
        enable_display: bool = settings.ENABLE_TERMINAL_DISPLAY,
    ):
        """
        Initialize the vision pipeline.

        Args:
            capture_fps: Frames per second for camera capture.
            analyzer_mode: Scene analyzer mode (yolo, vlm, cascade).
            relevance_threshold: Min relevance score to report.
            camera_device: Camera device index.
            molt_speak_runtime: Path to Molt-Speak runtime directory.
            enable_display: Enable terminal visualization.
        """
        self.enable_display = enable_display

        # State
        self.state_manager = StateManager()
        self._running = False

        # Inter-thread queues
        max_q = settings.MAX_QUEUE_SIZE
        self._capture_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_q)
        self._analysis_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_q)
        self._relevance_queue: queue.Queue[SceneDescription] = queue.Queue(maxsize=max_q)
        self._output_queue: queue.Queue[tuple[SceneDescription, RelevanceScore]] = queue.Queue(maxsize=max_q)

        # Components
        self._camera = CameraCapture(
            device_index=camera_device,
            target_fps=capture_fps,
        )
        self._change_detector = ChangeDetector()
        self._analyzer = create_analyzer(analyzer_mode)
        self._context_reader = ConversationContextReader(
            molt_speak_runtime_dir=molt_speak_runtime,
        )
        self._vision_memory = VisionMemory()
        self._relevance_engine = RelevanceEngine(
            relevance_threshold=relevance_threshold,
            context_reader=self._context_reader,
            vision_memory=self._vision_memory,
        )
        self._notifier = EyesStatusNotifier()

        # Output file
        self._observations_file = settings.VISUAL_OBSERVATIONS_FILE

        # Threads
        self._threads: list[threading.Thread] = []

        # Load persisted memory (long-term knowledge like known objects),
        # then clear short-term dedup state for a fresh session
        self._vision_memory.load_state()
        self._vision_memory.clear_session()

    def start(self) -> bool:
        """
        Start all pipeline threads.

        Returns:
            True if pipeline started successfully.
        """
        logger.info("Starting vision pipeline...")

        # Start camera
        if not self._camera.start():
            logger.error("Failed to start camera")
            self.state_manager.set_error("Camera unavailable")
            return False

        self._running = True

        # Clear session files (short-term only; long-term memory handled elsewhere)
        self._observations_file.parent.mkdir(parents=True, exist_ok=True)
        self._observations_file.write_text("")
        try:
            settings.STRUCTURED_OBSERVATIONS_FILE.parent.mkdir(
                parents=True, exist_ok=True
            )
            settings.STRUCTURED_OBSERVATIONS_FILE.write_text("")
        except Exception as e:
            logger.debug(f"Failed to clear structured observations: {e}")

        # Start threads
        thread_configs = [
            ("eyes-capture", self._capture_loop),
            ("eyes-detection", self._detection_loop),
            ("eyes-analysis", self._analysis_loop),
            ("eyes-relevance", self._relevance_loop),
            ("eyes-output", self._output_loop),
        ]
        if self.enable_display:
            thread_configs.append(("eyes-display", self._display_loop))

        for name, target in thread_configs:
            t = threading.Thread(target=target, daemon=True, name=name)
            t.start()
            self._threads.append(t)

        logger.info(
            f"Vision pipeline started with {len(self._threads)} threads"
        )
        self.state_manager.set_idle()
        return True

    def stop(self) -> None:
        """Gracefully stop all threads and release resources."""
        logger.info("Stopping vision pipeline...")
        self._running = False

        # Stop camera
        self._camera.stop()

        # Wait for threads
        for t in self._threads:
            t.join(timeout=2.0)

        # Persist memory
        self._vision_memory.save_state()

        # Cleanup
        self._notifier.cleanup()
        self.state_manager.set_stopped()
        logger.info("Vision pipeline stopped")

    def _capture_loop(self) -> None:
        """Thread 1: Capture frames from camera and queue them."""
        frame_interval = 1.0 / max(self._camera.target_fps, 0.1)

        while self._running:
            try:
                start = time.monotonic()
                frame = self._camera.get_frame()
                if frame is not None:
                    self.state_manager.set_capturing(self._camera.frame_count)
                    self._notifier.notify_capturing(self._camera.frame_count)
                    # Write frame to disk in background thread
                    threading.Thread(
                        target=self._write_latest_frame,
                        args=(frame,),
                        daemon=True,
                    ).start()
                    try:
                        self._capture_queue.put_nowait(frame)
                    except queue.Full:
                        pass
                # Drift-compensated sleep
                elapsed = time.monotonic() - start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            except Exception as e:
                logger.error(f"Capture error: {e}")
                time.sleep(1.0)

    def _write_latest_frame(self, frame: np.ndarray) -> None:
        """Write latest captured frame to disk for external viewers."""
        try:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            tmp_file = settings.LATEST_FRAME_FILE.with_suffix(".tmp")
            tmp_file.write_bytes(buf.tobytes())
            tmp_file.rename(settings.LATEST_FRAME_FILE)
        except Exception as e:
            logger.debug(f"Failed to write latest frame: {e}")

    def _detection_loop(self) -> None:
        """Thread 2: Check for scene changes, pass changed frames."""
        while self._running:
            try:
                frame = self._capture_queue.get(timeout=settings.QUEUE_TIMEOUT)
                magnitude = self._change_detector.get_change_magnitude(frame)
                self.state_manager.update_change_magnitude(magnitude)

                if self._change_detector.has_scene_changed(frame):
                    logger.debug(
                        f"Scene change detected (magnitude={magnitude:.3f})"
                    )
                    self._change_detector.update_reference(frame)
                    try:
                        self._analysis_queue.put_nowait(frame)
                    except queue.Full:
                        pass
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Detection error: {e}")

    @staticmethod
    def _serialize_detected_object(obj: DetectedObject) -> dict:
        """Serialize a DetectedObject to a JSON-safe dict."""
        data = {
            "label": obj.label,
            "confidence": obj.confidence,
            "bbox": list(obj.bbox),
            "area_fraction": obj.area_fraction,
        }
        if obj.landmarks is not None:
            data["landmarks"] = [
                {"x": round(lm.x, 4), "y": round(lm.y, 4), "z": round(lm.z, 4)}
                for lm in obj.landmarks
            ]
        if obj.blendshapes is not None:
            data["blendshapes"] = obj.blendshapes
        return data

    def _write_latest_detections(self, result: SceneDescription) -> None:
        """Write latest detection data to JSON for external viewers."""
        try:
            data = {
                "timestamp": result.timestamp,
                "frame_number": result.frame_number,
                "analysis_method": result.analysis_method,
                "description": result.description,
                "processing_time_ms": result.processing_time_ms,
                "detected_objects": [
                    self._serialize_detected_object(obj)
                    for obj in result.detected_objects
                ],
                "frame_resolution": [
                    settings.CAPTURE_RESOLUTION_W,
                    settings.CAPTURE_RESOLUTION_H,
                ],
            }
            tmp_file = settings.LATEST_DETECTIONS_FILE.with_suffix(".tmp")
            tmp_file.write_text(json.dumps(data))
            tmp_file.rename(settings.LATEST_DETECTIONS_FILE)
        except Exception as e:
            logger.debug(f"Failed to write latest detections: {e}")

    def _analysis_loop(self) -> None:
        """Thread 3: Run scene analysis on changed frames."""
        analyzer_failed = False
        while self._running:
            try:
                frame = self._analysis_queue.get(timeout=settings.QUEUE_TIMEOUT)

                if analyzer_failed:
                    continue  # Drain queue, don't retry broken analyzer

                self._notifier.notify_analyzing(
                    self._analyzer.__class__.__name__
                )
                self.state_manager.set_analyzing()

                result = self._analyzer.analyze(frame)
                logger.debug(
                    f"Analysis: {result.description} "
                    f"({result.processing_time_ms:.0f}ms)"
                )
                self._write_latest_detections(result)

                try:
                    self._relevance_queue.put_nowait(result)
                except queue.Full:
                    pass
            except queue.Empty:
                continue
            except RuntimeError as e:
                if not analyzer_failed:
                    logger.error(f"Analyzer unavailable, disabling analysis: {e}")
                    self._notifier.notify_error(str(e))
                    analyzer_failed = True
            except Exception as e:
                logger.error(f"Analysis error: {e}")

    def _relevance_loop(self) -> None:
        """Thread 4: Score all observations, pass everything to output."""
        while self._running:
            try:
                observation = self._relevance_queue.get(timeout=settings.QUEUE_TIMEOUT)
                self._vision_memory.record_observation(observation)

                score = self._relevance_engine.evaluate(observation)
                logger.debug(
                    f"Relevance: {score.overall_score:.2f} "
                    f"(report={score.should_report}, {score.reason})"
                )

                if score.should_report:
                    self._relevance_engine.mark_reported()
                    self._vision_memory.record_reported(observation)

                # Always send to output for structured logging;
                # output loop decides what goes to agent IPC vs full record
                try:
                    self._output_queue.put_nowait((observation, score))
                except queue.Full:
                    pass
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Relevance error: {e}")

    def _output_loop(self) -> None:
        """Thread 5: Record all observations; write agent IPC only for relevant ones."""
        structured_writes = 0
        while self._running:
            try:
                observation, score = self._output_queue.get(timeout=settings.QUEUE_TIMEOUT)

                obs_type = self._classify_observation(observation)
                timestamp = datetime.now(timezone.utc).isoformat()

                # Always write the full record to structured JSONL
                structured_record = json.dumps({
                    "timestamp": timestamp,
                    "observation_type": obs_type.value,
                    "description": observation.description,
                    "analysis_method": observation.analysis_method,
                    "processing_time_ms": observation.processing_time_ms,
                    "relevance_score": round(score.overall_score, 3),
                    "relevance_factors": {
                        "novelty": round(score.novelty, 3),
                        "context_match": round(score.context_match, 3),
                        "intrinsic_interest": round(score.intrinsic_interest, 3),
                        "timing": round(score.timing, 3),
                    },
                    "reported_to_agent": score.should_report,
                    "frame_resolution": [
                        settings.CAPTURE_RESOLUTION_W,
                        settings.CAPTURE_RESOLUTION_H,
                    ],
                    "detected_objects": [
                        self._serialize_detected_object(obj)
                        for obj in observation.detected_objects
                    ],
                }, separators=(",", ":")) + "\n"

                with open(settings.STRUCTURED_OBSERVATIONS_FILE, "a") as f:
                    f.write(structured_record)

                structured_writes += 1
                if structured_writes % 50 == 0:
                    self._maybe_rotate_structured_file()

                # Only write to agent IPC file when relevant
                if score.should_report:
                    self.state_manager.set_observing(observation.description)
                    self._notifier.notify_observing(observation.description)

                    text_line = f"{timestamp}|{obs_type.value}|{observation.description}\n"
                    with open(self._observations_file, "a") as f:
                        f.write(text_line)

                    self.state_manager.increment_reported()
                    logger.info(
                        f"Observation reported: [{obs_type.value}] "
                        f"{observation.description}"
                    )
                    self._notifier.notify_idle()
                    self.state_manager.set_idle()
                else:
                    logger.debug(
                        f"Observation recorded (below threshold): "
                        f"[{obs_type.value}] score={score.overall_score:.2f} "
                        f"{observation.description[:80]}"
                    )
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Output error: {e}")

    def _maybe_rotate_structured_file(self) -> None:
        """Truncate structured observations file if it exceeds max size."""
        try:
            fpath = settings.STRUCTURED_OBSERVATIONS_FILE
            if fpath.exists() and fpath.stat().st_size > settings.MAX_STRUCTURED_FILE_SIZE:
                lines = fpath.read_text().splitlines(keepends=True)
                half = len(lines) // 2
                fpath.write_text("".join(lines[half:]))
        except Exception as e:
            logger.debug(f"Failed to rotate structured observations: {e}")

    def _display_loop(self) -> None:
        """Thread 6: Terminal visualization of pipeline status."""
        while self._running:
            try:
                state = self.state_manager.state
                status_line = (
                    f"\r\033[K"
                    f"[Eyes] {state.status.value} | "
                    f"Frames: {state.frame_count} | "
                    f"Observations: {state.observations_reported} | "
                    f"Change: {state.last_change_magnitude:.2f}"
                )
                print(status_line, end="", flush=True)
                time.sleep(0.5)
            except Exception:
                time.sleep(1.0)

    @staticmethod
    def _classify_observation(
        observation: SceneDescription,
    ) -> ObservationType:
        """Classify an observation into a type based on content."""
        desc = observation.description.lower()
        labels = set(observation.object_labels)

        # Person-related
        if "person" in labels:
            if any(w in desc for w in ["walked in", "entered", "appeared", "new"]):
                return ObservationType.PERSON_DETECTED
            if any(w in desc for w in ["left", "gone", "disappeared"]):
                return ObservationType.PERSON_LEFT

        # Safety
        if labels & {"knife", "scissors"}:
            return ObservationType.SAFETY_ALERT

        # Object changes
        if any(w in desc for w in ["picked up", "put down", "moved", "grabbed"]):
            return ObservationType.OBJECT_CHANGE

        # Environment
        if any(w in desc for w in ["lighting", "dark", "bright", "background"]):
            return ObservationType.ENVIRONMENT_CHANGE

        # Default
        return ObservationType.SCENE_CHANGE
