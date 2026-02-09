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
from ..models.vision_models import (
    DetectedObject,
    FaceState,
    ObservationTier,
    ObservationType,
    RelevanceScore,
    SceneDescription,
)
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

        # Per-tier state for visual_context.txt
        self._scene_description: str = ""
        self._scene_objects: list[str] = []
        self._current_activity: str = ""
        self._recent_events: list[dict] = []
        self._max_recent_events = settings.CONTEXT_MAX_RECENT_CHANGES

        # Face tracker state
        self._face_state: Optional[FaceState] = None
        self._face_lock = threading.Lock()
        self._last_face_expression: str = "none"

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
        for clear_file in [
            settings.STRUCTURED_OBSERVATIONS_FILE,
            settings.VISUAL_CONTEXT_FILE,
        ]:
            try:
                clear_file.parent.mkdir(parents=True, exist_ok=True)
                clear_file.write_text("")
            except Exception as e:
                logger.debug(f"Failed to clear {clear_file.name}: {e}")

        # Start threads
        thread_configs = [
            ("eyes-capture", self._capture_loop),
            ("eyes-detection", self._detection_loop),
            ("eyes-analysis", self._analysis_loop),
            ("eyes-relevance", self._relevance_loop),
            ("eyes-output", self._output_loop),
        ]
        if settings.FACE_TRACKER_ENABLED:
            thread_configs.append(("eyes-face", self._face_tracker_loop))
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

                tier = self._classify_tier(observation)
                observation.tier = tier

                score = self._relevance_engine.evaluate(
                    observation, tier=tier.value
                )
                logger.debug(
                    f"Relevance [{tier.value}]: {score.overall_score:.2f} "
                    f"(report={score.should_report}, {score.reason})"
                )

                if score.should_report:
                    self._relevance_engine.mark_reported(tier=tier.value)
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
                tier = observation.tier or self._classify_tier(observation)
                timestamp = datetime.now(timezone.utc).isoformat()

                # Always write the full record to structured JSONL
                structured_record = json.dumps({
                    "timestamp": timestamp,
                    "observation_type": obs_type.value,
                    "tier": tier.value,
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

                    # Route to per-tier state and rebuild context file
                    self._update_tier_state(observation, tier, timestamp)
                    self._rebuild_visual_context()

                    self.state_manager.increment_reported()
                    logger.info(
                        f"Observation reported [{tier.value}]: "
                        f"{observation.description}"
                    )
                    self._notifier.notify_idle()
                    self.state_manager.set_idle()
                else:
                    logger.debug(
                        f"Observation recorded (below threshold): "
                        f"[{tier.value}] score={score.overall_score:.2f} "
                        f"{observation.description[:80]}"
                    )
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Output error: {e}")

    def _update_tier_state(
        self,
        observation: SceneDescription,
        tier: ObservationTier,
        timestamp: str,
    ) -> None:
        """Route an observation to the appropriate per-tier state buffer."""
        try:
            time_part = timestamp.split("T")[1][:8]
        except (IndexError, TypeError):
            time_part = timestamp

        if tier == ObservationTier.SCENE:
            self._scene_description = observation.description
            self._scene_objects = sorted(set(observation.object_labels)) if observation.object_labels else []
        elif tier == ObservationTier.ACTIVITY:
            self._current_activity = observation.description
        elif tier == ObservationTier.EVENT:
            self._recent_events.append({
                "time": time_part,
                "description": observation.description,
            })
            if len(self._recent_events) > self._max_recent_events:
                self._recent_events = self._recent_events[-self._max_recent_events:]

    def _rebuild_visual_context(self) -> None:
        """
        Rebuild the visual context file from all tier states.

        Called whenever any tier updates. Produces a layered view:
        SCENE, ACTIVITY, EXPRESSION, RECENT EVENTS.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        lines = [f"[Last updated: {timestamp}]", ""]

        if self._scene_description:
            lines.append("SCENE:")
            lines.append(self._scene_description)
            if self._scene_objects:
                lines.append(f"Objects: {', '.join(self._scene_objects)}")
            lines.append("")

        if self._current_activity:
            lines.append("ACTIVITY:")
            lines.append(self._current_activity)
            lines.append("")

        with self._face_lock:
            face = self._face_state
        if face and face.primary_expression != "none":
            lines.append("EXPRESSION:")
            lines.append(face.primary_expression)
            lines.append("")

        if self._recent_events:
            lines.append("RECENT EVENTS:")
            for event in self._recent_events:
                lines.append(f"- {event['time']} | {event['description']}")

        content = "\n".join(lines) + "\n"

        try:
            context_file = settings.VISUAL_CONTEXT_FILE
            tmp_file = context_file.with_suffix(".tmp")
            tmp_file.write_text(content)
            tmp_file.rename(context_file)
        except Exception as e:
            logger.debug(f"Failed to write visual context: {e}")

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

    @staticmethod
    def _classify_tier(observation: SceneDescription) -> ObservationTier:
        """Classify an observation into a tier for cooldown and output routing."""
        desc = observation.description.lower()

        # EVENT tier: immediate notable events
        event_keywords = [
            "entered", "left", "walked in", "appeared", "disappeared",
            "picked up", "put down", "grabbed", "dropped",
        ]
        if any(kw in desc for kw in event_keywords):
            return ObservationTier.EVENT

        # Safety is always an event
        if set(observation.object_labels) & {"knife", "scissors"}:
            return ObservationTier.EVENT

        # SCENE tier: rich VLM descriptions (cascade_detail, moondream, vlm)
        if observation.analysis_method in ("cascade_detail", "moondream", "vlm"):
            return ObservationTier.SCENE

        scene_keywords = ["room", "lighting", "background", "environment", "setting"]
        if any(kw in desc for kw in scene_keywords):
            return ObservationTier.SCENE

        # Default: ACTIVITY tier
        return ObservationTier.ACTIVITY

    @staticmethod
    def _build_face_state(result: SceneDescription) -> FaceState:
        """Build a FaceState from a MediaPipe SceneDescription."""
        expressions = [
            obj.label.split(":", 1)[1].strip()
            for obj in result.detected_objects
            if obj.label.startswith("face:")
        ]
        primary = expressions[0] if expressions else "none"
        blendshapes = None
        if result.detected_objects and result.detected_objects[0].blendshapes:
            blendshapes = result.detected_objects[0].blendshapes

        return FaceState(
            timestamp=time.time(),
            num_faces=len(result.detected_objects),
            primary_expression=primary,
            expressions=expressions,
            blendshapes=blendshapes,
            processing_time_ms=result.processing_time_ms,
        )

    def _face_tracker_loop(self) -> None:
        """Independent face tracker thread — runs MediaPipe at ~1s intervals."""
        try:
            from ..services.scene_analyzer import MediaPipeFaceAnalyzer
        except ImportError:
            logger.warning("MediaPipe not available, face tracker disabled")
            return

        face_analyzer = MediaPipeFaceAnalyzer()
        if not face_analyzer.is_available():
            logger.warning("MediaPipe face analyzer not available, face tracker disabled")
            return

        logger.info("Face tracker thread started")
        while self._running:
            try:
                frame = self._camera.get_frame()
                if frame is None:
                    time.sleep(settings.FACE_TRACKER_INTERVAL)
                    continue

                result = face_analyzer.analyze(frame)
                face_state = self._build_face_state(result)

                with self._face_lock:
                    self._face_state = face_state

                # Update context file only when expression changes
                changed = face_state.primary_expression != self._last_face_expression
                if not settings.FACE_EXPRESSION_CHANGE_ONLY or changed:
                    self._last_face_expression = face_state.primary_expression
                    self._rebuild_visual_context()

                time.sleep(settings.FACE_TRACKER_INTERVAL)
            except (RuntimeError, ImportError) as e:
                logger.warning(f"Face tracker unavailable, stopping: {e}")
                return
            except Exception as e:
                logger.error(f"Face tracker error: {e}")
                time.sleep(settings.FACE_TRACKER_INTERVAL)
