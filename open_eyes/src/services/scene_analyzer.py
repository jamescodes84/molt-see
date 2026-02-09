"""
Scene analysis service.

Provides Protocol-based scene analyzers: YOLO for fast local detection,
VLM for rich semantic descriptions, and a Cascade combining both.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Optional, Protocol

import cv2
import numpy as np

from ..config import settings
from ..models.vision_models import DetectedObject, FaceLandmark, SceneDescription

logger = logging.getLogger(__name__)


# ============================================================================
# Scene Analyzer Protocol
# ============================================================================


class SceneAnalyzerProtocol(Protocol):
    """Protocol for scene analysis backends."""

    def analyze(self, frame: np.ndarray) -> SceneDescription:
        """Analyze a frame and return a scene description."""
        ...

    def is_available(self) -> bool:
        """Check if the analyzer is ready."""
        ...


# ============================================================================
# YOLO Analyzer (Local, Fast)
# ============================================================================


class YOLOAnalyzer:
    """
    Fast local object detection using YOLOv8.

    Provides object labels with bounding boxes and confidence scores.
    Runs entirely on-device with no API calls. Uses MPS acceleration
    on Apple Silicon.
    """

    def __init__(
        self,
        model_size: str = settings.YOLO_MODEL,
        confidence_threshold: float = settings.YOLO_CONFIDENCE,
    ):
        """
        Initialize YOLO analyzer.

        Args:
            model_size: YOLO model variant (yolov8n, yolov8s, etc.).
            confidence_threshold: Minimum detection confidence.
        """
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._load_failed = False

    def _load_model(self) -> None:
        """Lazy-load the YOLO model. Stops retrying after first failure."""
        if self._load_failed:
            raise RuntimeError(
                "YOLO model unavailable (previous load failed)"
            )
        if self._model is None:
            try:
                from ultralytics import YOLO

                self._model = YOLO(f"{self.model_size}.pt")
                logger.info(f"YOLO model loaded: {self.model_size}")
            except Exception as e:
                self._load_failed = True
                logger.error(f"Failed to load YOLO model: {e}")
                raise

    def analyze(self, frame: np.ndarray) -> SceneDescription:
        """
        Run YOLO detection on a frame.

        Args:
            frame: BGR numpy array.

        Returns:
            SceneDescription with detected objects and labels.
        """
        self._load_model()
        start = time.time()

        results = self._model(frame, verbose=False, conf=self.confidence_threshold)
        elapsed_ms = (time.time() - start) * 1000

        detected_objects = []
        object_labels = []
        frame_area = frame.shape[0] * frame.shape[1]

        for result in results:
            for box in result.boxes:
                label = result.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                box_area = (x2 - x1) * (y2 - y1)

                detected_objects.append(
                    DetectedObject(
                        label=label,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        area_fraction=box_area / frame_area,
                    )
                )
                if label not in object_labels:
                    object_labels.append(label)

        # Build a simple description from labels
        if object_labels:
            description = f"Detected: {', '.join(object_labels)}"
        else:
            description = "No notable objects detected"

        return SceneDescription(
            timestamp=time.time(),
            description=description,
            detected_objects=detected_objects,
            object_labels=object_labels,
            analysis_method="yolo",
            processing_time_ms=elapsed_ms,
        )

    def is_available(self) -> bool:
        """Check if YOLO model can be loaded."""
        try:
            self._load_model()
            return True
        except Exception:
            return False


# ============================================================================
# VLM Analyzer (API-Based, Rich Descriptions)
# ============================================================================


class VLMAnalyzer:
    """
    Rich semantic scene understanding via Vision Language Model API.

    Sends frame to Claude Vision (or similar) for natural language
    scene description. More expensive but much richer understanding.
    """

    VLM_PROMPT = (
        "Describe what you see in this image in one concise sentence, "
        "as if casually mentioning it in conversation. Focus on what's "
        "notable or what might have changed. Don't list objects mechanically. "
        'Example good output: "Looks like you picked up a coffee mug" '
        'rather than "Objects detected: mug, hand, desk."'
    )

    def __init__(
        self,
        provider: str = settings.VLM_PROVIDER,
        model: str = settings.VLM_MODEL,
        api_key: Optional[str] = settings.VLM_API_KEY,
        max_tokens: int = settings.VLM_MAX_TOKENS,
    ):
        """
        Initialize VLM analyzer.

        Args:
            provider: API provider ("anthropic" or "openai").
            model: Model identifier.
            api_key: API key for the provider.
            max_tokens: Maximum tokens for response.
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        """Lazy-initialize the API client."""
        if self._client is None:
            if self.provider == "anthropic":
                import anthropic

                if self.api_key and "-oat" in self.api_key:
                    self._client = anthropic.Anthropic(auth_token=self.api_key)
                else:
                    self._client = anthropic.Anthropic(api_key=self.api_key)
            else:
                raise ValueError(f"Unsupported VLM provider: {self.provider}")
        return self._client

    def analyze(self, frame: np.ndarray) -> SceneDescription:
        """
        Send frame to VLM API for scene description.

        Args:
            frame: BGR numpy array.

        Returns:
            SceneDescription with natural language description.
        """
        start = time.time()

        # Encode frame as base64 JPEG
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        image_base64 = base64.b64encode(buffer).decode("utf-8")

        description = self._call_api(image_base64)
        elapsed_ms = (time.time() - start) * 1000

        return SceneDescription(
            timestamp=time.time(),
            description=description,
            object_labels=[],  # VLM doesn't provide structured labels
            analysis_method="vlm",
            processing_time_ms=elapsed_ms,
        )

    def _call_api(self, image_base64: str) -> str:
        """Call the VLM API with an image."""
        client = self._get_client()

        if self.provider == "anthropic":
            message = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": self.VLM_PROMPT,
                            },
                        ],
                    }
                ],
            )
            return message.content[0].text

        raise ValueError(f"Unsupported provider: {self.provider}")

    def is_available(self) -> bool:
        """Check if API key is configured."""
        return self.api_key is not None and len(self.api_key) > 0


# ============================================================================
# Moondream Analyzer (Local VLM, Free)
# ============================================================================


class MoondreamAnalyzer:
    """
    Local vision-language model using Moondream 2B.

    Generates natural language scene descriptions entirely on-device.
    Uses bfloat16 + MPS on Apple Silicon. No API key required.
    """

    MAX_INPUT_SIZE = 384  # Pre-resize for faster inference

    _REVISION = "2025-06-21"

    def __init__(self, model_name: str = "vikhyatk/moondream2"):
        self.model_name = model_name
        self._model = None
        self._load_failed = False

    def _load_model(self) -> None:
        """Lazy-load Moondream model. Stops retrying after first failure."""
        if self._load_failed:
            raise RuntimeError(
                "Moondream model unavailable (previous load failed)"
            )
        if self._model is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM

                if torch.backends.mps.is_available():
                    device = torch.device("mps")
                    dtype = torch.float16
                else:
                    device = torch.device("cpu")
                    dtype = torch.float32

                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    revision=self._REVISION,
                    trust_remote_code=True,
                    dtype=dtype,
                ).to(device)
                logger.info(
                    f"Moondream model loaded: {self.model_name} "
                    f"(rev {self._REVISION}) on {device}"
                )
            except Exception as e:
                self._load_failed = True
                logger.error(f"Failed to load Moondream model: {e}")
                raise

    def analyze(self, frame: np.ndarray) -> SceneDescription:
        """
        Generate a natural language caption for a frame.

        Args:
            frame: BGR numpy array from OpenCV.

        Returns:
            SceneDescription with conversational description.
        """
        from PIL import Image

        self._load_model()
        start = time.time()

        # Convert BGR (OpenCV) to RGB PIL Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        # Pre-resize for faster inference
        orig_w, orig_h = image.size
        max_dim = max(orig_w, orig_h)
        if max_dim > self.MAX_INPUT_SIZE:
            scale = self.MAX_INPUT_SIZE / max_dim
            image = image.resize(
                (int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS
            )

        result = self._model.caption(image, length="short")
        caption = result["caption"] if isinstance(result, dict) else str(result)
        elapsed_ms = (time.time() - start) * 1000

        return SceneDescription(
            timestamp=time.time(),
            description=caption,
            object_labels=[],
            analysis_method="moondream",
            processing_time_ms=elapsed_ms,
        )

    def is_available(self) -> bool:
        """Check if Moondream model can be loaded."""
        return not self._load_failed


# ============================================================================
# Cascade Analyzer (YOLO + VLM)
# ============================================================================


class CascadeAnalyzer:
    """
    Two-stage cascade: fast local screening, then VLM when interesting.

    The fast stage (YOLO-World + MediaPipe Face by default) runs on
    every changed frame. If it detects something interesting (person,
    face with emotion, household object), the VLM is invoked for a
    richer natural language description.
    """

    # Object classes that warrant deeper VLM analysis
    INTERESTING_CLASSES = {
        # Living things
        "person", "cat", "dog", "bird",
        # Body parts
        "hand",
        # Face / emotion
        "face", "face: smiling", "face: talking", "face: surprised",
        "face: frowning", "face: eyes_closed",
        # Household / activity
        "guitar", "piano", "microphone", "camera", "headphones",
        "book", "cell phone", "phone", "laptop",
        "bottle", "cup", "mug",
        # Safety
        "knife", "scissors",
    }

    def __init__(
        self,
        fast_analyzer=None,
        detail_analyzer=None,
        vlm_trigger_threshold: float = 0.6,
    ):
        """
        Initialize cascade analyzer.

        Args:
            fast_analyzer: Local analyzer for fast screening
                (any object with analyze()/is_available()).
            detail_analyzer: Analyzer for rich descriptions when
                interesting objects found (Moondream, VLM, etc.).
            vlm_trigger_threshold: Min confidence on interesting
                classes to trigger detail analyzer.
        """
        self.fast = fast_analyzer or YOLOAnalyzer()
        self.detail = detail_analyzer or MoondreamAnalyzer()
        self.vlm_trigger_threshold = vlm_trigger_threshold
        self._detail_disabled = False

    def analyze(self, frame: np.ndarray) -> SceneDescription:
        """
        Run cascade analysis: fast local screening, detail if interesting.

        Args:
            frame: BGR numpy array.

        Returns:
            SceneDescription from fast analyzer or detail analyzer.
        """
        # Stage 1: Fast local screening
        fast_result = self.fast.analyze(frame)

        # Stage 2: Rich description if something interesting was detected
        # or if enough time has passed for a periodic scene-tier refresh
        if (
            not self._detail_disabled
            and self._should_get_detail(fast_result)
            and self.detail.is_available()
        ):
            logger.debug(
                f"Interesting objects detected ({fast_result.object_labels}), "
                "invoking detail analyzer for richer description"
            )
            try:
                detail_result = self.detail.analyze(frame)
                # Merge: use detail description but keep structured data
                detail_result.detected_objects = fast_result.detected_objects
                detail_result.object_labels = fast_result.object_labels
                detail_result.analysis_method = "cascade_detail"
                return detail_result
            except (RuntimeError, ImportError):
                self._detail_disabled = True
                logger.warning(
                    "Detail analyzer unavailable, cascade will use "
                    "local screening only."
                )
                return fast_result
            except Exception as e:
                logger.warning(
                    f"Detail analysis failed, using local result: {e}"
                )
                return fast_result

        fast_result.analysis_method = "cascade_local"
        return fast_result

    def _should_get_detail(self, result: SceneDescription) -> bool:
        """Decide whether to invoke detail analyzer based on screening results."""
        for obj in result.detected_objects:
            if obj.confidence >= self.vlm_trigger_threshold and (
                obj.label in self.INTERESTING_CLASSES
                or obj.label.startswith("face:")
            ):
                return True
        return False

    def is_available(self) -> bool:
        """Check if the fast analyzer is available."""
        return self.fast.is_available()


# ============================================================================
# Florence-2 Analyzer (Local VLM, Open-Vocabulary)
# ============================================================================


class Florence2Analyzer:
    """
    Open-vocabulary detection + captioning via Microsoft Florence-2.

    Runs two tasks per frame:
      - <OD> for bounding-box object detection (open vocabulary)
      - <DETAILED_CAPTION> for natural language scene description

    Runs entirely on-device. Uses float16 + MPS on Apple Silicon.
    """

    MAX_INPUT_SIZE = 512  # Pre-resize frames for faster processing

    def __init__(self, model_name: str = settings.FLORENCE_MODEL):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._device = None
        self._dtype = None
        self._load_failed = False

    def _load_model(self) -> None:
        """Lazy-load Florence-2 model. Stops retrying after first failure."""
        if self._load_failed:
            raise RuntimeError(
                "Florence-2 model unavailable (previous load failed)"
            )
        if self._model is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoProcessor

                if torch.backends.mps.is_available():
                    self._device = "mps"
                    self._dtype = torch.float16
                else:
                    self._device = "cpu"
                    self._dtype = torch.float32

                self._processor = AutoProcessor.from_pretrained(
                    self.model_name, trust_remote_code=True
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=self._dtype,
                ).to(self._device)

                logger.info(
                    f"Florence-2 model loaded: {self.model_name} "
                    f"on {self._device} ({self._dtype})"
                )
            except Exception as e:
                self._load_failed = True
                logger.error(f"Failed to load Florence-2 model: {e}")
                raise

    def _run_task(self, image, task_prompt: str) -> dict:
        """Run a single Florence-2 task and return parsed results."""
        import torch

        inputs = self._processor(
            text=task_prompt, images=image, return_tensors="pt"
        )
        # Cast pixel_values to model dtype, keep input_ids as long
        inputs = {
            k: v.to(self._device, dtype=self._dtype)
            if v.is_floating_point()
            else v.to(self._device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=1,
            )

        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        return self._processor.post_process_generation(
            generated_text, task=task_prompt, image_size=image.size
        )

    def analyze(self, frame: np.ndarray) -> SceneDescription:
        """
        Run Florence-2 dense region captioning on a frame.

        Uses a single <DENSE_REGION_CAPTION> pass that returns both
        bounding boxes and descriptive labels — half the latency of
        running OD + caption separately.

        Args:
            frame: BGR numpy array from OpenCV.

        Returns:
            SceneDescription with bounding boxes and natural language caption.
        """
        from PIL import Image

        self._load_model()
        start = time.time()

        # Convert BGR (OpenCV) to RGB PIL Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        # Pre-resize for faster inference (Florence resizes internally anyway)
        orig_w, orig_h = image.size
        max_dim = max(orig_w, orig_h)
        if max_dim > self.MAX_INPUT_SIZE:
            scale = self.MAX_INPUT_SIZE / max_dim
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)

        frame_area = orig_w * orig_h

        # Single pass: dense region captioning (bboxes + descriptive labels)
        result = self._run_task(image, "<DENSE_REGION_CAPTION>")
        data = result.get("<DENSE_REGION_CAPTION>", {})
        bboxes = data.get("bboxes", [])
        labels = data.get("labels", [])

        # Scale bboxes back to original frame coordinates
        if max_dim > self.MAX_INPUT_SIZE:
            scale_x = orig_w / image.size[0]
            scale_y = orig_h / image.size[1]
        else:
            scale_x = scale_y = 1.0

        detected_objects = []
        object_labels = []
        for bbox, label in zip(bboxes, labels):
            x1 = int(bbox[0] * scale_x)
            y1 = int(bbox[1] * scale_y)
            x2 = int(bbox[2] * scale_x)
            y2 = int(bbox[3] * scale_y)
            box_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
            detected_objects.append(
                DetectedObject(
                    label=label,
                    confidence=1.0,
                    bbox=(x1, y1, x2, y2),
                    area_fraction=box_area / frame_area if frame_area else 0.0,
                )
            )
            if label not in object_labels:
                object_labels.append(label)

        elapsed_ms = (time.time() - start) * 1000

        # Build description from region captions
        if labels:
            description = "; ".join(labels)
        else:
            description = "No notable regions detected"

        return SceneDescription(
            timestamp=time.time(),
            description=description,
            detected_objects=detected_objects,
            object_labels=object_labels,
            analysis_method="florence",
            processing_time_ms=elapsed_ms,
        )

    def is_available(self) -> bool:
        """Check if Florence-2 model can be loaded."""
        return not self._load_failed


# ============================================================================
# YOLO-World Analyzer (Open-Vocabulary, Tight Boxes)
# ============================================================================


class YOLOWorldAnalyzer:
    """
    Open-vocabulary object detection using YOLO-World.

    Detects arbitrary objects specified by a configurable class list,
    producing tight YOLO-quality bounding boxes. Unlike standard YOLO
    (80 fixed classes), YOLO-World accepts any text class names.
    """

    def __init__(
        self,
        model_name: str = settings.YOLO_WORLD_MODEL,
        confidence_threshold: float = settings.YOLO_WORLD_CONFIDENCE,
        class_list: list[str] | None = None,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.class_list = class_list if class_list is not None else settings.YOLO_WORLD_CLASSES
        self._model = None
        self._load_failed = False

    def _load_model(self) -> None:
        """Lazy-load YOLO-World model. Stops retrying after first failure."""
        if self._load_failed:
            raise RuntimeError(
                "YOLO-World model unavailable (previous load failed)"
            )
        if self._model is None:
            try:
                from ultralytics import YOLOWorld

                self._model = YOLOWorld(f"{self.model_name}.pt")
                self._model.set_classes(self.class_list)
                logger.info(
                    f"YOLO-World model loaded: {self.model_name} "
                    f"with {len(self.class_list)} classes"
                )
            except Exception as e:
                self._load_failed = True
                logger.error(f"Failed to load YOLO-World model: {e}")
                raise

    def analyze(self, frame: np.ndarray) -> SceneDescription:
        """Run YOLO-World open-vocabulary detection on a frame."""
        self._load_model()
        start = time.time()

        results = self._model(
            frame, verbose=False, conf=self.confidence_threshold
        )
        elapsed_ms = (time.time() - start) * 1000

        detected_objects = []
        object_labels = []
        frame_area = frame.shape[0] * frame.shape[1]

        for result in results:
            for box in result.boxes:
                label = result.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                box_area = (x2 - x1) * (y2 - y1)

                detected_objects.append(
                    DetectedObject(
                        label=label,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        area_fraction=box_area / frame_area,
                    )
                )
                if label not in object_labels:
                    object_labels.append(label)

        if object_labels:
            description = f"Detected: {', '.join(object_labels)}"
        else:
            description = "No notable objects detected"

        return SceneDescription(
            timestamp=time.time(),
            description=description,
            detected_objects=detected_objects,
            object_labels=object_labels,
            analysis_method="yolo_world",
            processing_time_ms=elapsed_ms,
        )

    def is_available(self) -> bool:
        """Check if YOLO-World model can be loaded."""
        return not self._load_failed


# ============================================================================
# MediaPipe Face Analyzer (Face Detection + Expressions)
# ============================================================================


class MediaPipeFaceAnalyzer:
    """
    Face detection and expression recognition using MediaPipe Face Landmarker.

    Uses the MediaPipe Tasks API with face blendshapes for accurate
    expression classification. Downloads the model on first use.
    Runs on CPU, ~10ms per frame.
    """

    _MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    )
    _MODEL_FILENAME = "face_landmarker.task"

    # Blendshape thresholds for expression classification
    _SMILE_THRESHOLD = 0.4
    _JAW_OPEN_THRESHOLD = 0.3
    _BROW_UP_THRESHOLD = 0.3
    _BLINK_THRESHOLD = 0.4
    _FROWN_THRESHOLD = 0.3

    def __init__(
        self,
        min_detection_confidence: float = settings.MEDIAPIPE_FACE_CONFIDENCE,
        max_num_faces: int = settings.MEDIAPIPE_MAX_FACES,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.max_num_faces = max_num_faces
        self._detector = None
        self._load_failed = False

    def _get_model_path(self) -> str:
        """Get path to face landmarker model, downloading if needed."""
        import os
        import urllib.request

        runtime_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "runtime"
        )
        runtime_dir = os.path.normpath(runtime_dir)
        os.makedirs(runtime_dir, exist_ok=True)

        model_path = os.path.join(runtime_dir, self._MODEL_FILENAME)

        if not os.path.exists(model_path):
            logger.info("Downloading MediaPipe face landmarker model...")
            urllib.request.urlretrieve(self._MODEL_URL, model_path)
            logger.info(f"Model saved to {model_path}")

        return model_path

    def _load_model(self) -> None:
        """Lazy-load MediaPipe Face Landmarker (Tasks API)."""
        if self._load_failed:
            raise RuntimeError(
                "MediaPipe Face Landmarker unavailable (previous load failed)"
            )
        if self._detector is None:
            try:
                import mediapipe as mp

                model_path = self._get_model_path()

                base_options = mp.tasks.BaseOptions(
                    model_asset_path=model_path
                )
                options = mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=True,
                    num_faces=self.max_num_faces,
                    min_face_detection_confidence=self.min_detection_confidence,
                )
                self._detector = (
                    mp.tasks.vision.FaceLandmarker.create_from_options(options)
                )
                logger.info("MediaPipe Face Landmarker loaded")
            except Exception as e:
                self._load_failed = True
                logger.error(f"Failed to load MediaPipe Face Landmarker: {e}")
                raise

    def analyze(self, frame: np.ndarray) -> SceneDescription:
        """Detect faces and infer expressions from blendshapes."""
        import mediapipe as mp

        self._load_model()
        start = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)
        elapsed_ms = (time.time() - start) * 1000

        detected_objects = []
        object_labels = []
        h, w = frame.shape[:2]
        frame_area = h * w

        for i, face_landmarks in enumerate(result.face_landmarks):
            xs = [lm.x * w for lm in face_landmarks]
            ys = [lm.y * h for lm in face_landmarks]
            x1 = int(min(xs))
            y1 = int(min(ys))
            x2 = int(max(xs))
            y2 = int(max(ys))

            # Preserve all 478 landmarks (normalized coordinates)
            landmarks = [
                FaceLandmark(x=lm.x, y=lm.y, z=lm.z)
                for lm in face_landmarks
            ]

            # Get expression from blendshapes
            blendshapes_dict = None
            if result.face_blendshapes and i < len(result.face_blendshapes):
                blendshapes_dict = {
                    bs.category_name: round(bs.score, 4)
                    for bs in result.face_blendshapes[i]
                }

            expression = self._classify_expression(blendshapes_dict)
            label = f"face: {expression}"
            box_area = max(0, (x2 - x1)) * max(0, (y2 - y1))

            detected_objects.append(
                DetectedObject(
                    label=label,
                    confidence=0.9,
                    bbox=(x1, y1, x2, y2),
                    area_fraction=box_area / frame_area if frame_area else 0.0,
                    landmarks=landmarks,
                    blendshapes=blendshapes_dict,
                )
            )
            if "face" not in object_labels:
                object_labels.append("face")
            if label not in object_labels:
                object_labels.append(label)

        if detected_objects:
            expressions = [obj.label for obj in detected_objects]
            description = f"Face detected: {', '.join(expressions)}"
        else:
            description = "No faces detected"

        return SceneDescription(
            timestamp=time.time(),
            description=description,
            detected_objects=detected_objects,
            object_labels=object_labels,
            analysis_method="mediapipe_face",
            processing_time_ms=elapsed_ms,
        )

    def _classify_expression(self, blendshapes: dict | None) -> str:
        """Classify facial expression from blendshape scores."""
        if blendshapes is None:
            return "neutral"

        smile = max(
            blendshapes.get("mouthSmileLeft", 0),
            blendshapes.get("mouthSmileRight", 0),
        )
        jaw_open = blendshapes.get("jawOpen", 0)
        brow_up = max(
            blendshapes.get("browOuterUpLeft", 0),
            blendshapes.get("browOuterUpRight", 0),
        )
        blink_l = blendshapes.get("eyeBlinkLeft", 0)
        blink_r = blendshapes.get("eyeBlinkRight", 0)
        frown = max(
            blendshapes.get("mouthFrownLeft", 0),
            blendshapes.get("mouthFrownRight", 0),
        )

        # Both eyes closed
        if blink_l > self._BLINK_THRESHOLD and blink_r > self._BLINK_THRESHOLD:
            return "eyes_closed"
        # Surprise: jaw open + brows raised
        if jaw_open > self._JAW_OPEN_THRESHOLD and brow_up > self._BROW_UP_THRESHOLD:
            return "surprised"
        # Talking: jaw open but no surprise brows
        if jaw_open > self._JAW_OPEN_THRESHOLD:
            return "talking"
        # Smiling
        if smile > self._SMILE_THRESHOLD:
            return "smiling"
        # Frowning
        if frown > self._FROWN_THRESHOLD:
            return "frowning"

        return "neutral"

    def is_available(self) -> bool:
        """Check if MediaPipe Face Landmarker can be loaded."""
        return not self._load_failed


# ============================================================================
# MediaPipe Pose Analyzer (Body Pose + Gesture Recognition)
# ============================================================================


class MediaPipePoseAnalyzer:
    """
    Body pose detection and gesture recognition using MediaPipe Pose Landmarker.

    Detects 33 body landmarks per person and classifies gestures
    (waving, hand raised, etc.) from relative landmark positions.
    Downloads the model on first use. Runs on CPU, ~15ms per frame.
    """

    _MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/latest/"
        "pose_landmarker_lite.task"
    )
    _MODEL_FILENAME = "pose_landmarker_lite.task"

    # Landmark indices (MediaPipe Pose 33-point model)
    _LEFT_SHOULDER = 11
    _RIGHT_SHOULDER = 12
    _LEFT_ELBOW = 13
    _RIGHT_ELBOW = 14
    _LEFT_WRIST = 15
    _RIGHT_WRIST = 16
    _LEFT_HIP = 23
    _RIGHT_HIP = 24

    # Thresholds
    _VISIBILITY_THRESHOLD = 0.5
    _WAVE_Y_MARGIN = 0.08  # wrist must be this much above shoulder (normalized)
    _WAVE_X_MARGIN = 0.10  # wrist must be this far laterally from shoulder

    def __init__(self, min_detection_confidence: float = 0.5):
        self.min_detection_confidence = min_detection_confidence
        self._detector = None
        self._load_failed = False

    def _get_model_path(self) -> str:
        """Get path to pose landmarker model, downloading if needed."""
        import os
        import urllib.request

        runtime_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "runtime"
        )
        runtime_dir = os.path.normpath(runtime_dir)
        os.makedirs(runtime_dir, exist_ok=True)

        model_path = os.path.join(runtime_dir, self._MODEL_FILENAME)

        if not os.path.exists(model_path):
            logger.info("Downloading MediaPipe pose landmarker model...")
            urllib.request.urlretrieve(self._MODEL_URL, model_path)
            logger.info(f"Model saved to {model_path}")

        return model_path

    def _load_model(self) -> None:
        """Lazy-load MediaPipe Pose Landmarker (Tasks API)."""
        if self._load_failed:
            raise RuntimeError(
                "MediaPipe Pose Landmarker unavailable (previous load failed)"
            )
        if self._detector is None:
            try:
                import mediapipe as mp

                model_path = self._get_model_path()

                base_options = mp.tasks.BaseOptions(
                    model_asset_path=model_path
                )
                options = mp.tasks.vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    num_poses=3,
                    min_pose_detection_confidence=self.min_detection_confidence,
                )
                self._detector = (
                    mp.tasks.vision.PoseLandmarker.create_from_options(options)
                )
                logger.info("MediaPipe Pose Landmarker loaded")
            except Exception as e:
                self._load_failed = True
                logger.error(f"Failed to load MediaPipe Pose Landmarker: {e}")
                raise

    def detect_gesture(self, frame: np.ndarray) -> tuple[str, bool, int, float]:
        """
        Detect body pose and classify gesture.

        Returns:
            Tuple of (gesture, body_visible, num_people, processing_time_ms).
            gesture is one of: "waving", "hand_raised", "both_hands_raised", "none".
        """
        import mediapipe as mp

        self._load_model()
        start = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)
        elapsed_ms = (time.time() - start) * 1000

        if not result.pose_landmarks:
            return "none", False, 0, elapsed_ms

        num_people = len(result.pose_landmarks)
        # Classify gesture from the first (closest) person
        landmarks = result.pose_landmarks[0]
        gesture = self._classify_gesture(landmarks)

        return gesture, True, num_people, elapsed_ms

    def _classify_gesture(self, landmarks) -> str:
        """Classify gesture from pose landmarks.

        In normalized image coordinates: y=0 is top, y=1 is bottom.
        So 'above' means smaller y value.
        """
        left_raised, l_lateral = self._check_hand_raised(
            landmarks, self._LEFT_WRIST, self._LEFT_SHOULDER
        )
        right_raised, r_lateral = self._check_hand_raised(
            landmarks, self._RIGHT_WRIST, self._RIGHT_SHOULDER
        )

        if left_raised and right_raised:
            return "both_hands_raised"

        if left_raised:
            return "waving" if l_lateral > self._WAVE_X_MARGIN else "hand_raised"

        if right_raised:
            return "waving" if r_lateral > self._WAVE_X_MARGIN else "hand_raised"

        return "none"

    def _check_hand_raised(
        self, landmarks, wrist_idx: int, shoulder_idx: int
    ) -> tuple[bool, float]:
        """Check if a hand is raised above its shoulder.

        Returns (is_raised, lateral_distance).
        """
        wrist = landmarks[wrist_idx]
        shoulder = landmarks[shoulder_idx]

        if (
            wrist.visibility < self._VISIBILITY_THRESHOLD
            or shoulder.visibility < self._VISIBILITY_THRESHOLD
        ):
            return False, 0.0

        raised = wrist.y < shoulder.y - self._WAVE_Y_MARGIN
        lateral = abs(wrist.x - shoulder.x)
        return raised, lateral

    def is_available(self) -> bool:
        """Check if MediaPipe Pose Landmarker can be loaded."""
        return not self._load_failed


# ============================================================================
# Composite Analyzer (YOLO-World + MediaPipe Face)
# ============================================================================


class CompositeAnalyzer:
    """
    Multi-model composite: YOLO-World + MediaPipe Face.

    Runs both sub-analyzers sequentially on each frame and merges
    results into a single SceneDescription. Each sub-analyzer can
    fail independently without blocking the other.
    """

    def __init__(
        self,
        yolo_world: YOLOWorldAnalyzer | None = None,
        face_analyzer: MediaPipeFaceAnalyzer | None = None,
    ):
        self.yolo_world = yolo_world or YOLOWorldAnalyzer()
        self.face_analyzer = face_analyzer or MediaPipeFaceAnalyzer()
        self._yolo_world_failed = False
        self._face_failed = False

    def analyze(self, frame: np.ndarray) -> SceneDescription:
        """Run composite multi-model analysis on a frame."""
        start = time.time()
        all_objects: list[DetectedObject] = []
        all_labels: list[str] = []
        descriptions: list[str] = []

        # Step 1: YOLO-World open-vocabulary detection
        if not self._yolo_world_failed:
            try:
                yolo_result = self.yolo_world.analyze(frame)
                all_objects.extend(yolo_result.detected_objects)
                for label in yolo_result.object_labels:
                    if label not in all_labels:
                        all_labels.append(label)
                if yolo_result.description != "No notable objects detected":
                    descriptions.append(yolo_result.description)
            except (RuntimeError, ImportError):
                logger.warning(
                    "YOLO-World unavailable, skipping object detection"
                )
                self._yolo_world_failed = True
            except Exception as e:
                logger.error(f"YOLO-World error: {e}")

        # Step 2: MediaPipe face detection + expressions
        if not self._face_failed:
            try:
                face_result = self.face_analyzer.analyze(frame)
                all_objects.extend(face_result.detected_objects)
                for label in face_result.object_labels:
                    if label not in all_labels:
                        all_labels.append(label)
                if face_result.description != "No faces detected":
                    descriptions.append(face_result.description)
            except (RuntimeError, ImportError):
                logger.warning(
                    "MediaPipe unavailable, skipping face detection"
                )
                self._face_failed = True
            except Exception as e:
                logger.error(f"MediaPipe error: {e}")

        elapsed_ms = (time.time() - start) * 1000

        if descriptions:
            merged_description = "; ".join(descriptions)
        else:
            merged_description = "No notable objects or faces detected"

        return SceneDescription(
            timestamp=time.time(),
            description=merged_description,
            detected_objects=all_objects,
            object_labels=all_labels,
            analysis_method="composite",
            processing_time_ms=elapsed_ms,
        )

    def is_available(self) -> bool:
        """Available if at least one sub-analyzer works."""
        return not self._yolo_world_failed or not self._face_failed
