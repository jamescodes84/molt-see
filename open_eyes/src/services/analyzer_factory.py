"""
Scene analyzer factory.

Creates scene analyzer instances based on configuration.
Supports YOLO, VLM, and Cascade modes.
"""

import logging
from typing import Union

from ..config import settings
from .scene_analyzer import (
    CascadeAnalyzer,
    CompositeAnalyzer,
    Florence2Analyzer,
    MediaPipeFaceAnalyzer,
    MoondreamAnalyzer,
    VLMAnalyzer,
    YOLOAnalyzer,
    YOLOWorldAnalyzer,
)

logger = logging.getLogger(__name__)


def create_analyzer(
    mode: str = settings.ANALYZER_MODE,
) -> Union[YOLOAnalyzer, VLMAnalyzer, CascadeAnalyzer, Florence2Analyzer, CompositeAnalyzer]:
    """
    Create a scene analyzer based on the configured mode.

    Args:
        mode: Analyzer mode ("yolo", "vlm", "cascade").

    Returns:
        Configured scene analyzer instance.

    Raises:
        ValueError: If mode is not recognized.
    """
    if mode == "yolo":
        logger.info(f"Creating YOLO analyzer: model={settings.YOLO_MODEL}")
        return YOLOAnalyzer(
            model_size=settings.YOLO_MODEL,
            confidence_threshold=settings.YOLO_CONFIDENCE,
        )

    elif mode == "vlm":
        logger.info(
            f"Creating VLM analyzer: provider={settings.VLM_PROVIDER}, "
            f"model={settings.VLM_MODEL}"
        )
        analyzer = VLMAnalyzer(
            provider=settings.VLM_PROVIDER,
            model=settings.VLM_MODEL,
            api_key=settings.VLM_API_KEY,
            max_tokens=settings.VLM_MAX_TOKENS,
        )
        if not analyzer.is_available():
            logger.warning(
                "VLM API key not configured. "
                "Set ANTHROPIC_API_KEY or VLM_API_KEY in .env"
            )
        return analyzer

    elif mode == "cascade":
        logger.info(
            "Creating Cascade analyzer "
            "(YOLO-World + MediaPipe Face → Moondream)"
        )
        yolo_world = YOLOWorldAnalyzer(
            model_name=settings.YOLO_WORLD_MODEL,
            confidence_threshold=settings.YOLO_WORLD_CONFIDENCE,
            class_list=settings.YOLO_WORLD_CLASSES,
        )
        face = MediaPipeFaceAnalyzer(
            min_detection_confidence=settings.MEDIAPIPE_FACE_CONFIDENCE,
            max_num_faces=settings.MEDIAPIPE_MAX_FACES,
        )
        fast = CompositeAnalyzer(yolo_world=yolo_world, face_analyzer=face)
        detail = MoondreamAnalyzer(model_name=settings.MOONDREAM_MODEL)
        return CascadeAnalyzer(fast_analyzer=fast, detail_analyzer=detail)

    elif mode == "florence":
        logger.info(
            f"Creating Florence-2 analyzer: model={settings.FLORENCE_MODEL}"
        )
        return Florence2Analyzer(model_name=settings.FLORENCE_MODEL)

    elif mode == "composite":
        logger.info(
            f"Creating Composite analyzer: "
            f"YOLO-World ({settings.YOLO_WORLD_MODEL}) + MediaPipe Face"
        )
        yolo_world = YOLOWorldAnalyzer(
            model_name=settings.YOLO_WORLD_MODEL,
            confidence_threshold=settings.YOLO_WORLD_CONFIDENCE,
            class_list=settings.YOLO_WORLD_CLASSES,
        )
        face = MediaPipeFaceAnalyzer(
            min_detection_confidence=settings.MEDIAPIPE_FACE_CONFIDENCE,
            max_num_faces=settings.MEDIAPIPE_MAX_FACES,
        )
        return CompositeAnalyzer(yolo_world=yolo_world, face_analyzer=face)

    else:
        raise ValueError(
            f"Unknown analyzer mode: {mode}. "
            "Expected 'yolo', 'vlm', 'cascade', 'florence', or 'composite'."
        )
