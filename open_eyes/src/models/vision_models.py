"""
Data models for vision processing.

Pydantic models for data contracts between pipeline stages.
"""

# Standard library
from enum import Enum
from typing import List, Optional

# Third-party
from pydantic import BaseModel, ConfigDict, Field


class ObservationType(str, Enum):
    """Types of visual observations."""

    SCENE_CHANGE = "SCENE_CHANGE"
    PERSON_DETECTED = "PERSON_DETECTED"
    PERSON_LEFT = "PERSON_LEFT"
    OBJECT_CHANGE = "OBJECT_CHANGE"
    CONTEXT_RELEVANT = "CONTEXT_RELEVANT"
    ENVIRONMENT_CHANGE = "ENVIRONMENT_CHANGE"
    SAFETY_ALERT = "SAFETY_ALERT"


class ObservationTier(str, Enum):
    """Tier controlling cooldown cadence and output section."""

    SCENE = "SCENE"         # Tier 1: environment description (~3 min cooldown)
    ACTIVITY = "ACTIVITY"   # Tier 2: what's happening (~15s cooldown)
    EVENT = "EVENT"         # Tier 3: immediate notable events (~5s cooldown)
    FACE = "FACE"           # Tier 4: expression/sentiment (~1-2s, independent)


class FaceLandmark(BaseModel):
    """A single facial landmark point (normalized 0-1)."""

    x: float = Field(..., description="Normalized x coordinate (0-1)")
    y: float = Field(..., description="Normalized y coordinate (0-1)")
    z: float = Field(0.0, description="Normalized z depth (0-1)")


class DetectedObject(BaseModel):
    """A single detected object in a frame."""

    model_config = ConfigDict(frozen=False)

    label: str = Field(..., description="Object class label")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence"
    )
    bbox: tuple[int, int, int, int] = Field(
        ..., description="Bounding box (x1, y1, x2, y2)"
    )
    area_fraction: float = Field(
        0.0, description="Fraction of frame area occupied"
    )
    landmarks: Optional[List[FaceLandmark]] = Field(
        None, description="Face landmarks (478 points, normalized 0-1)"
    )
    blendshapes: Optional[dict[str, float]] = Field(
        None, description="Face blendshape scores (52 values, 0-1)"
    )


class CapturedFrame(BaseModel):
    """A captured video frame with metadata."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    frame_number: int = Field(..., description="Sequential frame number")
    timestamp: float = Field(..., description="Capture timestamp (epoch)")
    resolution: tuple[int, int] = Field(..., description="Width x Height")
    change_magnitude: float = Field(
        0.0, ge=0.0, le=1.0, description="How much the scene changed (0-1)"
    )


class FaceState(BaseModel):
    """Current face detection / expression state from the face tracker."""

    model_config = ConfigDict(frozen=False)

    timestamp: float = Field(..., description="Detection timestamp (epoch)")
    num_faces: int = Field(0, description="Number of faces detected")
    primary_expression: str = Field(
        "none", description="Dominant expression of the closest face"
    )
    expressions: List[str] = Field(
        default_factory=list, description="Expressions for all detected faces"
    )
    blendshapes: Optional[dict[str, float]] = Field(
        None, description="Raw blendshape scores for primary face"
    )
    processing_time_ms: float = Field(
        0.0, description="Face analysis latency in milliseconds"
    )


class PoseState(BaseModel):
    """Current body pose / gesture state from the pose tracker."""

    model_config = ConfigDict(frozen=False)

    timestamp: float = Field(..., description="Detection timestamp (epoch)")
    gesture: str = Field(
        "none", description="Classified gesture (waving, hand_raised, etc.)"
    )
    body_visible: bool = Field(False, description="Whether a body is detected")
    num_people: int = Field(0, description="Number of people with pose detected")
    processing_time_ms: float = Field(
        0.0, description="Pose analysis latency in milliseconds"
    )


class SceneDescription(BaseModel):
    """Result of scene analysis."""

    model_config = ConfigDict(frozen=False)

    timestamp: float = Field(..., description="Analysis timestamp")
    description: str = Field(
        ..., description="Natural language scene description"
    )
    detected_objects: List[DetectedObject] = Field(default_factory=list)
    object_labels: List[str] = Field(
        default_factory=list, description="Simple list of detected labels"
    )
    analysis_method: str = Field(
        ..., description="Which analyzer produced this"
    )
    processing_time_ms: float = Field(
        0.0, description="Analysis latency in milliseconds"
    )
    frame_number: int = Field(0, description="Source frame number")
    tier: Optional[ObservationTier] = Field(
        None, description="Observation tier (set by pipeline)"
    )


class RelevanceScore(BaseModel):
    """Result of relevance evaluation."""

    model_config = ConfigDict(frozen=False)

    overall_score: float = Field(
        ..., ge=0.0, le=1.0, description="Combined relevance score"
    )
    novelty: float = Field(
        0.0, ge=0.0, le=1.0, description="How novel is this observation"
    )
    context_match: float = Field(
        0.0, ge=0.0, le=1.0, description="Relevance to conversation"
    )
    intrinsic_interest: float = Field(
        0.0, ge=0.0, le=1.0, description="Inherent noteworthiness"
    )
    timing: float = Field(
        0.0, ge=0.0, le=1.0, description="Is now a good time"
    )
    should_report: bool = Field(False, description="Final reporting decision")
    reason: str = Field("", description="Human-readable explanation")


class VisualObservation(BaseModel):
    """A complete visual observation ready for the agent."""

    model_config = ConfigDict(frozen=False)

    timestamp: str = Field(..., description="ISO timestamp")
    observation_type: ObservationType = Field(
        ..., description="Category of observation"
    )
    description: str = Field(
        ..., description="Natural language description for agent"
    )
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    scene_context: Optional[str] = Field(
        None, description="Brief scene summary"
    )
    was_reported: bool = Field(
        False, description="Whether this was sent to the agent"
    )
