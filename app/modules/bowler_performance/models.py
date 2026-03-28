from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

YORKER_MAX = 2.0
FULL_MAX = 4.0
GOOD_LENGTH_MAX = 7.0
SHORT_OF_LENGTH_MAX = 9.0


class LengthClass(StrEnum):
    YORKER = "yorker"
    FULL = "full"
    GOOD_LENGTH = "good_length"
    SHORT_OF_LENGTH = "short_of_length"
    SHORT = "short"


class WicketRiskBand(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BouncePoint(BaseModel):
    x_metres: float
    z_metres: float


class TrajectoryPoint3D(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    frame_idx: float = Field(serialization_alias="frameIdx")
    x: float
    y: float
    z: float


class TrajectoryPointPitch(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    frame_idx: float = Field(serialization_alias="frameIdx")
    pitch_x: float = Field(serialization_alias="pitchX")
    pitch_z: float = Field(serialization_alias="pitchZ")


class BallTrackPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pitch_x: float | None = Field(default=None, serialization_alias="pitchX")
    pitch_y: float | None = Field(default=None, serialization_alias="pitchY")
    min_frame_idx: float = Field(serialization_alias="minFrameIdx")
    max_frame_idx: float = Field(serialization_alias="maxFrameIdx")
    parameter_x_array: list[float] = Field(serialization_alias="parameterXArray")
    parameter_y_array: list[float] = Field(serialization_alias="parameterYArray")
    parameter_z_array: list[float] = Field(serialization_alias="parameterZArray")
    deviation_array: list[float] = Field(serialization_alias="deviationArray")
    speed: float | None = None
    spin: float | None = None
    swing: float | None = None
    trajectory_mode: str | None = Field(default=None, serialization_alias="trajectoryMode")
    trajectory_points_3d: list[TrajectoryPoint3D] = Field(
        default_factory=list,
        serialization_alias="trajectoryPoints3D",
    )
    trajectory_points_pitch: list[TrajectoryPointPitch] = Field(
        default_factory=list,
        serialization_alias="trajectoryPointsPitch",
    )


class CameraCalibrationPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    principal_point_array: list[float] = Field(serialization_alias="principalPointArray")
    score: float
    translation_array: list[float] = Field(serialization_alias="translationArray")
    dimensions: list[int]
    distortion: float = 0.0
    instrinsic_matrix_array: list[float] = Field(serialization_alias="instrinsicMatrixArray")
    position_array: list[float] = Field(serialization_alias="positionArray")
    rotation_euler_array: list[float] = Field(serialization_alias="rotationEulerArray")
    focal: float
    rotation_matrix_array: list[float] = Field(serialization_alias="rotationMatrixArray")
    fovy: float


class DeliveryFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    batter_mode: str = Field(serialization_alias="batterMode")
    has_bat_contact: str = Field(serialization_alias="hasBatContact")
    contact_method: str = Field(serialization_alias="contactMethod")
    trajectory_reliable: str = Field(serialization_alias="trajectoryReliable")
    length_class: str | None = Field(default=None, serialization_alias="lengthClass")
    line_bucket: str | None = Field(default=None, serialization_alias="lineBucket")
    pace_bucket: str | None = Field(default=None, serialization_alias="paceBucket")
    fps: float
    release_frame_idx: float | None = Field(default=None, serialization_alias="releaseFrameIdx")
    bounce_frame_idx: float | None = Field(default=None, serialization_alias="bounceFrameIdx")
    contact_frame_idx: float | None = Field(default=None, serialization_alias="contactFrameIdx")
    release_timestamp_s: float | None = Field(
        default=None,
        serialization_alias="releaseTimestampS",
    )
    bounce_timestamp_s: float | None = Field(
        default=None,
        serialization_alias="bounceTimestampS",
    )
    contact_timestamp_s: float | None = Field(
        default=None,
        serialization_alias="contactTimestampS",
    )
    release_to_bounce_ms: float | None = Field(
        default=None,
        serialization_alias="releaseToBounceMs",
    )
    bounce_to_contact_ms: float | None = Field(
        default=None,
        serialization_alias="bounceToContactMs",
    )
    release_to_contact_ms: float | None = Field(
        default=None,
        serialization_alias="releaseToContactMs",
    )
    pre_bounce_detection_count: int = Field(serialization_alias="preBounceDetectionCount")
    post_bounce_detection_count: int = Field(serialization_alias="postBounceDetectionCount")
    selected_track_detection_count: int = Field(
        serialization_alias="selectedTrackDetectionCount"
    )
    inlier_count: int = Field(serialization_alias="inlierCount")
    bounce_pitch_x: float | None = Field(default=None, serialization_alias="bouncePitchX")
    bounce_pitch_y: float | None = Field(default=None, serialization_alias="bouncePitchY")
    tracking_confidence: float = Field(serialization_alias="trackingConfidence")
    release_confidence: float | None = Field(default=None, serialization_alias="releaseConfidence")
    contact_score: float | None = Field(default=None, serialization_alias="contactScore")
    release_height_m: float | None = Field(default=None, serialization_alias="releaseHeightM")
    contact_height_m: float | None = Field(default=None, serialization_alias="contactHeightM")
    release_pitch_x: float | None = Field(default=None, serialization_alias="releasePitchX")
    contact_pitch_x: float | None = Field(default=None, serialization_alias="contactPitchX")
    pre_bounce_lateral_delta: float | None = Field(
        default=None,
        serialization_alias="preBounceLateralDelta",
    )
    post_bounce_lateral_delta: float | None = Field(
        default=None,
        serialization_alias="postBounceLateralDelta",
    )
    approach_to_stumps: float | None = Field(
        default=None,
        serialization_alias="approachToStumps",
    )


class WicketRiskPrediction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    probability: float
    percentage: float
    risk_band: WicketRiskBand = Field(serialization_alias="riskBand")
    model_name: str = Field(serialization_alias="modelName")
    model_version: str = Field(serialization_alias="modelVersion")


class FlutterPayloadEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    video_url: str | None = Field(default=None, serialization_alias="videoURL")
    delivery_features: DeliveryFeatures | None = Field(
        default=None,
        serialization_alias="deliveryFeatures",
    )
    wicket_risk: WicketRiskPrediction | None = Field(
        default=None,
        serialization_alias="wicketRisk",
    )
    ball_track: BallTrackPayload | None = Field(default=None, serialization_alias="ballTrack")
    camera_calibration: CameraCalibrationPayload | None = Field(
        default=None,
        serialization_alias="cameraCalibration",
    )


class BowlerPerformanceResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    speed_kmh: float | None
    swing_metres: float | None
    bounce_point: BouncePoint | None
    length_class: LengthClass | None
    confidence: float
    inlier_count: int
    raw_speed_ms: float | None
    trajectory_reliable: bool = Field(
        default=True,
        serialization_alias="trajectoryReliable",
    )
    trajectory_warning: str | None = Field(
        default=None,
        serialization_alias="trajectoryWarning",
    )
    delivery_features: DeliveryFeatures | None = Field(
        default=None,
        serialization_alias="deliveryFeatures",
    )
    wicket_risk: WicketRiskPrediction | None = Field(
        default=None,
        serialization_alias="wicketRisk",
    )
    video_url: str | None = Field(default=None, serialization_alias="videoURL")
    ball_track: BallTrackPayload | None = Field(default=None, serialization_alias="ballTrack")
    camera_calibration: CameraCalibrationPayload | None = Field(
        default=None,
        serialization_alias="cameraCalibration",
    )
    flutter_payload: list[FlutterPayloadEntry] = Field(
        default_factory=list,
        serialization_alias="flutterPayload",
    )


def classify_length(z_metres: float) -> LengthClass:
    if z_metres <= YORKER_MAX:
        return LengthClass.YORKER
    if z_metres <= FULL_MAX:
        return LengthClass.FULL
    if z_metres <= GOOD_LENGTH_MAX:
        return LengthClass.GOOD_LENGTH
    if z_metres <= SHORT_OF_LENGTH_MAX:
        return LengthClass.SHORT_OF_LENGTH
    return LengthClass.SHORT
