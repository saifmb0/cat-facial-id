"""Core module for feature extraction and model inference."""

from .inference import FAISSIndex, PredictionEngine
from .preprocessing import DimensionalityReducer, FeatureExtractor, FeatureFuser

# Optional ONNX export (requires onnx package)
try:
    from .export import ONNXExporter, ONNXExportError, export_to_onnx

    _ONNX_EXPORTS = ["ONNXExporter", "export_to_onnx", "ONNXExportError"]
except ImportError:
    _ONNX_EXPORTS = []

__all__ = [
    "FeatureExtractor",
    "FeatureFuser",
    "DimensionalityReducer",
    "FAISSIndex",
    "PredictionEngine",
    *_ONNX_EXPORTS,
]
