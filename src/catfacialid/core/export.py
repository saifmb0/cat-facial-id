"""ONNX export functionality for Cat Facial ID models.

This module provides utilities for exporting trained models to ONNX format
for deployment interoperability across different inference frameworks.

Example:
    ```python
    from catfacialid.core.export import ONNXExporter

    # Export preprocessor
    exporter = ONNXExporter()
    exporter.export_preprocessor(preprocessor, "preprocessor.onnx")

    # Export with metadata
    exporter.export_preprocessor(
        preprocessor,
        "preprocessor.onnx",
        metadata={"version": "1.0.0", "author": "CatFacialID"}
    )
    ```
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import onnx
    from onnx import helper, numpy_helper, TensorProto

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import onnxruntime as ort

    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False


class ONNXExportError(Exception):
    """Exception raised when ONNX export fails."""

    pass


class ONNXExporter:
    """Export Cat Facial ID models to ONNX format.

    This exporter creates ONNX models from trained preprocessing components,
    enabling deployment with various inference engines (ONNX Runtime, TensorRT,
    OpenVINO, etc.).

    Attributes:
        opset_version: ONNX opset version to use (default: 13).

    Example:
        ```python
        exporter = ONNXExporter(opset_version=13)
        exporter.export_preprocessor(preprocessor, "model.onnx")
        ```
    """

    def __init__(self, opset_version: int = 13) -> None:
        """Initialize ONNX exporter.

        Args:
            opset_version: ONNX opset version (default: 13).

        Raises:
            ImportError: If ONNX is not installed.
        """
        if not ONNX_AVAILABLE:
            raise ImportError(
                "ONNX required for export. Install with: pip install onnx"
            )
        self.opset_version = opset_version

    def export_preprocessor(
        self,
        preprocessor: Any,
        output_path: Path,
        input_shape: Optional[Tuple[int, ...]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Path:
        """Export feature preprocessor to ONNX.

        This exports the linear transformations (PCA, LDA, ICA) as matrix
        multiplications that can be efficiently executed in ONNX Runtime.

        Args:
            preprocessor: Fitted FeaturePreprocessor instance.
            output_path: Path to save ONNX model.
            input_shape: Optional input shape (batch_size, feature_dim).
                If None, uses dynamic batch size.
            metadata: Optional metadata to embed in model.

        Returns:
            Path to saved ONNX model.

        Raises:
            ONNXExportError: If preprocessor is not fitted or export fails.
        """
        output_path = Path(output_path)

        # Validate preprocessor
        if not hasattr(preprocessor, "pca") or preprocessor.pca is None:
            raise ONNXExportError(
                "Preprocessor must be fitted before export. Call fit() first."
            )

        # Get transformation matrices
        transforms = self._extract_transforms(preprocessor)

        # Build ONNX graph
        graph = self._build_preprocessing_graph(transforms, input_shape)

        # Create model
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", self.opset_version)],
        )

        # Add metadata
        if metadata:
            for key, value in metadata.items():
                meta = model.metadata_props.add()
                meta.key = key
                meta.value = str(value)

        # Add standard metadata
        model.doc_string = "Cat Facial ID Preprocessor"
        model.model_version = 1
        model.producer_name = "catfacialid"

        # Validate and save
        onnx.checker.check_model(model)
        onnx.save(model, str(output_path))

        return output_path

    def _extract_transforms(self, preprocessor: Any) -> Dict[str, np.ndarray]:
        """Extract transformation matrices from preprocessor.

        Args:
            preprocessor: Fitted FeaturePreprocessor.

        Returns:
            Dictionary of transformation matrices.
        """
        transforms: Dict[str, np.ndarray] = {}

        # PCA components
        if hasattr(preprocessor, "pca") and preprocessor.pca is not None:
            transforms["pca_components"] = preprocessor.pca.components_.T.astype(
                np.float32
            )
            transforms["pca_mean"] = preprocessor.pca.mean_.astype(np.float32)

        # LDA components (if fitted)
        if hasattr(preprocessor, "lda") and preprocessor.lda is not None:
            transforms["lda_scalings"] = preprocessor.lda.scalings_.astype(np.float32)
            transforms["lda_xbar"] = preprocessor.lda.xbar_.astype(np.float32)

        # ICA components
        if hasattr(preprocessor, "ica") and preprocessor.ica is not None:
            transforms["ica_components"] = preprocessor.ica.components_.T.astype(
                np.float32
            )
            transforms["ica_mean"] = preprocessor.ica.mean_.astype(np.float32)

        return transforms

    def _build_preprocessing_graph(
        self,
        transforms: Dict[str, np.ndarray],
        input_shape: Optional[Tuple[int, ...]],
    ) -> Any:
        """Build ONNX graph for preprocessing pipeline.

        Args:
            transforms: Dictionary of transformation matrices.
            input_shape: Optional input shape.

        Returns:
            ONNX GraphProto.
        """
        # Input
        if input_shape:
            batch_size, feature_dim = input_shape
        else:
            batch_size = None  # Dynamic
            feature_dim = transforms["pca_mean"].shape[0]

        input_tensor = helper.make_tensor_value_info(
            "input",
            TensorProto.FLOAT,
            [batch_size, feature_dim],
        )

        # Nodes and initializers
        nodes: List[Any] = []
        initializers: List[Any] = []
        intermediate_outputs: List[str] = []

        current_input = "input"

        # PCA transform: (X - mean) @ components
        if "pca_components" in transforms:
            # Subtract mean
            initializers.append(
                numpy_helper.from_array(transforms["pca_mean"], "pca_mean")
            )
            nodes.append(
                helper.make_node("Sub", [current_input, "pca_mean"], ["pca_centered"])
            )

            # Matrix multiply
            initializers.append(
                numpy_helper.from_array(transforms["pca_components"], "pca_components")
            )
            nodes.append(
                helper.make_node(
                    "MatMul", ["pca_centered", "pca_components"], ["pca_output"]
                )
            )
            intermediate_outputs.append("pca_output")

        # LDA transform
        if "lda_scalings" in transforms:
            initializers.append(
                numpy_helper.from_array(transforms["lda_xbar"], "lda_xbar")
            )
            nodes.append(
                helper.make_node("Sub", [current_input, "lda_xbar"], ["lda_centered"])
            )

            initializers.append(
                numpy_helper.from_array(transforms["lda_scalings"], "lda_scalings")
            )
            nodes.append(
                helper.make_node(
                    "MatMul", ["lda_centered", "lda_scalings"], ["lda_output"]
                )
            )
            intermediate_outputs.append("lda_output")

        # ICA transform
        if "ica_components" in transforms:
            initializers.append(
                numpy_helper.from_array(transforms["ica_mean"], "ica_mean")
            )
            nodes.append(
                helper.make_node("Sub", [current_input, "ica_mean"], ["ica_centered"])
            )

            initializers.append(
                numpy_helper.from_array(transforms["ica_components"], "ica_components")
            )
            nodes.append(
                helper.make_node(
                    "MatMul", ["ica_centered", "ica_components"], ["ica_output"]
                )
            )
            intermediate_outputs.append("ica_output")

        # Concatenate all outputs
        if len(intermediate_outputs) > 1:
            nodes.append(
                helper.make_node(
                    "Concat",
                    intermediate_outputs,
                    ["concat_output"],
                    axis=1,
                )
            )
            final_output = "concat_output"
        else:
            final_output = intermediate_outputs[0]

        # L2 Normalization
        # ||x||_2
        nodes.append(
            helper.make_node(
                "ReduceL2",
                [final_output],
                ["l2_norm"],
                axes=[1],
                keepdims=1,
            )
        )
        # Epsilon to prevent division by zero
        epsilon = numpy_helper.from_array(
            np.array([1e-12], dtype=np.float32), "epsilon"
        )
        initializers.append(epsilon)
        nodes.append(
            helper.make_node("Max", ["l2_norm", "epsilon"], ["l2_norm_safe"])
        )
        nodes.append(
            helper.make_node("Div", [final_output, "l2_norm_safe"], ["output"])
        )

        # Output
        output_tensor = helper.make_tensor_value_info(
            "output",
            TensorProto.FLOAT,
            [batch_size, None],  # Dynamic output dimension
        )

        # Build graph
        graph = helper.make_graph(
            nodes,
            "CatFacialIDPreprocessor",
            [input_tensor],
            [output_tensor],
            initializers,
        )

        return graph

    def verify_export(
        self,
        onnx_path: Path,
        preprocessor: Any,
        test_input: Optional[np.ndarray] = None,
        rtol: float = 1e-4,
        atol: float = 1e-5,
    ) -> bool:
        """Verify ONNX export matches original preprocessor output.

        Args:
            onnx_path: Path to exported ONNX model.
            preprocessor: Original fitted preprocessor.
            test_input: Test input array. If None, generates random input.
            rtol: Relative tolerance for comparison.
            atol: Absolute tolerance for comparison.

        Returns:
            True if outputs match within tolerance.

        Raises:
            ImportError: If ONNX Runtime is not installed.
            ONNXExportError: If verification fails.
        """
        if not ONNXRUNTIME_AVAILABLE:
            raise ImportError(
                "ONNX Runtime required for verification. "
                "Install with: pip install onnxruntime"
            )

        # Load ONNX model
        session = ort.InferenceSession(str(onnx_path))

        # Generate test input if needed
        if test_input is None:
            input_dim = preprocessor.pca.mean_.shape[0]
            test_input = np.random.randn(10, input_dim).astype(np.float32)

        # Run original preprocessor
        original_output = preprocessor.transform(test_input)

        # Run ONNX model
        onnx_output = session.run(None, {"input": test_input})[0]

        # Compare
        try:
            np.testing.assert_allclose(
                original_output,
                onnx_output,
                rtol=rtol,
                atol=atol,
            )
            return True
        except AssertionError as e:
            raise ONNXExportError(f"Verification failed: {e}")


def export_to_onnx(
    preprocessor: Any,
    output_path: Path,
    verify: bool = True,
    metadata: Optional[Dict[str, str]] = None,
) -> Path:
    """Convenience function to export preprocessor to ONNX.

    Args:
        preprocessor: Fitted FeaturePreprocessor instance.
        output_path: Path to save ONNX model.
        verify: Whether to verify export (requires ONNX Runtime).
        metadata: Optional metadata to embed.

    Returns:
        Path to saved ONNX model.

    Example:
        ```python
        from catfacialid.core.export import export_to_onnx

        path = export_to_onnx(
            preprocessor,
            "model.onnx",
            verify=True,
            metadata={"version": "1.0.0"}
        )
        print(f"Exported to {path}")
        ```
    """
    exporter = ONNXExporter()
    result_path = exporter.export_preprocessor(
        preprocessor,
        output_path,
        metadata=metadata,
    )

    if verify and ONNXRUNTIME_AVAILABLE:
        exporter.verify_export(result_path, preprocessor)

    return result_path
