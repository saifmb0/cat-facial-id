"""Unit tests for configuration management."""

from catfacialid.config import (
    PreprocessingConfig,
    ModelConfig,
    DataConfig,
    SystemConfig,
)


class TestPreprocessingConfig:
    """Test suite for PreprocessingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PreprocessingConfig()
        assert config.pca_variance_threshold == 0.95
        assert config.ica_n_components == 200
        assert config.ica_max_iterations == 200
        assert config.random_seed == 42

    def test_custom_values(self):
        """Test setting custom values."""
        config = PreprocessingConfig(
            pca_variance_threshold=0.90,
            ica_n_components=150,
            random_seed=123,
        )
        assert config.pca_variance_threshold == 0.90
        assert config.ica_n_components == 150
        assert config.random_seed == 123


class TestModelConfig:
    """Test suite for ModelConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.num_classes == 500
        assert config.top_k_predictions == 3
        assert config.random_seed == 42

    def test_custom_values(self):
        """Test setting custom values."""
        config = ModelConfig(
            num_classes=1000,
            top_k_predictions=5,
            random_seed=999,
        )
        assert config.num_classes == 1000
        assert config.top_k_predictions == 5


class TestDataConfig:
    """Test suite for DataConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DataConfig()
        assert config.output_dir == "./outputs"
        assert config.batch_size == 32
        assert config.train_features_path is None
        assert config.test_features_path is None

    def test_custom_values(self):
        """Test setting custom values."""
        config = DataConfig(
            train_features_path="/path/to/train.pkl",
            test_features_path="/path/to/test.pkl",
            output_dir="/custom/output",
        )
        assert config.train_features_path == "/path/to/train.pkl"
        assert config.output_dir == "/custom/output"


class TestSystemConfig:
    """Test suite for SystemConfig."""

    def test_default_factory(self):
        """Test default configuration factory method."""
        config = SystemConfig.default()

        assert isinstance(config.preprocessing, PreprocessingConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.data, DataConfig)
        assert config.use_cuda is False
        assert config.verbose is True

    def test_custom_system_config(self):
        """Test creating custom system configuration."""
        preprocessing = PreprocessingConfig(pca_variance_threshold=0.9)
        model = ModelConfig(num_classes=1000)
        data = DataConfig(batch_size=64)

        config = SystemConfig(
            preprocessing=preprocessing,
            model=model,
            data=data,
            use_cuda=True,
            verbose=False,
        )

        assert config.preprocessing.pca_variance_threshold == 0.9
        assert config.model.num_classes == 1000
        assert config.data.batch_size == 64
        assert config.use_cuda is True
        assert config.verbose is False
