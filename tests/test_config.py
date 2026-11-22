"""配置默认值测试。"""

import os

from chatlm.config import DpoConfig, InferConfig, SFTConfig, T5ModelConfig, TrainConfig


def test_infer_config_defaults() -> None:
    config = InferConfig()

    assert config.port == 8812
    assert config.max_seq_len == 320
    assert config.model_dir


def test_train_config_paths_are_absolute() -> None:
    config = TrainConfig()

    for path in (
        config.train_file,
        config.validation_file,
        config.test_file,
        config.model_file,
        config.train_state_dir,
        config.output_dir,
    ):
        assert os.path.isabs(path)
        assert "model_save" not in path


def test_sft_and_dpo_config_outputs_are_absolute() -> None:
    assert os.path.isabs(SFTConfig().output_dir)
    assert os.path.isabs(DpoConfig().output_dir)


def test_t5_model_config_dimensions() -> None:
    config = T5ModelConfig()

    assert config.d_model % config.num_heads == 0
    assert config.d_model // config.num_heads == config.d_kv
