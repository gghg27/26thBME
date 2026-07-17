"""Single source of truth for V9 data, model, and training settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


V9_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = V9_ROOT.parent

# These boundaries are copied from preprcocess/preprocess_test.py, which produced
# the existing DE files. Do not duplicate band definitions in model/dataset code.
FREQUENCY_BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
)


@dataclass
class BFGCNConfig:
    """Serializable configuration used by training, inference, and checkpoints."""

    train_index_csv: str = str(PROJECT_ROOT / "com_index_sub_2s.csv")
    test_index_csv: str = str(PROJECT_ROOT / "com_test_window_index_2s.csv")
    plv_cache_dir: str = str(V9_ROOT / "cache" / "plv")
    cache_plv: bool = True
    checkpoint_dir: str = str(V9_ROOT / "checkpoints")
    result_dir: str = str(V9_ROOT / "results")
    output_dir: str = str(V9_ROOT / "outputs")

    sampling_rate: int = 250
    window_length_samples: int = 500
    window_step_samples: int = 250
    target_trial_samples: int = 2500
    num_channels: int = 30
    frequency_bands: tuple[tuple[str, float, float], ...] = FREQUENCY_BANDS

    hidden_dim: int = 64
    classifier_dim: int = 64
    graph_order: int = 2
    graph_conv_type: str = "multi_order"
    pool_type: str = "mean_max"
    dropout: float = 0.3
    use_learnable_graph: bool = True
    use_functional_graph: bool = True
    use_common_branch: bool = True
    use_band_attention: bool = True
    use_branch_attention: bool = True

    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    label_smoothing: float = 0.05
    class_weight: list[float] | None = None
    auto_class_weight: bool = False
    early_stopping_patience: int = 15
    scheduler: str = "cosine"
    gradient_clip: float = 5.0
    amp: bool = True
    num_workers: int = 0
    n_splits: int = 10
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Return a checkpoint-safe plain dictionary."""
        data = asdict(self)
        data["frequency_bands"] = [list(x) for x in self.frequency_bands]
        return data

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "BFGCNConfig":
        """Recreate a config while tolerating future extra checkpoint fields."""
        names = cls.__dataclass_fields__.keys()
        clean = {key: value for key, value in values.items() if key in names}
        if "frequency_bands" in clean:
            clean["frequency_bands"] = tuple(tuple(x) for x in clean["frequency_bands"])
        return cls(**clean)


DEFAULT_CONFIG = BFGCNConfig()
