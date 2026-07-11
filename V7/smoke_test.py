"""Small synthetic shape/mask/gradient/checkpoint test for Experiment A."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import pandas as pd
from torch.utils.data import Dataset

from V7.experiment_a_model import Stage2ExpertEmotionAdaptationModel
from V7.temporal_aggregator import TemporalTrialAggregator
from V7.trial_sequence import TrialSequenceDataset, trial_sequence_collate


class _FakeWindows(Dataset):
    def __init__(self) -> None:
        self.df = pd.DataFrame({
            "subject_id": [1, 1, 2, 1, 2], "trial_id": [4, 4, 7, 4, 7],
            "start": [20, 0, 10, 10, 0], "label4": [1, 1, 2, 1, 2],
            "emotion_label": [1, 1, 0, 1, 0], "diagnosis_label": [0, 0, 1, 0, 1],
            "domain_id": [0, 0, 1, 0, 1],
        })

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return {
            "x": torch.full((30, 20), float(row.start)),
            "de_feat": torch.zeros(30, 5), "label4": torch.tensor(int(row.label4)),
            "emotion_label": torch.tensor(int(row.emotion_label)),
            "diagnosis_label": torch.tensor(int(row.diagnosis_label)),
            "subject_id": torch.tensor(int(row.subject_id)), "domain_id": torch.tensor(int(row.domain_id)),
            "trial_id": torch.tensor(int(row.trial_id)), "user_id": str(int(row.subject_id)),
            "target_key": f"fake:{int(row.subject_id)}",
        }


def main() -> None:
    torch.manual_seed(1)
    trials = TrialSequenceDataset(_FakeWindows(), trial_num_windows=0, name="smoke-order")
    first, second = trials[0], trials[1]
    assert first["window_start"].tolist() == [0, 10, 20]
    assert second["window_start"].tolist() == [0, 10]
    padded = trial_sequence_collate([first, second])
    assert padded["window_mask"].tolist() == [[True, True, True], [True, True, False]]
    print("ordered_window_indices", first["window_indices"].tolist(), "starts", first["window_start"].tolist())
    aggregator = TemporalTrialAggregator(185, 32, 185, dropout=0.1)
    sequence = torch.randn(2, 4, 185, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    end, weight, features = aggregator(sequence, mask)
    assert end.shape == (2, 185) and weight.shape == (2, 4) and features.shape == (2, 4, 32)
    assert torch.equal(weight[~mask], torch.zeros_like(weight[~mask]))
    assert torch.allclose(weight.sum(1), torch.ones(2), atol=1e-6) and torch.isfinite(end).all()

    model_args = dict(
        num_domains=3, sfreq=250, topk=4, dropout=0.1, use_biomarkers=False,
        use_subject_relative_de=False, use_subject_relative_bio=False,
        temporal_hidden_dim=32, temporal_dropout=0.1,
    )
    model = Stage2ExpertEmotionAdaptationModel(**model_args)
    x = torch.randn(2, 3, 30, 500)
    de = torch.randn(2, 3, 30, 5)
    trial_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    out = model(x, de, trial_mask, lambda_subject=0.01)
    feature_dim = model.in_dim
    expected = {
        "z_emotion_seq": (2, 3, feature_dim), "z_diag_seq": (2, 3, feature_dim),
        "z_end_emotion": (2, feature_dim), "z_end_diag": (2, feature_dim),
        "temporal_attention_emotion": (2, 3), "temporal_attention_diag": (2, 3),
    }
    for key, shape in expected.items():
        assert tuple(out[key].shape) == shape, (key, out[key].shape)
    assert torch.equal(out["temporal_attention_emotion"][~trial_mask], torch.zeros(1))
    assert torch.equal(out["temporal_attention_diag"][~trial_mask], torch.zeros(1))
    loss = torch.nn.functional.nll_loss(
        torch.log(out["mix_prob"].clamp_min(1e-8)), torch.tensor([0, 1])
    ) + torch.nn.functional.cross_entropy(out["diag_logits"], torch.tensor([0, 1]))
    loss.backward()
    groups = {
        "backbone": model.shared_encoder.backbone,
        "emotion_temporal": model.shared_encoder.emotion_temporal_aggregator,
        "diagnosis_temporal": model.shared_encoder.diagnosis_temporal_aggregator,
        "stage2_head": model.shared_emotion_head,
    }
    for name, module in groups.items():
        nonzero = sum(
            int(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
            for p in module.parameters()
        )
        assert nonzero > 0, (name, nonzero)
        print(name, "nonzero_grad_params", nonzero)
    clone = Stage2ExpertEmotionAdaptationModel(**model_args)
    loaded = clone.load_state_dict(model.state_dict(), strict=False)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    print("shape_mask_gradient_checkpoint_smoke=PASS", float(loss.detach()))


if __name__ == "__main__":
    main()
