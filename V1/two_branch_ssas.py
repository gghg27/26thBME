# -*- coding: utf-8 -*-
from two_branch import *  # noqa: F401,F403


_BaseTwoBranchModel = TwoBranchModel


class TwoBranchModel(_BaseTwoBranchModel):
    """
    SSAS final model.

    The original subject-domain head is kept for checkpoint compatibility, but
    Stage 2 uses ssas_domain_logits: binary source/target logits with GRL.
    """

    def __init__(self, *args, ssas_domain_hidden_dim: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        self.ssas_domain_head = SubjectDomainHead(
            in_dim=self.diag_dim,
            num_subjects=2,
            hidden_dim=ssas_domain_hidden_dim,
            dropout=kwargs.get("dropout", 0.2),
        )

    def forward(
            self,
            x: torch.Tensor,
            de_feat: torch.Tensor,
            lambda_dom: float = 0.0,
            dataset_name: str = "comp4",
            **kwargs,
    ) -> Dict[str, torch.Tensor]:
        out = super().forward(
            x=x,
            de_feat=de_feat,
            lambda_dom=0.0,
            dataset_name=dataset_name,
            **kwargs,
        )
        out["ssas_domain_logits"] = self.ssas_domain_head(out["z_diag"], lambda_grl=lambda_dom)
        return out


class SourceSelectionModel(_BaseTwoBranchModel):
    """
    Stage 1 model for SSAS source selection.

    It reuses the V1 two-branch backbone and emotion heads, then adds a binary
    source/target domain classifier on z_diag. The final Stage 2 model still
    uses TwoBranchModel.
    """

    def __init__(self, *args, ss_domain_hidden_dim: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        self.ss_domain_head = ClassificationHead(
            in_dim=self.diag_dim,
            num_classes=2,
            hidden_dim=ss_domain_hidden_dim,
            dropout=kwargs.get("dropout", 0.2),
        )

    def forward(
            self,
            x: torch.Tensor,
            de_feat: torch.Tensor,
            lambda_dom: float = 0.0,
            dataset_name: str = "comp4",
            **kwargs,
    ) -> Dict[str, torch.Tensor]:
        out = super().forward(
            x=x,
            de_feat=de_feat,
            lambda_dom=0.0,
            dataset_name=dataset_name,
            **kwargs,
        )
        out["ss_domain_logits"] = self.ss_domain_head(out["z_diag"])
        return out


def mmd_loss(
        source_feat: torch.Tensor,
        target_feat: torch.Tensor,
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
        fix_sigma: float | None = None,
) -> torch.Tensor:
    if source_feat.numel() == 0 or target_feat.numel() == 0:
        return source_feat.new_tensor(0.0)

    source_feat = source_feat.flatten(1)
    target_feat = target_feat.flatten(1)
    total = torch.cat([source_feat, target_feat], dim=0)
    total0 = total.unsqueeze(0)
    total1 = total.unsqueeze(1)
    l2_distance = ((total0 - total1) ** 2).sum(dim=2)

    if fix_sigma is None:
        denom = max(total.size(0) ** 2 - total.size(0), 1)
        bandwidth = l2_distance.detach().sum() / denom
    else:
        bandwidth = source_feat.new_tensor(float(fix_sigma))

    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernels = sum(torch.exp(-l2_distance / (bw + 1e-8)) for bw in bandwidth_list)

    ns = source_feat.size(0)
    nt = target_feat.size(0)
    xx = kernels[:ns, :ns]
    yy = kernels[ns:, ns:]
    xy = kernels[:ns, ns:]
    yx = kernels[ns:, :ns]
    return xx.mean() + yy.mean() - xy.mean() - yx.mean()
