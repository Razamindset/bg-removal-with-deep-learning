import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss that ignores pixels where valid == 0.

    Uses the sigmoid probabilities (no threshold), so gradients flow.
    The Dice *metric* thresholds the prediction, which has no gradient,
    so it can't be trained on.
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets, valid):
        # zero out ignored pixels in BOTH terms, so they add nothing
        # to the intersection or to either sum (and get zero gradient)
        probs = torch.sigmoid(logits) * valid
        targets = targets * valid

        dims = (1, 2, 3)  # Dice per image, then averaged over the batch
        intersection = (probs * targets).sum(dim=dims)
        denominator = probs.sum(dim=dims) + targets.sum(dim=dims)

        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        return (1.0 - dice).mean()


class BCEDiceLoss(nn.Module):
    """Weighted BCE + Dice, both ignoring pixels where valid == 0."""

    def __init__(self, bce_weight=1.0, dice_weight=1.0, eps=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss(eps)

    def forward(self, logits, targets, valid):
        # per-pixel BCE, masked, averaged over the valid pixels only
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        bce = (bce * valid).sum() / valid.sum().clamp(min=1.0)

        dice = self.dice(logits, targets, valid)
        return self.bce_weight * bce + self.dice_weight * dice

if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W = 4, 32, 32
    targets = (torch.rand(B, 1, H, W) > 0.5).float()
    valid = torch.ones(B, 1, H, W)
    loss_fn = BCEDiceLoss()

    # 1. perfect prediction -> loss close to 0
    perfect = torch.where(targets == 1, 20.0, -20.0)
    l = loss_fn(perfect, targets, valid).item()
    print("1. perfect prediction:", l)
    assert l < 1e-3

    # 2. changing predictions where valid == 0 must not change the loss
    valid_b = valid.clone()
    valid_b[:, :, :8, :] = 0                      # ignore the top 8 rows
    logits = torch.randn(B, 1, H, W)
    logits_b = logits.clone()
    logits_b[valid_b == 0] = torch.randn(int((valid_b == 0).sum())) * 50
    l1 = loss_fn(logits, targets, valid_b)
    l2 = loss_fn(logits_b, targets, valid_b)
    print("2. ignored region:", l1.item(), l2.item())
    assert torch.allclose(l1, l2, atol=1e-6)

    # 3. valid all ones -> BCE part equals plain BCE
    bce_only = BCEDiceLoss(bce_weight=1.0, dice_weight=0.0)
    a = bce_only(logits, targets, valid)
    b = F.binary_cross_entropy_with_logits(logits, targets)
    print("3a. bce vs plain bce:", a.item(), b.item())
    assert torch.allclose(a, b, atol=1e-6)

    # 3b. valid all ones -> Dice part equals a hand-computed Dice loss
    dice_only = BCEDiceLoss(bce_weight=0.0, dice_weight=1.0)
    p = torch.sigmoid(logits)
    inter = (p * targets).sum(dim=(1, 2, 3))
    den = p.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    manual = (1 - (2 * inter + 1e-6) / (den + 1e-6)).mean()
    d = dice_only(logits, targets, valid)
    print("3b. dice vs manual:", d.item(), manual.item())
    assert torch.allclose(d, manual, atol=1e-6)

    # 4. gradients must be zero on ignored pixels
    x = logits.clone().requires_grad_(True)
    loss_fn(x, targets, valid_b).backward()
    g = x.grad[valid_b == 0].abs().max().item()
    print("4. max grad on ignored pixels:", g)
    assert g == 0.0

    # 5. fully ignored batch must not give NaN
    zero_valid = torch.zeros_like(valid)
    l = loss_fn(logits, targets, zero_valid)
    print("5. all-ignored batch:", l.item())
    assert torch.isfinite(l)

    print("all tests passed")