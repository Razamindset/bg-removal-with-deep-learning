import torch

def _binarize(logits, threshold=0.5):
    return (torch.sigmoid(logits) > threshold).float()

@torch.no_grad()
def dice_score(logits, target, valid, eps=1e-7):
    preds = _binarize(logits) * valid
    target = target * valid
    inter = (preds * target).sum(dim=(1, 2, 3))
    total = preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (total + eps)).mean()

@torch.no_grad()
def iou_score(logits, target, valid, eps=1e-7):
    preds = _binarize(logits) * valid
    target = target * valid
    inter = (preds * target).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean()