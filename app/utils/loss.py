import torch
import torch.nn as nn
import torch.nn.functional as F


class UNETRLoss(nn.Module):
    """
    Custom loss function for UNETR model in lung nodule detection and classification.

    This loss function combines segmentation and classification losses:
    - For segmentation: Combination of Dice Loss and Focal Loss
    - For classification: Binary Cross-Entropy Loss

    Attributes:
        segmentation_weight (float): Weight for the segmentation loss component.
        classification_weight (float): Weight for the classification loss component.
        alpha (float): Weighting factor in Focal Loss to balance positive vs negative samples.
        gamma (float): Focusing parameter in Focal Loss to focus on hard examples.
        smooth (float): Smoothing factor for Dice Loss to avoid division by zero.
        eps (float): Small constant to avoid numerical instability.
    """

    def __init__(
        self,
        segmentation_weight: float = 1.0,
        classification_weight: float = 1.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
        smooth: float = 1e-5,
        eps: float = 1e-7,
    ):
        """
        Initialize the UNETRLoss.

        Args:
            segmentation_weight (float): Weight for segmentation loss. Default is 1.0.
            classification_weight (float): Weight for classification loss. Default is 1.0.
            alpha (float): Weighting factor in Focal Loss. Default is 0.25.
            gamma (float): Focusing parameter in Focal Loss. Default is 2.0.
            smooth (float): Smoothing factor for Dice Loss. Default is 1e-5.
            eps (float): Small constant for numerical stability. Default is 1e-7.
        """
        super().__init__()
        self.segmentation_weight = segmentation_weight
        self.classification_weight = classification_weight
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.eps = eps

    def forward(
        self,
        pred_seg: torch.Tensor,
        true_seg: torch.Tensor,
        pred_class: torch.Tensor,
        true_class: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the combined loss for segmentation and classification.

        Args:
            pred_seg (torch.Tensor): Predicted segmentation mask, shape (B, C, D, H, W)
            true_seg (torch.Tensor): Ground truth segmentation mask, shape (B, C, D, H, W)
            pred_class (torch.Tensor): Predicted classification probabilities, shape (B, 1)
            true_class (torch.Tensor): Ground truth classification labels, shape (B, 1)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - total_loss: Combined weighted loss
                - seg_loss: Segmentation loss (Dice + Focal)
                - class_loss: Classification loss (Binary Cross-Entropy)
        """
        # Compute segmentation loss (Dice + Focal)
        dice_loss = self.dice_loss(pred_seg, true_seg)
        focal_loss = self.focal_loss(pred_seg, true_seg)
        seg_loss = dice_loss + focal_loss

        # Compute classification loss (Binary Cross-Entropy)
        class_loss = F.binary_cross_entropy(pred_class, true_class, reduction="mean")

        # Combine losses with weights
        total_loss = self.segmentation_weight * seg_loss + self.classification_weight * class_loss

        return total_loss, seg_loss, class_loss

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice Loss for segmentation.

        Dice Loss is effective for handling class imbalance in segmentation tasks.

        Args:
            pred (torch.Tensor): Predicted segmentation mask, shape (B, C, D, H, W)
            target (torch.Tensor): Ground truth segmentation mask, shape (B, C, D, H, W)

        Returns:
            torch.Tensor: Dice Loss value
        """
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]

        dice_loss = torch.zeros(1, device=pred.device)
        for class_idx in range(num_classes):
            pred_class = pred[:, class_idx, ...]
            target_class = (target == class_idx).float()

            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum()

            dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += 1 - dice_score

        return dice_loss / num_classes

    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss for segmentation.

        Focal Loss focuses on hard, misclassified examples, which is useful for
        segmenting small nodules in large 3D volumes.

        Args:
            pred (torch.Tensor): Predicted segmentation mask, shape (B, C, D, H, W)
            target (torch.Tensor): Ground truth segmentation mask, shape (B, C, D, H, W)

        Returns:
            torch.Tensor: Focal Loss value
        """
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]

        focal_loss = torch.zeros(1, device=pred.device)
        for class_idx in range(num_classes):
            pred_class = pred[:, class_idx, ...]
            target_class = (target == class_idx).float()

            # Compute Focal Loss
            pt = target_class * pred_class + (1 - target_class) * (1 - pred_class)
            focal_weight = (1 - pt) ** self.gamma
            focal_loss += -self.alpha * focal_weight * torch.log(pt + self.eps)

        return torch.mean(focal_loss)
