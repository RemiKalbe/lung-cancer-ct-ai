from typing import Any, Dict, List, Optional, Tuple

import torch as tc
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Converts a 3D volume into a sequence of patch embeddings.

    This class is inspired by the Vision Transformer (ViT) architecture, adapted for 3D volumes
    as used in the UNETR paper.

    In the context of 3D medical imaging, this layer divides the input volume into non-overlapping
    3D patches and projects each patch into a lower-dimensional embedding space. This is the first
    step in processing the input for the transformer encoder in the UNETR architecture.

    Attributes:
        patch_size (int): The size of the patches in each dimension (assumed to be cubic).
        proj (nn.Conv3d): 3D convolution layer that both divides the input into patches and projects them.
    """

    def __init__(self, patch_size: int = 16, in_channels: int = 1, embed_dim: int = 768) -> None:
        """
        Initializes the PatchEmbedding module.

        Args:
            patch_size (int): The size of the patches. Default is 16, as used in the UNETR paper.
            in_channels (int): The number of input channels. Default is 1 for typical CT scans.
            embed_dim (int): The dimension of the embedding space. Default is 768, following the UNETR paper.
        """
        super().__init__()
        self.patch_size: int = patch_size

        # The Conv3d layer serves dual purpose:
        # 1. It divides the input into patches (by using kernel_size and stride equal to patch_size)
        # 2. It projects these patches into the embedding space (by setting out_channels to embed_dim)
        self.proj: nn.Conv3d = nn.Conv3d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: tc.Tensor) -> tc.Tensor:
        """
        Forward pass of the PatchEmbedding module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W, D)
                              where H, W, D are the height, width, and depth of the input volume.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, num_patches, embed_dim)
                          where num_patches = (H/patch_size) * (W/patch_size) * (D/patch_size)

        Raises:
            ValueError: If input tensor dimensions are not divisible by patch_size.
        """
        # Check if input dimensions are divisible by patch_size
        if any(dim % self.patch_size != 0 for dim in x.shape[2:]):
            raise ValueError(
                f"Input tensor spatial dimensions must be divisible by patch_size {self.patch_size}"
            )

        # Apply the projection (which also divides into patches)
        x = self.proj(x)  # Shape: (batch_size, embed_dim, H', W', D')
        # where H', W', D' are the reduced spatial dimensions

        # Reshape and transpose to get the final sequence of patch embeddings
        return x.flatten(2).transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)

    @property
    def num_patches(self) -> Tuple[int, int, int]:
        """
        Calculates the number of patches in each dimension.

        Returns:
            Tuple[int, int, int]: Number of patches in height, width, and depth dimensions.
        """
        return (self.patch_size, self.patch_size, self.patch_size)


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder for the UNETR architecture.

    This class implements the transformer encoder part of the UNETR model as described in the paper.
    It consists of multiple transformer encoder layers that process the patch embeddings.

    The encoder produces outputs at different depths, which are later used by the decoder
    for multi-scale feature fusion, similar to skip connections in U-Net architectures.

    Attributes:
        layers (nn.ModuleList): List of transformer encoder layers.
        norm (nn.LayerNorm): Layer normalization applied after the transformer layers.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
    ) -> None:
        """
        Initialize the TransformerEncoder.

        Args:
            embed_dim (int): Dimension of the input embeddings. Default is 768.
            depth (int): Number of transformer layers. Default is 12.
            num_heads (int): Number of attention heads in each transformer layer. Default is 12.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default is 4.0.
            dropout (float): Dropout rate. Default is 0.1.
            attention_dropout (float): Dropout rate for attention weights. Default is 0.0.
        """
        super().__init__()

        self.depth = depth
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embed_dim * mlp_ratio),
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: tc.Tensor, mask: Optional[tc.Tensor] = None) -> List[tc.Tensor]:
        """
        Forward pass of the TransformerEncoder.

        This method processes the input through all transformer layers and returns
        the outputs of specific layers for later use in the decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, embed_dim).
            mask (Optional[torch.Tensor]): Attention mask (if needed). Default is None.

        Returns:
            List[torch.Tensor]: List of output tensors from specific layers of the transformer.
                                Each tensor has shape (batch_size, num_patches, embed_dim).
        """
        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x, src_mask=mask)
            if i in self.get_output_layers():  # Collect features from these specific layers
                features.append(self.norm(x))

        return features

    def get_output_layers(self) -> List[int]:
        """
        Determine which layer outputs to collect based on the depth of the transformer.

        Returns:
            List[int]: Indices of layers whose outputs should be collected.
        """
        if self.depth <= 4:
            return [self.depth - 1]  # Only return the last layer if depth is 4 or less
        elif self.depth <= 8:
            return [self.depth // 2 - 1, self.depth - 1]  # Return middle and last layer
        else:
            # Divide the depth into 4 segments and return the last layer of each segment
            return [i - 1 for i in range(self.depth // 4, self.depth + 1, self.depth // 4)]


class TransformerEncoderLayer(nn.Module):
    """
    A single Transformer Encoder Layer.

    This class implements a single layer of the transformer encoder, including
    multi-head self-attention and a feed-forward network.

    Attributes:
        self_attn (nn.MultiheadAttention): Multi-head self-attention module.
        linear1 (nn.Linear): First linear layer of the feed-forward network.
        dropout (nn.Dropout): Dropout layer.
        linear2 (nn.Linear): Second linear layer of the feed-forward network.
        norm1 (nn.LayerNorm): Layer normalization after self-attention.
        norm2 (nn.LayerNorm): Layer normalization after feed-forward network.
        dropout1 (nn.Dropout): Dropout after self-attention.
        dropout2 (nn.Dropout): Dropout after feed-forward network.
        activation (nn.Module): Activation function for the feed-forward network.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        """
        Initialize the TransformerEncoderLayer.

        Args:
            d_model (int): The number of expected features in the input.
            nhead (int): The number of heads in the multiheadattention model.
            dim_feedforward (int): The dimension of the feedforward network model. Default is 2048.
            dropout (float): Dropout value. Default is 0.1.
            attention_dropout (float): Dropout for attention weights. Default is 0.0.
            activation (str): The activation function for the feed-forward network. Default is "relu".
        """
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=attention_dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)

    def _get_activation_fn(self, activation: str) -> nn.Module:
        """
        Get the activation function.

        Args:
            activation (str): Name of the activation function.

        Returns:
            nn.Module: The activation function.

        Raises:
            RuntimeError: If the activation function is not supported.
        """
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        raise RuntimeError(f"Activation should be relu/gelu, not {activation}")

    def forward(
        self,
        src: tc.Tensor,
        src_mask: Optional[tc.Tensor] = None,
        src_key_padding_mask: Optional[tc.Tensor] = None,
    ) -> tc.Tensor:
        """
        Forward pass of the TransformerEncoderLayer.

        Args:
            src (torch.Tensor): The input tensor of shape (seq_len, batch_size, embed_dim).
            src_mask (Optional[torch.Tensor]): The mask for the src sequence. Default is None.
            src_key_padding_mask (Optional[torch.Tensor]): The mask for the src keys per batch. Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (seq_len, batch_size, embed_dim).
        """
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class DecoderBlock(nn.Module):
    """
    Decoder Block for the UNETR architecture.

    This class implements a single decoder block of the UNETR model as described in the paper.
    It performs upsampling of the input features and combines them with skip connections
    from the encoder, similar to the U-Net architecture.

    Each decoder block consists of:
    1. A 3D transposed convolution for upsampling
    2. Concatenation with skip connection features (if provided)
    3. Two 3D convolutions with batch normalization and ReLU activation

    Attributes:
        conv_trans (nn.ConvTranspose3d): Transposed convolution for upsampling
        conv1 (nn.Conv3d): First 3D convolution
        conv2 (nn.Conv3d): Second 3D convolution
        norm1 (nn.BatchNorm3d): Batch normalization after first convolution
        norm2 (nn.BatchNorm3d): Batch normalization after second convolution
        relu (nn.ReLU): ReLU activation function
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2):
        """
        Initialize the DecoderBlock.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the convolutional kernel. Default is 3.
            stride (int): Stride of the transposed convolution (determines the upsampling factor). Default is 2.
        """
        super().__init__()

        # Transposed convolution for upsampling
        self.conv_trans = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=stride, stride=stride
        )

        # First convolution and normalization
        self.conv1 = nn.Conv3d(out_channels * 2, out_channels, kernel_size=kernel_size, padding=1)
        self.norm1 = nn.BatchNorm3d(out_channels)

        # Second convolution and normalization
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.norm2 = nn.BatchNorm3d(out_channels)

        # ReLU activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: tc.Tensor, skip: Optional[tc.Tensor] = None) -> tc.Tensor:
        """
        Forward pass of the DecoderBlock.

        This method upsamples the input, combines it with skip connections if provided,
        and applies convolutions with normalizations and activations.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
            skip (Optional[torch.Tensor]): Skip connection tensor from the encoder,
                                           of shape (batch_size, out_channels, D*stride, H*stride, W*stride).
                                           If None, no skip connection is used. Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D*stride, H*stride, W*stride)

        Raises:
            ValueError: If the spatial dimensions of the skip connection do not match the upsampled input.
        """
        # Upsample input
        x = self.conv_trans(x)

        # Combine with skip connection if provided
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                raise ValueError(
                    f"Shape mismatch in DecoderBlock: "
                    f"upsampled shape {x.shape[2:]} != skip shape {skip.shape[2:]}"
                )
            x = tc.cat([x, skip], dim=1)

        # Apply convolutions with normalizations and activations
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))

        return x


class UNETR(nn.Module):
    """
    UNETR: UNet Transformer for 3D Medical Image Segmentation

    This class implements the UNETR architecture as described in the paper
    "UNETR: Transformers for 3D Medical Image Segmentation" (Hatamizadeh et al., 2022).

    UNETR combines a transformer encoder with a CNN decoder for 3D medical image segmentation.
    It uses skip connections from the transformer encoder to the CNN decoder for multi-scale feature fusion.

    Attributes:
        patch_embedding (PatchEmbedding): Module to convert input volume into patch embeddings.
        transformer (TransformerEncoder): Transformer encoder to process patch embeddings.
        decoder_blocks (nn.ModuleList): List of decoder blocks for upsampling and feature fusion.
        segmentation_head (nn.Conv3d): Final convolution layer for segmentation output.
        classification_head (nn.Sequential): Sequence of layers for nodule classification.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        img_size: Tuple[int, int, int] = (128, 128, 128),
        patch_size: int = 16,
        embed_dim: int = 768,
        num_heads: int = 12,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        """
        Initialize the UNETR model.

        Args:
            in_channels (int): Number of input channels in the image. Default is 1 for CT scans.
            out_channels (int): Number of output channels (classes) for segmentation. Default is 2.
            img_size (Tuple[int, int, int]): Size of the input image (D, H, W). Default is (128, 128, 128).
            patch_size (int): Size of the patches to be extracted from the input image. Default is 16.
            embed_dim (int): Dimension of the patch embeddings. Default is 768.
            num_heads (int): Number of attention heads in the transformer. Default is 12.
            depth (int): Number of transformer layers. Default is 12.
            mlp_ratio (float): Ratio of MLP hidden dim to embedding dim in transformer. Default is 4.0.
            dropout (float): Dropout rate in the transformer. Default is 0.1.
        """
        super().__init__()

        self.patch_embedding = PatchEmbedding(patch_size, in_channels, embed_dim)
        self.transformer = TransformerEncoder(embed_dim, depth, num_heads, mlp_ratio, dropout)

        # Calculate the number of patches in each dimension
        self.patches_per_dim = [img_size[i] // patch_size for i in range(3)]

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(embed_dim, 512),
                DecoderBlock(512, 256),
                DecoderBlock(256, 128),
                DecoderBlock(128, 64),
            ]
        )

        # Segmentation head
        self.segmentation_head = nn.Conv3d(64, out_channels, kernel_size=1)

        # Classification head
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: tc.Tensor) -> Tuple[tc.Tensor, tc.Tensor]:
        """
        Forward pass of the UNETR model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Segmentation output of shape (batch_size, out_channels, D, H, W)
                - Classification output of shape (batch_size, 1)

        Raises:
            ValueError: If input tensor shape doesn't match the expected input size.
        """
        # Check input size
        if x.shape[2:] != tuple(
            self.patches_per_dim[i] * self.patch_embedding.patch_size for i in range(3)
        ):
            raise ValueError(f"Input tensor shape {x.shape} doesn't match the expected input size.")

        # Patch embedding
        x = self.patch_embedding(x)

        # Transformer encoder
        features = self.transformer(x)

        # Reshape features for decoder
        batch_size = x.shape[0]
        features = [
            feat.transpose(1, 2).view(batch_size, -1, *self.patches_per_dim) for feat in features
        ]

        # Decoder
        x = features[-1]
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = features[-(i + 2)] if i < len(features) - 1 else None
            x = decoder_block(x, skip)

        # Segmentation output
        segmentation = self.segmentation_head(x)

        # Classification output
        classification = self.classification_head(features[-1])

        return segmentation, classification

    def get_parameters(self) -> List[Dict[str, Any]]:
        """
        Get all parameters of the model.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing parameter groups.
        """
        return [
            {"params": self.patch_embedding.parameters()},
            {"params": self.transformer.parameters()},
            {"params": self.decoder_blocks.parameters()},
            {"params": self.segmentation_head.parameters()},
            {"params": self.classification_head.parameters()},
        ]
