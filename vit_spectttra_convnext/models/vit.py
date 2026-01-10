import torch
import torch.nn as nn
from ..layers import (
    SinusoidPositionalEncoding,
    LearnedPositionalEncoding,
    Transformer,
)
from timm.layers import PatchEmbed


class ViT(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        embed_dim,
        num_heads,
        num_layers,
        pe_learnable=False,
        patch_norm=False,
        pos_drop_rate=0.0,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        mlp_ratio=4.0,
    ):
        super().__init__()
        assert (
            image_size[0] % patch_size == 0 and image_size[1] % patch_size == 0
        ), "Image dimensions must be divisible by patch size."

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pe_learnable = pe_learnable
        self.patch_norm = patch_norm
        self.pos_drop_rate = pos_drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.proj_drop_rate = proj_drop_rate
        self.mlp_ratio = mlp_ratio

        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)

        # self.patch_conv = nn.Conv2d(
        #     1, embed_dim, kernel_size=patch_size, stride=patch_size
        # )  # Original ViT has 3 input channels
        self.patch_encoder = PatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=1,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm if patch_norm else None,
        )
        self.pos_encoder = (
            SinusoidPositionalEncoding(embed_dim)
            if not pe_learnable
            else LearnedPositionalEncoding(embed_dim, self.num_patches)
        )
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        self.transformer = Transformer(
            embed_dim,
            num_heads,
            num_layers,
            attn_drop=self.attn_drop_rate,
            proj_drop=self.proj_drop_rate,
            mlp_ratio=self.mlp_ratio,
        )

    def forward(self, x):
        B = x.shape[0]
        # x = x.unsqueeze(1)  # B x 1 x n_mels x n_frames # taken care of in the AudioClassifier
        if x.dim() == 3:
            x = x.unsqueeze(1)  # timm PatchEmbed expects 4D tensor

        # Convolutional patch embedding
        # patches = self.patch_conv(x)  # B x embed_dim x num_patches_h x num_patches_w
        patches = self.patch_encoder(x)

        # # Reshape patches
        # patches = patches.permute(
        #     0, 2, 3, 1
        # ).contiguous()  # B x num_patches_h x num_patches_w x embed_dim
        # patches = patches.view(B, -1, patches.size(-1))  # B x num_patches x embed_dim

        # Add positional embeddings
        embeddings = self.pos_encoder(patches)

        # Positional dropout
        embeddings = self.pos_drop(embeddings)

        # Transformer encoding
        output = self.transformer(embeddings)  # B x num_patches x embed_dim

        return output


batch_size = 1
input_height = 128
input_width = 384 * 6 * 4
patch_size = 16
