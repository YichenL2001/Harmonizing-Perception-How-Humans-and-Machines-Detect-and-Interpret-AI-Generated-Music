import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers import PatchEmbed

from .feature import FeatureExtractor
from .layers import (
    SinusoidPositionalEncoding,
    LearnedPositionalEncoding,
    Transformer,
    STTokenizer,
)


def _timm_embed_dim(encoder: nn.Module) -> int:
    for attr in ("num_features", "head_hidden_size", "embed_dim"):
        val = getattr(encoder, attr, None)
        if isinstance(val, int):
            return val
    raise ValueError("Could not infer timm embed dim")


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
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("Image dimensions must be divisible by patch size.")

        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
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
            attn_drop=attn_drop_rate,
            proj_drop=proj_drop_rate,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        patches = self.patch_encoder(x)
        embeddings = self.pos_encoder(patches)
        embeddings = self.pos_drop(embeddings)
        return self.transformer(embeddings)


class SpecTTTra(nn.Module):
    def __init__(
        self,
        input_spec_dim,
        input_temp_dim,
        embed_dim,
        t_clip,
        f_clip,
        num_heads,
        num_layers,
        pre_norm=False,
        pe_learnable=False,
        pos_drop_rate=0.0,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.st_tokenizer = STTokenizer(
            input_spec_dim,
            input_temp_dim,
            t_clip,
            f_clip,
            embed_dim,
            pre_norm=pre_norm,
            pe_learnable=pe_learnable,
        )
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        self.transformer = Transformer(
            embed_dim,
            num_heads,
            num_layers,
            attn_drop=attn_drop_rate,
            proj_drop=proj_drop_rate,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        tokens = self.st_tokenizer(x)
        tokens = self.pos_drop(tokens)
        return self.transformer(tokens)


class AudioClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        input_shape,
        num_classes: int,
        melspec_cfg: dict,
        model_cfg: dict,
    ):
        super().__init__()
        self.model_name = model_name
        self.input_shape = tuple(input_shape)
        self.num_classes = int(num_classes)
        self.ft_extractor = FeatureExtractor(**melspec_cfg)
        self.encoder = self._build_encoder(model_cfg)
        self.embed_dim = self._embed_dim()
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)

    def _build_encoder(self, cfg: dict) -> nn.Module:
        if self.model_name == "SpecTTTra":
            return SpecTTTra(
                input_spec_dim=self.input_shape[0],
                input_temp_dim=self.input_shape[1],
                embed_dim=cfg["embed_dim"],
                t_clip=cfg["t_clip"],
                f_clip=cfg["f_clip"],
                num_heads=cfg["num_heads"],
                num_layers=cfg["num_layers"],
                pre_norm=cfg["pre_norm"],
                pe_learnable=cfg["pe_learnable"],
                pos_drop_rate=cfg["pos_drop_rate"],
                attn_drop_rate=cfg["attn_drop_rate"],
                proj_drop_rate=cfg["proj_drop_rate"],
                mlp_ratio=cfg["mlp_ratio"],
            )
        if self.model_name == "ViT":
            return ViT(
                image_size=self.input_shape,
                patch_size=cfg["patch_size"],
                embed_dim=cfg["embed_dim"],
                num_heads=cfg["num_heads"],
                num_layers=cfg["num_layers"],
                pe_learnable=cfg["pe_learnable"],
                patch_norm=cfg["patch_norm"],
                pos_drop_rate=cfg["pos_drop_rate"],
                attn_drop_rate=cfg["attn_drop_rate"],
                proj_drop_rate=cfg["proj_drop_rate"],
                mlp_ratio=cfg["mlp_ratio"],
            )
        if self.model_name == "ConvNeXt":
            name = cfg.get("timm_name", "convnext_tiny")
            return timm.create_model(name, pretrained=cfg.get("pretrained", False), in_chans=1, num_classes=0)
        raise ValueError("Unsupported model name {}".format(self.model_name))

    def _embed_dim(self) -> int:
        if self.model_name == "ConvNeXt":
            return _timm_embed_dim(self.encoder)
        return getattr(self.encoder, "embed_dim")

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        spec = self.ft_extractor(audio)
        spec = spec.unsqueeze(1)
        spec = F.interpolate(spec, size=self.input_shape, mode="bilinear")
        features = self.encoder(spec)
        if features.dim() == 4:
            features = features.mean(dim=(2, 3))
        elif features.dim() == 3:
            features = features.mean(dim=1)
        return self.classifier(features)
