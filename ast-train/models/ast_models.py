# -*- coding: utf-8 -*-
# @Time    : 6/10/21 5:04 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : ast_models.py

from collections import OrderedDict
from pathlib import Path
import os

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import timm
from timm.models.layers import to_2tuple, trunc_normal_

TORCH_HOME = Path(__file__).resolve().parents[2] / "pretrained_models"
os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ASTModel(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384]
    """

    def __init__(
        self,
        label_dim=2,
        fstride=10,
        tstride=10,
        input_fdim=128,
        input_tdim=1024,
        imagenet_pretrain=True,
        audioset_pretrain=True,
        model_size="base384",
        verbose=True,
    ):
        super().__init__()
        assert timm.__version__ == "0.4.5", "Please use timm == 0.4.5"

        if verbose:
            print("---------------AST Model Summary---------------")
            print(
                "ImageNet pretraining: {}, AudioSet pretraining: {}".format(
                    str(imagenet_pretrain), str(audioset_pretrain)
                )
            )

        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        if not audioset_pretrain:
            if model_size == "tiny224":
                self.v = timm.create_model("vit_deit_tiny_distilled_patch16_224", pretrained=imagenet_pretrain)
            elif model_size == "small224":
                self.v = timm.create_model("vit_deit_small_distilled_patch16_224", pretrained=imagenet_pretrain)
            elif model_size == "base224":
                self.v = timm.create_model("vit_deit_base_distilled_patch16_224", pretrained=imagenet_pretrain)
            elif model_size == "base384":
                self.v = timm.create_model("vit_deit_base_distilled_patch16_384", pretrained=imagenet_pretrain)
            else:
                raise ValueError("Model size must be one of tiny224, small224, base224, base384.")

            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches**0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, label_dim),
            )

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose:
                print("frequncey stride={}, time stride={}".format(fstride, tstride))
                print("number of patches={}".format(num_patches))

            new_proj = torch.nn.Conv2d(
                1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride)
            )
            if imagenet_pretrain:
                new_proj.weight = torch.nn.Parameter(
                    torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1)
                )
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            if imagenet_pretrain:
                new_pos_embed = (
                    self.v.pos_embed[:, 2:, :]
                    .detach()
                    .reshape(1, self.original_num_patches, self.original_embedding_dim)
                    .transpose(1, 2)
                    .reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                )
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[
                        :,
                        :,
                        :,
                        int(self.oringal_hw / 2) - int(t_dim / 2) : int(self.oringal_hw / 2) - int(t_dim / 2)
                        + t_dim,
                    ]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(
                        new_pos_embed, size=(self.oringal_hw, t_dim), mode="bilinear"
                    )
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[
                        :,
                        :,
                        int(self.oringal_hw / 2) - int(f_dim / 2) : int(self.oringal_hw / 2) - int(f_dim / 2)
                        + f_dim,
                        :,
                    ]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(
                        new_pos_embed, size=(f_dim, t_dim), mode="bilinear"
                    )
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
                self.v.pos_embed = nn.Parameter(
                    torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1)
                )
            else:
                new_pos_embed = nn.Parameter(
                    torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim)
                )
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=0.02)

        else:
            if not imagenet_pretrain:
                raise ValueError(
                    "Model pretrained on only audioset is not supported, set imagenet_pretrain=True"
                )
            if model_size != "base384":
                raise ValueError("Only base384 AudioSet pretrained model is supported.")

            pretrained_path = TORCH_HOME / "audioset_10_10_0.4593.pth"
            if not pretrained_path.exists():
                raise FileNotFoundError(
                    "AudioSet pretrained model not found at {}".format(pretrained_path)
                )

            sd = torch.load(pretrained_path, map_location="cpu")
            new_sd = OrderedDict()
            for k, v in sd.items():
                new_key = k
                if k.startswith("module."):
                    new_key = k[len("module.") :]
                new_sd[new_key] = v

            audio_model = ASTModel(
                label_dim=527,
                fstride=10,
                tstride=10,
                input_fdim=128,
                input_tdim=1024,
                imagenet_pretrain=False,
                audioset_pretrain=False,
                model_size="base384",
                verbose=False,
            )
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(new_sd, strict=False)
            self.v = audio_model.module.v

            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, label_dim),
            )

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose:
                print("frequncey stride={}, time stride={}".format(fstride, tstride))
                print("number of patches={}".format(num_patches))

            new_pos_embed = (
                self.v.pos_embed[:, 2:, :]
                .detach()
                .reshape(1, 1212, 768)
                .transpose(1, 2)
                .reshape(1, 768, 12, 101)
            )
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim / 2) : 50 - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode="bilinear")
            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim / 2) : 6 - int(f_dim / 2) + f_dim, :]
            elif f_dim > 12:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode="bilinear")
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(
                torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1)
            )

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1000):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    @autocast()
    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        batch_size = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(batch_size, -1, -1)
        dist_token = self.v.dist_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        x = self.mlp_head(x)
        return x


if __name__ == "__main__":
    input_tdim = 100
    ast_mdl = ASTModel(input_tdim=input_tdim)
    test_input = torch.rand([10, input_tdim, 128])
    test_output = ast_mdl(test_input)
    print(test_output.shape)

    input_tdim = 256
    ast_mdl = ASTModel(input_tdim=input_tdim, label_dim=50, audioset_pretrain=True)
    test_input = torch.rand([10, input_tdim, 128])
    test_output = ast_mdl(test_input)
    print(test_output.shape)
