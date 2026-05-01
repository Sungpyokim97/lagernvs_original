# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time

import einops
import torch
import torch.nn as nn
from models.layers.embeddings import (
    init_weights_normal,
    PatchEmbed,
)
from models.layers.final_layer import FinalLayer
from models.layers.renderer_blocks import (
    BidirectionalCrossAttentionBlock,
    CrossAttentionBlock,
    FullAttentionBlock,
)


class Renderer(nn.Module):
    def __init__(
        self,
        depth,
        hidden_size,
        patch_size,
        num_heads,
        pre_transformer_norm_bias=False,
        out_channels=3,
        attention_to_features_type="cross_attention",
    ):
        super().__init__()

        self.out_channels = out_channels
        self.patch_size = patch_size

        tgt_ch = 6
        self.tgt_embedder = PatchEmbed(self.patch_size, tgt_ch, hidden_size, bias=False)
        self.tgt_embedder_4x4 = PatchEmbed(4, tgt_ch, hidden_size, bias=False)
        self.tgt_norm = nn.LayerNorm(hidden_size, bias=pre_transformer_norm_bias)

        self.depth = depth

        self.n_registers = 4
        self.per_view_register_tokens = nn.Parameter(
            torch.zeros(1, self.n_registers, hidden_size, dtype=torch.bfloat16)
        )

        self.attention_to_features_type = attention_to_features_type
        if attention_to_features_type == "cross_attention":
            self.renderer_core = CrossAttentionRendererCore(
                hidden_size, num_heads, depth
            )
        elif attention_to_features_type == "bidirectional_cross_attention":
            self.renderer_core = BidirectionalCrossAttentionRendererCore(
                hidden_size, num_heads, depth
            )
        elif attention_to_features_type == "full_attention":
            self.renderer_core = FullAttentionRendererCore(
                hidden_size, num_heads, depth
            )
        else:
            raise ValueError(
                f"Unknown attention_to_features_type {attention_to_features_type}"
            )
        self.patch_start_idx = self.n_registers

        self.final_layer = FinalLayer(
            hidden_size=hidden_size,
            patch_size=self.patch_size,
            out_channels=self.out_channels,
        )
        self.final_layer_4x4 = FinalLayer(
            hidden_size=hidden_size,
            patch_size=4,
            out_channels=self.out_channels,
        )
        self.output_act = nn.Sigmoid()
        self.initialize_weights()

    def initialize_weights(self):
        for idx, block in enumerate(self.renderer_core.renderer_blocks):
            weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
            block.apply(lambda module: init_weights_normal(module, weight_init_std))

        wc = self.tgt_embedder.proj.weight.data
        nn.init.normal_(wc.view([wc.shape[0], -1]), mean=0.0, std=0.02)
        if self.tgt_embedder.proj.bias is not None:
            nn.init.constant_(self.tgt_embedder.proj.bias, 0)
            
        wc_4x4 = self.tgt_embedder_4x4.proj.weight.data
        nn.init.normal_(wc_4x4.view([wc_4x4.shape[0], -1]), mean=0.0, std=0.02)
        if self.tgt_embedder_4x4.proj.bias is not None:
            nn.init.constant_(self.tgt_embedder_4x4.proj.bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        if self.final_layer.linear.bias is not None:
            nn.init.constant_(self.final_layer.linear.bias, 0)
            
        nn.init.constant_(self.final_layer_4x4.linear.weight, 0)
        if self.final_layer_4x4.linear.bias is not None:
            nn.init.constant_(self.final_layer_4x4.linear.bias, 0)

    def forward(self, rec_tokens, target_rays, timeit=False):
        """
        Inputs:
            rec_tokens: (B x V_target) x (V_input x P) x C
            target_rays: B x V_target x C x H x W
        """
        if timeit:
            torch.cuda.synchronize()
            start_time = time.time()
        b, v_target, _, h_tgt, w_tgt = target_rays.shape
        target_rays = einops.rearrange(target_rays, "b v c h w -> (b v) c h w")
        
        # Use 4x4 embedder for 64x64 targets to maintain spatial dims, 
        # else use default patch size
        if h_tgt == 64 and w_tgt == 64:
            target_tokens = self.tgt_embedder_4x4(target_rays)
            current_patch_size = 4
        else:
            target_tokens = self.tgt_embedder(target_rays)
            current_patch_size = self.patch_size
            
        target_tokens = self.tgt_norm(target_tokens)

        register_tokens_target = einops.repeat(
            self.per_view_register_tokens, "n p c -> (n b1) p c", b1=b * v_target
        )

        x = torch.cat([register_tokens_target, target_tokens], dim=1)

        x = self.renderer_core(x, rec_tokens)

        # register tokens are not used for final prediction, so we can discard them before the final layer
        x = x[:, self.patch_start_idx :, :]

        # start srno frome here

        if current_patch_size == 4:
            x = self.final_layer_4x4(x)
        else:
            x = self.final_layer(x)
            
        x = self.output_act(x)

        rendered_images = einops.rearrange(
            x,
            "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=v_target,
            h=h_tgt // current_patch_size,
            w=w_tgt // current_patch_size,
            p1=current_patch_size,
            p2=current_patch_size,
            c=3,
        )

        
        # 최종 아웃풋 rendered_pixels: B x V_target x 3 x H_lr x W_lr 
        # random 좌표 sampling 을 input size 만큼 진행후 reshape 한 결과
        if timeit:
            torch.cuda.synchronize()
            end_time = time.time()
            return rendered_images, end_time - start_time
        return rendered_images


class CrossAttentionRendererCore(nn.Module):
    """Renderer transformer that conditions on encoder features via cross-attention."""

    def __init__(self, hidden_size, num_heads, depth):
        super().__init__()
        self.depth = depth
        self.renderer_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    hidden_dim=hidden_size,
                    num_heads=num_heads,
                )
                for _ in range(self.depth)
            ]
        )

    def forward(self, x, rec_tokens):
        for renderer_block_idx in range(self.depth):
            if self.training:
                x = torch.utils.checkpoint.checkpoint(
                    self.renderer_blocks[renderer_block_idx],
                    x,
                    rec_tokens,
                    use_reentrant=False,
                )
            else:
                x = self.renderer_blocks[renderer_block_idx](x, rec_tokens)

        return x


class BidirectionalCrossAttentionRendererCore(nn.Module):
    """Renderer transformer with bidirectional cross-attention between target and encoder features."""

    def __init__(self, hidden_size, num_heads, depth):
        super().__init__()
        self.depth = depth
        self.renderer_blocks = nn.ModuleList(
            [
                BidirectionalCrossAttentionBlock(
                    hidden_dim=hidden_size,
                    num_heads=num_heads,
                )
                for _ in range(self.depth - 1)
            ]
            + [CrossAttentionBlock(hidden_dim=hidden_size, num_heads=num_heads)]
        )

    def forward(self, x, rec_tokens):
        for renderer_block_idx in range(self.depth - 1):
            if self.training:
                x, rec_tokens = torch.utils.checkpoint.checkpoint(
                    self.renderer_blocks[renderer_block_idx],
                    x,
                    rec_tokens,
                    use_reentrant=False,
                )
            else:
                x, rec_tokens = self.renderer_blocks[renderer_block_idx](x, rec_tokens)
        if self.training:
            x = torch.utils.checkpoint.checkpoint(
                self.renderer_blocks[-1],
                x,
                rec_tokens,
                use_reentrant=False,
            )
        else:
            x = self.renderer_blocks[-1](x, rec_tokens)

        return x


class FullAttentionRendererCore(nn.Module):
    """Rendeif self.training:
                x = torch.utils.checkpoint.checkpoint(
                    self.renderer_blocks[renderer_block_idx],
                    x,
                    attn_bias=None,
                    use_reentrant=False,
                )
            else:
                rer transformer with full self-attention over concatenated target and encoder features."""

    def __init__(self, hidden_size, num_heads, depth):
        super().__init__()
        self.depth = depth
        self.renderer_blocks = nn.ModuleList(
            [
                FullAttentionBlock(hidden_dim=hidden_size, num_heads=num_heads)
                for _ in range(self.depth)
            ]
        )

    def forward(self, x, rec_tokens):
        num_rec_tokens = rec_tokens.shape[1]
        x = torch.cat([rec_tokens, x], dim=1)
        for renderer_block_idx in range(self.depth):
            x = self.renderer_blocks[renderer_block_idx](x, attn_bias=None)
        x = x[:, num_rec_tokens:, :]
        return x
