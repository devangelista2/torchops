import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from skimage import data

from _blocks import MLP, PatchEncoder, Patches, ResidualBlock


class ViT(nn.Module):
    def __init__(self, input_shape: tuple[int], 
                 patch_size: int,
                 projection_dim: int,
                 n_transformer_layers: int,
                 n_heads: int,
                 mlp_head_units: tuple[int]):
        super(ViT, self).__init__()
        
        self.bs, self.c, self.h, self.w = input_shape
        self.patch_size = patch_size

        self.num_patches_h = self.h // self.patch_size
        self.num_patches_w = self.w // self.patch_size
        self.n_patches = self.num_patches_h * self.num_patches_w
        self.patch_dim = self.patch_size * self.patch_size * self.c

        self.projection_dim = projection_dim

        self.n_transformer_layers = n_transformer_layers
        self.n_heads = n_heads
        self.mlp_head_units = mlp_head_units
        
        # Define modules
        self.patcher = Patches(patch_size)
        self.patch_encoder = PatchEncoder(self.n_patches, self.patch_dim, projection_dim)

        self.normalizations1 = nn.ModuleList([
            nn.LayerNorm((self.n_patches, self.patch_dim), eps=1e-6) for _ in range(n_transformer_layers)
        ])
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(self.projection_dim, n_heads, dropout=0.1) for _ in range(n_transformer_layers)
        ])
        self.normalizations2 = nn.ModuleList([
            nn.LayerNorm((self.n_patches, self.patch_dim), eps=1e-6) for _ in range(n_transformer_layers)
        ])
        self.mlps = nn.ModuleList([
            MLP(dims=(self.projection_dim,) + mlp_head_units, dropout_rate=0.1) for _ in range(n_transformer_layers)
        ])

        # Last layer norm
        self.normalization3 = nn.LayerNorm((self.n_patches, self.patch_dim), eps=1e-6)

        ### Upsampling
        # Processing
        self.conv_process1 = nn.Conv2d(self.projection_dim, self.projection_dim // 2, kernel_size=1, padding="same")
        self.conv_process2 = nn.Conv2d(self.projection_dim // 2, self.projection_dim // 4, kernel_size=1, padding="same")
        self.conv_process3 = nn.Conv2d(self.projection_dim // 4, self.projection_dim // 8, kernel_size=1, padding="same")
        self.conv_process4 = nn.Conv2d(self.projection_dim // 8, self.c, kernel_size=1, padding="same")

        # Residual conv
        self.residual_conv1 = ResidualBlock(self.projection_dim, self.projection_dim, self.num_patches_h, self.num_patches_w)
        self.residual_conv2 = ResidualBlock(self.projection_dim // 2, self.projection_dim // 2, self.num_patches_h * 2, self.num_patches_w * 2)
        self.residual_conv3 = ResidualBlock(self.projection_dim // 4, self.projection_dim // 4, self.num_patches_h * 4, self.num_patches_w * 4)
        self.residual_conv4 = ResidualBlock(self.projection_dim // 8, self.projection_dim // 8, self.num_patches_h * 8, self.num_patches_w * 8)

        # Transpose convolutions
        self.conv_t1 = nn.ConvTranspose2d(self.projection_dim, self.projection_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_t2 = nn.ConvTranspose2d(self.projection_dim // 2, self.projection_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_t3 = nn.ConvTranspose2d(self.projection_dim // 4, self.projection_dim // 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_t4 = nn.ConvTranspose2d(self.projection_dim // 8, self.projection_dim // 8, kernel_size=3, stride=2, padding=1, output_padding=1)

    def __call__(self, x):
        # Compute patches
        p = self.patcher(x)
        encoded_patches = self.patch_encoder(p)

        for i in range(self.n_transformer_layers):
            # Layer norm1
            x1 = self.normalizations1[i](encoded_patches)

            # Multihead Attention
            attention = self.attentions[i](x1, x1, x1, need_weights=False)[0]

            # Skip connection 1
            x2 = attention + encoded_patches

            # Layer norm 2
            x3 = self.normalizations2[i](x2)

            # MLP
            x3 = self.mlps[i](x3)

            # Skip connection 2
            encoded_patches = x3 + x2

        # Create a [batch_size, projection_dim] tensor.
        out = self.normalization3(encoded_patches)

        # Spatial reshape of the patches
        out = out.view(self.bs, self.projection_dim, self.num_patches_h, self.num_patches_w)

        # Residual + ConvTranspose + Process (1)
        out = self.residual_conv1(out)
        out = nn.ReLU()(self.conv_t1(out))
        out = nn.ReLU()(self.conv_process1(out))

        # Residual + ConvTranspose + Process (2)
        out = self.residual_conv2(out)
        out = nn.ReLU()(self.conv_t2(out))
        out = nn.ReLU()(self.conv_process2(out))

        # Residual + ConvTranspose + Process (3)
        out = self.residual_conv3(out)
        out = nn.ReLU()(self.conv_t3(out))
        out = nn.ReLU()(self.conv_process3(out))

        # Residual + ConvTranspose + Process (4)
        out = self.residual_conv4(out)
        out = nn.ReLU()(self.conv_t4(out))
        out = nn.ReLU()(self.conv_process4(out))

        return out



def main():
    #######################################
    # HYPERPARAMETERS
    #######################################
    PATCH_xDIM = 128
    PROJECTION_DIM = 256

    N_HEADS = 8
    IMAGE_SIZE = 256
    PATCH_SIZE = 16
    MLP_HEAD_UNITS = (
        PROJECTION_DIM * 2,
        PROJECTION_DIM,
    ) # Size of the transformer layers
    N_TRANSFORMER_LAYERS = 6

    #######################################
    # TEST
    #######################################
    # Define the tensor
    x = torch.randn((1, 1, IMAGE_SIZE, IMAGE_SIZE))
    x = (x - x.min()) / (x.max() - x.min())
    print(f"Starting size: {x.shape}")

    # Define ViT
    ViT_model = ViT(input_shape=x.shape,
                    patch_size=PATCH_SIZE,
                    projection_dim=PROJECTION_DIM,
                    n_transformer_layers=N_TRANSFORMER_LAYERS,
                    n_heads=N_HEADS,
                    mlp_head_units=MLP_HEAD_UNITS)
    
    # Test
    y = ViT_model(x)
    print(f"Processed size: {y.shape}.")

if __name__ == '__main__':
    main()