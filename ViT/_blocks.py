import torch
import torch.nn as nn


# Patcher and Patch encoder
class Patches(nn.Module):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def __call__(self, x):
        # Compute the shape of the patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        
        # Move the channels to last place
        patches = torch.permute(patches, (0, 2, 3, 4, 5, 1))

        # Get shape
        bs, n_patch_x, n_patch_y, dim_x, dim_y, n_ch = patches.shape
        self.n_patches = n_patch_x * n_patch_y
        self.patch_dim = dim_x * dim_y

        # Flatten
        patches = torch.reshape(patches, (-1, self.n_patches, self.patch_dim))
        return patches
    
class PatchEncoder(nn.Module):
    def __init__(self, num_patches, patch_dim, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim

        # Define projection layer
        self.projection = nn.Linear(patch_dim, projection_dim)
        self.position_embedding = nn.Embedding(num_patches, projection_dim)

    def __call__(self, patch):
        positions = torch.arange(start=0,
                                 end=self.num_patches,
                                 step=1).unsqueeze(0)
        
        projected_patches = self.projection(patch)
        embedding_positions = self.position_embedding(positions)
        encoded = projected_patches + embedding_positions
        return encoded

# only used in postprocessing 
class ResidualBlock(nn.Module):
    def __init__(self, ch_in, ch_out, h, w):
        super(ResidualBlock, self).__init__()
        self.preprocess_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)
        self.layer_norm = nn.LayerNorm((ch_out, h, w))

        self.conv1 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding="same")

    def __call__(self, x):
        x = self.preprocess_conv(x)
        h = self.layer_norm(x)
        h = nn.SiLU()(self.conv1(h))
        h = self.conv2(h)

        return x + h
    
# MLP
class MLP(nn.Module):
    def __init__(self, dims: list[int], dropout_rate: int) -> None:
        super(MLP, self).__init__()
        self.linears = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)
            ])
        self.dropout = nn.Dropout(p=dropout_rate)

    def __call__(self, x):
        for layer in self.linears:
            x = nn.GELU()(layer(x))
        return self.dropout(x)