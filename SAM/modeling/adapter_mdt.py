# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Convpass(nn.Module):
    def __init__(self, n_embd,dim=8, xavier_init=False):
        super().__init__()


        self.adapter_conv1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.adapter_conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.adapter_conv3 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:

            nn.init.zeros_(self.adapter_conv1.weight)
            nn.init.zeros_(self.adapter_conv2.weight)
            nn.init.zeros_(self.adapter_conv3.weight)

            self.adapter_conv1.weight.data[:, :, 0, 0] += torch.eye(dim, dtype=torch.float)
            self.adapter_conv2.weight.data[:, :, 1, 1] += torch.eye(dim, dtype=torch.float)
            self.adapter_conv3.weight.data[:, :, 2, 2] += torch.eye(dim, dtype=torch.float)


        nn.init.zeros_(self.adapter_conv1.bias)
        nn.init.zeros_(self.adapter_conv2.bias)
        nn.init.zeros_(self.adapter_conv3.bias)

        self.adapter_down = nn.Linear(n_embd, dim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(dim, n_embd)  # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

        self.bn = nn.BatchNorm2d(dim)
        self.sa = SA()

    def forward(self, x):

        B, H, W, C = x.shape

        x = x.view(B,-1,C)

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        x_patch = x_down.reshape(B, H, W, self.dim).permute(0, 3, 1, 2)

        x_patch1 = self.adapter_conv1(x_patch)
        x_patch2 = self.adapter_conv2(x_patch)
        x_patch3 = self.adapter_conv3(x_patch)

        x_patch = x_patch1 + x_patch2 + x_patch3
        x_patch = self.bn(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, H * W, self.dim)

        x_down = self.act(x_patch)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        x_up = x_up.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x_up = self.sa(x_up)
        x_up = x_up.permute(0, 2, 3, 1)


        return x_up


class Adapter_MDT(nn.Module):
    def __init__(self,
                 embedding_dim=96,
                 mlp_dim=96,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = embedding_dim
        self.mlp_dim = mlp_dim

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.conv_pass = Convpass(self.n_embd,self.mlp_dim)

        self.dropout = dropout


    def forward(self, x,add_residual=True, residual=None):

        residual = x if residual is None else residual

        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        x = self.conv_pass(x)       

        x = x * self.scale

        if self.adapter_layernorm_option == 'out':
            x = self.adapter_layer_norm_before(x)

        if add_residual:
            output = x + residual
        else:
            output = x

        return output



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class SA(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()

        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):

        residual = x

        out = x * self.sa(x)

        return out + residual
