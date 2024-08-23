import math

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import classifier

# from ex_fea import encoder
from modules.ex_fea_convunet import encoder
import timm
import torchvision.transforms as transforms

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class Cov_SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
        super(Cov_SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(hidden_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        x = x.permute(0, 1, 3, 2, 4)
        return x

    def forward(self, input_tensor):
        input_tensor=input_tensor.float()
        mixed_query_layer = self.query(input_tensor.float())
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 1, 3, 2, 4).contiguous()
        context_layer = einops.rearrange(context_layer, 'b c w h1 h2 -> b c w (h1 h2)')
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 2, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):

        if self.residual:

            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.GELU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),

        )

        self.emb_layer = nn.Sequential(
            nn.GELU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb



class UNet_conditional(nn.Module):
    def __init__(self, c_in=4, c_out=4,con_c_in=3, time_dim=256, device="cuda"):
        super().__init__()
        self.encoder = encoder(in_ch=con_c_in)
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 32)  # 64
        self.down1 = Down(32, 64)  # 64 128 112

        self.down2 = Down(64, 128)  # 128  256  56

        self.down3 = Down(128, 256)  # 256  256 28

        self.down4 = Down(256, 512)  # 256  256 14

        self.down5 = Down(512, 512)  # 256  256 7

        self.bot1 = DoubleConv(512, 1024)  # 256  512
        self.bot2 = DoubleConv(1024, 1024)  # 512  512
        self.bot3 = DoubleConv(1024, 512)  # 512  256

        self.up1 = Up(1024, 256)  # 512  256

        self.up2 = Up(512, 128)  # 256 64

        self.up3 = Up(256, 64)  # 128 64

        self.up4 = Up(128, 32)  # 128 64
        self.up5 = Up(64, 32)  # 128 64
        self.at0 = Cov_SelfAttention(4,256,256,0.2)
        self.at1 = Cov_SelfAttention(4,128,128,0.2)
        self.at2 = Cov_SelfAttention(4,64,64,0.2)
        self.sa1 = SelfAttention(1024, 8)
        self.sa2 = SelfAttention(1024, 8)
        self.sa3 = SelfAttention(512, 8)
        self.gelu = nn.GELU()

        self.outc = nn.Conv2d(32, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        ex_x_fea = self.encoder(y)


        ex_x_fea[0] =self.at0(ex_x_fea[0])
        x1 = self.inc(x) + ex_x_fea[0]

        ex_x_fea[1] =  self.at1(ex_x_fea[1])

        x2 = self.down1(x1, t) + ex_x_fea[1]

        ex_x_fea[2] = self.at2(ex_x_fea[2])

        x3 = self.down2(x2, t) + ex_x_fea[2]
        x4 = self.down3(x3, t) + ex_x_fea[3]
        x5 = self.down4(x4, t) + ex_x_fea[4]
        x6 = self.down5(x5, t) + ex_x_fea[5]

        x6 = self.bot1(x6)
        x6 = self.sa1(x6)
        x6 = self.bot2(x6)
        x6 = self.sa2(x6)
        x6 = self.bot3(x6)
        x6 = self.sa3(x6)

        x = self.up1(x6, x5, t)

        x = self.up2(x, x4, t)

        x = self.up3(x, x3, t)

        x = self.up4(x, x2, t)

        x = self.up5(x, x1, t)
        output = self.outc(x)

        return output