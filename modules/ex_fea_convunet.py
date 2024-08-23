import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            # nn.LeakyReLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

class Conv(nn.Module):
    def __init__(self, dim):
        super(Conv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, stride=1, groups=dim, padding_mode='reflect') # depthwise conv
        self.norm1 = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act1 = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.act2 = nn.GELU()
    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm1(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act1(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.norm2(x)
        x = self.act2(residual + x)

        return x

class ConvolutionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(ConvolutionLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1 * dilation,
                      dilation=(1 * dilation, 1 * dilation)),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.layer(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, layer_num=1):
        super(Up, self).__init__()
        C = in_channels // 2
        self.norm = nn.BatchNorm2d(C)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.gate = nn.Linear(C, 3 * C)
        self.linear1 = nn.Linear(C, C)
        self.linear2 = nn.Linear(C, C)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        layers = nn.ModuleList()
        for i in range(layer_num):
            layers.append(Conv(out_channels))
        self.conv = nn.Sequential(*layers)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.norm(x1)
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        #attention
        B, C, H, W = x1.shape
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        gate = self.gate(x1).reshape(B, H, W, 3, C).permute(3, 0, 1, 2, 4)
        g1, g2, g3 = gate[0], gate[1], gate[2]
        x2 = torch.sigmoid(self.linear1(g1 + x2)) * x2 + torch.sigmoid(g2) * torch.tanh(g3)
        x2 = self.linear2(x2)
        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)

        x = self.conv1x1(torch.cat([x2, x1], dim=1))
        x = self.conv(x)
        return x


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels, layer_num=1):
        layers = nn.ModuleList()
        for i in range(layer_num):
            layers.append(Conv(out_channels))
        super(Down, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            *layers,
            nn.BatchNorm2d(out_channels)
        )

class UNet1(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(UNet1, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            Conv(out_channels)
        )

        self.conv1 = ConvolutionLayer(out_channels, mid_channels, dilation=1)
        self.conv1_ = Conv(mid_channels)
        self.down1 = Down(mid_channels, mid_channels * 2, layer_num=1)

        self.conv2 = ConvolutionLayer( mid_channels * 2,  mid_channels * 2, dilation=1)
        self.conv2_ = Conv(mid_channels * 2)
        self.down2 = Down( mid_channels * 2,  mid_channels * 4, layer_num=2)

        self.conv3 = ConvolutionLayer(mid_channels * 4, mid_channels * 4, dilation=1)
        self.conv3_ = Conv(mid_channels * 4)
        self.down3 = Down(mid_channels * 4, mid_channels * 8, layer_num=3)

        self.conv4 = ConvolutionLayer(mid_channels * 8, mid_channels * 8, dilation=1)
        self.conv4_ = Conv(mid_channels * 8)
        self.down4 = Down(mid_channels * 8, mid_channels * 16, layer_num=4)

        self.conv5 = ConvolutionLayer(mid_channels * 16, mid_channels * 16, dilation=1)
        self.conv5_ = Conv(mid_channels * 16)
        self.down5 = Down(mid_channels * 16, mid_channels * 32, layer_num=5)

        self.conv6 = ConvolutionLayer(mid_channels * 32, mid_channels * 16, dilation=1)
        self.conv6_ = Conv(mid_channels * 16)
        self.conv7 = ConvolutionLayer(mid_channels * 16, mid_channels * 16, dilation=2)
        self.conv7_ = Conv(mid_channels * 16)
        self.at = Cov_SelfAttention(4,8,8,0.2)

        self.up1 = Up(mid_channels * 32,  mid_channels * 16)
        self.up2 = Up(mid_channels * 16, mid_channels * 8)
        self.up3 = Up(mid_channels * 8, mid_channels * 4)
        self.up4 = Up(mid_channels * 4, mid_channels * 2)
        self.up5 = Up(mid_channels * 2, mid_channels)
        self.conv8 = ConvolutionLayer(mid_channels *  32, mid_channels * 16, dilation=1)
        self.conv8_ = Conv(mid_channels * 16)
        self.conv9 = ConvolutionLayer(mid_channels * 32, mid_channels * 8, dilation=1)
        self.conv9_ = Conv(mid_channels * 8)
        self.conv10 = ConvolutionLayer(mid_channels * 16, mid_channels * 4, dilation=1)
        self.conv10_ = Conv(mid_channels * 4)
        self.conv11 = ConvolutionLayer(mid_channels * 8, mid_channels * 2, dilation=1)
        self.conv11_ = Conv(mid_channels * 2)
        self.conv12 = ConvolutionLayer(mid_channels * 4, mid_channels, dilation=1)
        self.conv12_ = Conv(mid_channels)
        self.conv13 = ConvolutionLayer(mid_channels * 2, out_channels, dilation=1)
        self.conv13_ = Conv(out_channels)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.conv1(x0)
        x1_ = self.conv1_(x1)
        x1 = x1 + x1_
        d1 = self.down1(x1)

        x2 = self.conv2(d1)
        x2_ = self.conv2_(x2)
        x2 = x2 + x2_
        d2 = self.down2(x2)

        x3 = self.conv3(d2)
        x3_ = self.conv3_(x3)
        x3 = x3 + x3_
        d3 = self.down3(x3)

        x4 = self.conv4(d3)
        x4_ = self.conv4_(x4)
        x4 = x4 + x4_
        d4 = self.down4(x4)

        x5 = self.conv5(d4)
        x5_ = self.conv5_(x5)
        x5 = x5 + x5_
        d5 = self.down5(x5)

        x6 = self.conv6(d5)
        x6_ = self.conv6_(x6)
        x6 = x6 + x6_

        x7 = self.conv7(x6)
        x7_ = self.conv7_(x7)
        x7 = x7 + x7_

        x8 = self.conv8(torch.cat((x7, x6), 1))
        x8_ = self.conv8_(x8)
        x8 = x8 + x8_
        x8 = self.at(x8)
        up1 = self.up1(x8, x5)
        x9 = self.conv9(torch.cat((up1, x5), 1))
        x9_ = self.conv9_(x9)
        x9 = x9 + x9_
        up2 = self.up2(x9, x4)

        x10 = self.conv10(torch.cat((up2, x4), 1))
        x10_ = self.conv10_(x10)
        x10 = x10 + x10_
        up3 = self.up3(x10, x3)

        x11 = self.conv11(torch.cat((up3, x3), 1))
        x11_ = self.conv11_(x11)
        x11 = x11 + x11_
        up4 = self.up4(x11, x2)

        x12 = self.conv12(torch.cat((up4, x2), 1))
        x12_ = self.conv12_(x12)
        x12 = x12 + x12_

        up5 = self.up5(x12, x1)

        x13 = self.conv13(torch.cat((up5, x1), 1))
        x13_ = self.conv13_(x13)
        x13 = x13 + x13_
        return x13 + x0


class UNet2(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels):
        super(UNet2, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            Conv(out_channels)
        )

        self.conv1 = ConvolutionLayer(out_channels, mid_channels, dilation=1)
        self.conv1_ = Conv(mid_channels)
        self.down1 = Down(mid_channels, mid_channels * 2, layer_num=1)

        self.conv2 = ConvolutionLayer(mid_channels * 2, mid_channels * 2, dilation=1)
        self.conv2_ = Conv(mid_channels * 2)
        self.down2 = Down(mid_channels * 2, mid_channels * 4, layer_num=2)

        self.conv3 = ConvolutionLayer(mid_channels * 4, mid_channels * 4, dilation=1)
        self.conv3_ = Conv(mid_channels * 4)
        self.down3 = Down(mid_channels * 4, mid_channels * 8, layer_num=3)

        self.conv4 = ConvolutionLayer(mid_channels * 8, mid_channels * 8, dilation=1)
        self.conv4_ = Conv(mid_channels * 8)
        self.down4 = Down(mid_channels * 8, mid_channels * 16, layer_num=4)

        self.conv5 = ConvolutionLayer(mid_channels * 16, mid_channels * 16, dilation=1)
        self.conv5_ = Conv(mid_channels * 16)
        self.conv6 = ConvolutionLayer(mid_channels * 16, mid_channels * 16, dilation=1)
        self.conv6_ = Conv(mid_channels * 16)
        self.at = Cov_SelfAttention(4,8,8,0.2)
        self.up1 = Up(mid_channels * 16, mid_channels * 8)
        self.up2 = Up(mid_channels * 8, mid_channels * 4)
        self.up3 = Up(mid_channels * 4, mid_channels * 2)
        self.up4 = Up(mid_channels * 2, mid_channels)
        self.conv7 = ConvolutionLayer(mid_channels * 32, mid_channels * 8, dilation=2)
        self.conv7_ = Conv(mid_channels * 8)
        self.conv8 = ConvolutionLayer(mid_channels * 16, mid_channels * 4, dilation=1)
        self.conv8_ = Conv(mid_channels * 4)
        self.conv9 = ConvolutionLayer(mid_channels * 8, mid_channels * 2, dilation=1)
        self.conv9_ = Conv(mid_channels * 2)
        self.conv10 = ConvolutionLayer(mid_channels * 4, mid_channels , dilation=1)
        self.conv10_ = Conv(mid_channels)
        self.conv11 = ConvolutionLayer(mid_channels * 2,out_channels, dilation=1)
        self.conv11_ = Conv(out_channels)

    def forward(self, x):
        x0 = self.in_conv(x)

        x1 = self.conv1(x0)
        x1_ = self.conv1_(x1)
        x1 = x1 + x1_
        d1 = self.down1(x1)

        x2 = self.conv2(d1)
        x2_ = self.conv2_(x2)
        x2 = x2 + x2_
        d2 = self.down2(x2)

        x3 = self.conv3(d2)
        x3_ = self.conv3_(x3)
        x3 = x3 + x3_
        d3 = self.down3(x3)

        x4 = self.conv4(d3)
        x4_ = self.conv4_(x4)
        x4 = x4 + x4_
        d4 = self.down4(x4)

        x5 = self.conv5(d4)
        x5_ = self.conv5_(x5)
        x5 = x5 + x5_
        x6 = self.conv6(x5)
        x6_ = self.conv6_(x6)
        x6 = x6 + x6_
        x7 = self.conv7(torch.cat((x6, x5), dim=1))
        x7_ = self.conv7_(x7)
        x7 = x7 + x7_
        x7 = self.at(x7)
        up1 = self.up1(x7, x4)

        x8 = self.conv8(torch.cat((up1, x4), dim=1))
        x8_ = self.conv8_(x8)
        x8 = x8 + x8_
        up2 = self.up2(x8, x3)

        x9 = self.conv9(torch.cat((up2, x3), dim=1))
        x9_ = self.conv9_(x9)
        x9 = x9 + x9_
        up3 = self.up3(x9, x2)

        x10 = self.conv10(torch.cat((up3, x2), dim=1))
        x10_ = self.conv10_(x10)
        x10 = x10 + x10_
        up4 = self.up4(x10, x1)

        x11 = self.conv11(torch.cat((up4, x1), dim=1))
        x11_ = self.conv11_(x11)
        x11 = x11 + x11_

        return x11 + x0


class UNet3(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels):
        super(UNet3, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            Conv(out_channels)
        )

        self.conv1 = ConvolutionLayer(out_channels, mid_channels, dilation=1)
        self.conv1_ = Conv(mid_channels)
        self.down1 = Down(mid_channels, mid_channels * 2, layer_num=1)

        self.conv2 = ConvolutionLayer(mid_channels * 2, mid_channels * 2, dilation=1)
        self.conv2_ = Conv(mid_channels * 2)
        self.down2 = Down(mid_channels * 2, mid_channels * 4, layer_num=2)

        self.conv3 = ConvolutionLayer(mid_channels * 4, mid_channels * 4, dilation=1)
        self.conv3_ = Conv(mid_channels * 4)
        self.down3 = Down(mid_channels * 4, mid_channels * 8, layer_num=3)

        self.conv4 = ConvolutionLayer(mid_channels * 8, mid_channels * 8, dilation=1)
        self.conv4_ = Conv(mid_channels * 8)
        self.conv5 = ConvolutionLayer(mid_channels * 8, mid_channels * 8, dilation=2)
        self.conv5_ = Conv(mid_channels * 8)
        self.at = Cov_SelfAttention(4,8,8,0.2)
        self.up1 = Up(mid_channels * 8, mid_channels * 4)
        self.up2 = Up(mid_channels * 4, mid_channels * 2)
        self.up3 = Up(mid_channels * 2, mid_channels)
        self.conv6 = ConvolutionLayer(mid_channels * 16, mid_channels * 4, dilation=1)
        self.conv6_ = Conv( mid_channels * 4)
        self.conv7 = ConvolutionLayer(mid_channels * 8, mid_channels * 2, dilation=1)
        self.conv7_ = Conv(mid_channels * 2)
        self.conv8 = ConvolutionLayer(mid_channels * 4, mid_channels, dilation=1)
        self.conv8_ = Conv(mid_channels)
        self.conv9 = ConvolutionLayer(mid_channels * 2, out_channels, dilation=1)
        self.conv9_ = Conv(out_channels)

    def forward(self, x):
        x0 = self.in_conv(x)

        x1 = self.conv1(x0)
        x1_ = self.conv1_(x1)
        x1 = x1 + x1_
        d1 = self.down1(x1)

        x2 = self.conv2(d1)
        x2_ = self.conv2_(x2)
        x2 = x2 + x2_
        d2 = self.down2(x2)

        x3 = self.conv3(d2)
        x3_ = self.conv3_(x3)
        x3 = x3 + x3_
        d3 = self.down3(x3)

        x4 = self.conv4(d3)
        x4_ = self.conv4_(x4)
        x4 = x4 + x4_

        x5 = self.conv5(x4)
        x5_ = self.conv5_(x5)
        x5 = x5 + x5_
        x6 = self.conv6(torch.cat((x5, x4), 1))
        x6_ = self.conv6_(x6)
        x6 = x6 + x6_
        x6 = self.at(x6)
        up1 = self.up1(x6, x3)

        x7 = self.conv7(torch.cat((up1, x3), 1))
        x7_ = self.conv7_(x7)
        x7 = x7 + x7_
        up2 = self.up2(x7, x2)

        x8 = self.conv8(torch.cat((up2, x2), 1))
        x8_ = self.conv8_(x8)
        x8 = x8 + x8_
        up3 = self.up3(x8, x1)

        x9 = self.conv9(torch.cat((up3, x1), 1))
        x9_ = self.conv9_(x9)
        x9 = x9 + x9_
        return x9 + x0


class UNet4(nn.Module):

    def __init__(self, in_channels, mid_channels=12, out_channels=32):
        super(UNet4, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            Conv(out_channels)
        )

        self.conv1 = ConvolutionLayer(out_channels, mid_channels, dilation=1)
        self.conv1_ = Conv(mid_channels)
        self.down1 = Down(mid_channels, mid_channels * 2, layer_num=1)

        self.conv2 = ConvolutionLayer(mid_channels * 2, mid_channels * 2, dilation=1)
        self.conv2_ = Conv(mid_channels * 2)
        self.down2 = Down(mid_channels * 2, mid_channels * 4, layer_num=2)

        self.conv3 = ConvolutionLayer(mid_channels * 4, mid_channels * 4, dilation=1)
        self.conv3_ = Conv(mid_channels * 4)

        self.conv4 = ConvolutionLayer(mid_channels * 4, mid_channels * 4, dilation=2)
        self.conv4_ = Conv(mid_channels * 4)
        self.conv5 = ConvolutionLayer(mid_channels * 8, mid_channels * 2, dilation=1)
        self.conv5_ = Conv(mid_channels * 2)
        self.at = Cov_SelfAttention(4,8,8,0.2)
        self.up1 = Up(mid_channels * 4, mid_channels * 2)
        self.up2 = Up(mid_channels * 2, mid_channels)
        self.conv6 = ConvolutionLayer(mid_channels * 4, mid_channels, dilation=1)
        self.conv6_ = Conv(mid_channels)
        self.conv7 = ConvolutionLayer(mid_channels * 2, out_channels, dilation=1)
        self.conv7_ = Conv(out_channels)

    def forward(self, x):
        """encode"""
        x0 = self.in_conv(x)

        x1 = self.conv1(x0)
        x1_ = self.conv1_(x1)
        x1 = x1 + x1_
        d1 = self.down1(x1)

        x2 = self.conv2(d1)
        x2_ = self.conv2_(x2)
        x2 = x2 + x2_
        d2 = self.down2(x2)

        x3 = self.conv3(d2)
        x3_ = self.conv3_(x3)
        x3 = x3 + x3_
        x4 = self.conv4(x3)
        x4_ = self.conv4_(x4)
        x4 = x4 + x4_
        """decode"""
        x5 = self.conv5(torch.cat((x4, x3), 1))
        x5_ = self.conv5_(x5)
        x5 = x5 + x5_
        x5 = self.at(x5)
        up1 = self.up1(x5, x2)

        x6 = self.conv6(torch.cat((up1, x2), 1))
        x6_ = self.conv6_(x6)
        x6 = x6 + x6_
        up2 = self.up2(x6, x1)

        x7 = self.conv7(torch.cat((up2, x1), 1))
        x7_ = self.conv7_(x7)
        x7 = x7 + x7_
        return x7 + x0


class UNet5(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels):
        super(UNet5, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            Conv(out_channels)
        )

        self.conv1 = ConvolutionLayer(out_channels, out_channels, dilation=1)
        self.conv1_ = Conv(out_channels)
        self.conv2 = ConvolutionLayer(out_channels, out_channels, dilation=2)
        self.conv2_ = Conv(out_channels)
        self.conv3 = ConvolutionLayer(out_channels, out_channels, dilation=4)
        self.conv3_ = Conv(out_channels)
        self.conv4 = ConvolutionLayer(out_channels, out_channels, dilation=8)
        self.conv4_ = Conv(out_channels)
        self.conv5 = ConvolutionLayer(out_channels * 2, out_channels, dilation=4)
        self.conv5_ = Conv(out_channels)
        self.conv6 = ConvolutionLayer(out_channels * 2, out_channels, dilation=2)
        self.conv6_ = Conv(out_channels)
        self.conv7 = ConvolutionLayer(out_channels * 2, out_channels, dilation=1)
        self.conv7_ = Conv(out_channels)

    def forward(self, x):
        x0 = self.in_conv(x)

        x1 = self.conv1(x0)
        x1_ = self.conv1_(x1)
        x1 = x1 + x1_
        x2 = self.conv2(x1)
        x2_ = self.conv2_(x2)
        x2 = x2 + x2_
        x3 = self.conv3(x2)
        x3_ = self.conv3_(x3)
        x3 = x3 + x3_
        x4 = self.conv4(x3)
        x4_ = self.conv4_(x4)
        x4 = x4 + x4_

        x5 = self.conv5(torch.cat((x4, x3), 1))
        x5_ = self.conv5_(x5)
        x5 = x5 + x5_
        x6 = self.conv6(torch.cat((x5, x2), 1))
        x6_ = self.conv6_(x6)
        x6 = x6 + x6_
        x7 = self.conv7(torch.cat((x6, x1), 1))
        x7_ = self.conv7_(x7)
        x7 = x7 + x7_
        return x7 + x0

class encoder(nn.Module):

    def __init__(self, in_ch=3):
        super(encoder, self).__init__()

        self.en_1 = UNet1(in_ch, 16, 32)
        self.d1 = Down(32, 32)

        self.en_2 = UNet2(32, 16, 64)
        self.d2 = Down(64, 64)

        self.en_3 = UNet3(64, 16, 128)
        self.d3 = Down(128, 128)

        self.en_4 = UNet4(128, 16, 256)
        self.d4 = Down(256, 256)

        self.en_5 = UNet5(256, 16, 512)
        self.d5 = Down(512, 512)

        self.en_6 = UNet5(512, 16, 512)

    def forward(self, x):
        # ------encode ------
        end = []

        x1 = self.en_1(x)
        end.append(x1)
        x1 = self.d1(x1)

        x2 = self.en_2(x1)
        end.append(x2)
        x2 = self.d2(x2)

        x3 = self.en_3(x2)
        end.append(x3)
        x3 = self.d3(x3)

        x4 = self.en_4(x3)
        end.append(x4)
        x4 = self.d4(x4)

        x5 = self.en_5(x4)
        end.append(x5)
        x5 = self.d5(x5)

        x6 = self.en_6(x5)
        end.append(x6)

        return end

