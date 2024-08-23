import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
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
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
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
            nn.MaxPool3d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None,None, None].repeat(1, 1, x.shape[-3],x.shape[-2], x.shape[-1])

        return x + emb

class incondition__(nn.Module):
    def __init__(self,in_channels = 4,out_channels=32,emb_dim=256,t_out=16,t_in=8):
        super(incondition__, self).__init__()
        self.conv_in = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )
        self.Turn_Time = nn.Sequential(
            DoubleConv(t_in, t_in, residual=True),
            DoubleConv(t_in, t_out),
        )


    def forward(self,x,t):
        x = self.conv_in(x)
        x = einops.rearrange(x,"b c t w h -> b t c w h")
        x = self.Turn_Time(x)
        x = einops.rearrange(x,"b c t w h -> b t c w h")
        emb = self.emb_layer(t)[:, :, None,None, None].repeat(1, 1, x.shape[-3],x.shape[-2], x.shape[-1])
        return x+emb



class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
               emb_dim,
               out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)

        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None, None].repeat(1, 1, x.shape[-3], x.shape[-2], x.shape[-1])
        return x + emb


class UNet_conditional_3d(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256,  device="cuda"):
        super().__init__()

        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 16)# 64
        self.down1 = Down(16, 32)# 64 128
        self.incondition = incondition__(in_channels=1,out_channels=16)
        #
        self.down2 = Down(32, 64)# 128  256
        #
        self.down3 = Down(64, 64)# 256  256
        #
        #
        self.bot1 = DoubleConv(64, 128)# 256  512
        self.bot2 = DoubleConv(128, 128)# 512  512
        self.bot3 = DoubleConv(128, 64)# 512  256
        #
        self.up1 = Up(128, 32)# 512  256

        self.up2 = Up(64, 16)# 256 64

        self.up3 = Up(32, 16)# 128 64
        #
        self.outc = nn.Conv3d(16, c_out, kernel_size=1)
        # self.inLiner = nn.Linear(in_features=1024,out_features=256)


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
        #
        # if y is not None:
        #     t += self.label_emb(y)
        x = torch.unsqueeze(x,dim=1)



        x1 = self.inc(x)
        # print(x.shape)
        #
        y = torch.unsqueeze(y,dim=1)
        y = self.incondition(y,t)

        x1 = x1+y
        x2 = self.down1(x1, t)

        x3 = self.down2(x2, t)

        x4 = self.down3(x3, t)

        #
        #
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)


        x = self.up1(x4, x3, t)

        x = self.up2(x, x2, t)

        #
        x = self.up3(x, x1, t)

        #
        output = self.outc(x)

        return torch.squeeze(output)


