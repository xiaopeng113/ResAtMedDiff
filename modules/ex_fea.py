import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = (nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False))
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.LeakyReLU = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DownConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        return self.LeakyReLU(self.conv(x))


class UpConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        return self.LeakyReLU(self.conv(torch.cat([x1, x2], dim=1)))


class RSU(nn.Module):
    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()

        assert height >= 2
        self.conv_in = ConvBNReLU(in_ch, out_ch)

        encode_list = [DownConvBNReLU(out_ch, mid_ch, flag=False)]
        decode_list = [UpConvBNReLU(mid_ch * 2, mid_ch, flag=False)]
        for i in range(height - 2):
            encode_list.append(DownConvBNReLU(mid_ch, mid_ch))
            decode_list.append(UpConvBNReLU(mid_ch * 2, mid_ch if i < height - 3 else out_ch))

        encode_list.append(ConvBNReLU(mid_ch, mid_ch, dilation=2))
        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(x, x2)

        return x + x_in


class RSU4F(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        self.encode_modules = nn.ModuleList([ConvBNReLU(out_ch, mid_ch),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=8)])

        self.decode_modules = nn.ModuleList([ConvBNReLU(mid_ch * 2, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch * 2, out_ch)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)



        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(torch.cat([x, x2], dim=1))

        return x + x_in


class encoder(nn.Module):
    def __init__(self,in_ch=3):
        super(encoder, self).__init__()

        self.en_1 = RSU(7,in_ch,16,32)
        self.d1 = DownConvBNReLU(in_ch=32,out_ch=32)
        self.en_2 = RSU(6, 32, 16, 64)

        self.d2 = DownConvBNReLU(in_ch=64, out_ch=64)

        self.en_3 = RSU(5, 64, 16, 128)
        self.d3 = DownConvBNReLU(in_ch=128, out_ch=128)

        self.en_4 = RSU(4, 128, 16, 256)
        self.d4 = DownConvBNReLU(in_ch=256, out_ch=256)


        self.en_5 = RSU4F(256, 16, 512)
        self.d5 = DownConvBNReLU(in_ch=512, out_ch=512)

        self.en_6 = RSU4F(512, 16, 512)
        self.d6 = DownConvBNReLU(in_ch=512, out_ch=512)



    def forward(self,x):
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



