from typing import Dict
import torch
import torch.nn as nn

from model.PseTNet.image_part import DoubleConv, Down, Up, OutConv
from model.PseTNet.text_part.text_module import Text_Projection, Text_Conv, Text_Encoder


class Vision_Part(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(Vision_Part, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)

        self.text_project = Text_Projection(input_text_len=77, input_dim=512)

        self.text_conv_0 = Text_Conv(base_c, 512, top_feature=True)
        self.text_conv_1 = Text_Conv(base_c * 2, 256)
        self.text_conv_2 = Text_Conv(base_c * 4, 128, down_factor=2)
        self.text_conv_3 = Text_Conv(base_c * 8, 64, down_factor=4)
        self.text_conv_4 = Text_Conv(base_c * 8, 64, down_factor=4)

        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 8)
        self.down5 = Down(base_c * 8, base_c * 8)

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor, text_mid_features) -> Dict[str, torch.Tensor]:
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        text_features_set = self.text_project(text_mid_features)
        text_f1, text_f2, text_f3, text_f4, text_f5 = text_features_set

        x1 = self.in_conv(x)
        x1_mid = self.text_conv_0(x1, text_f1)
        x2 = self.down1(x1)
        x2_mid = self.text_conv_1(x2, text_f2)
        x3 = self.down2(x2)
        x3_mid = self.text_conv_2(x3, text_f3)
        x4 = self.down3(x3)
        x4_mid = self.text_conv_3(x4, text_f4)
        x5 = self.down4(x4)
        x5_mid = self.text_conv_4(x5, text_f5)
        x = self.up1(x5_mid, x4_mid)
        x = self.up2(x, x3_mid)
        x = self.up3(x, x2_mid)
        x = self.up4(x, x1_mid)
        logits = self.out_conv(x)

        return logits


class PseTNet_model(nn.Module):
    def __init__(self, num_classes=2, keywords='vertebrae', n_ctx=32,
                 clip_params_path='./model/PseTNet/text_part/CLIP/ViT-B-16.pt'
                 ):
        super(PseTNet_model, self).__init__()
        self.text_encoder = Text_Encoder(keywords, n_ctx, clip_params_path)
        self.vision_part = Vision_Part(num_classes=num_classes)

    def forward(self, x):
        text_mid_features = self.text_encoder()
        out = self.vision_part(x, text_mid_features)
        return out

if __name__ == '__main__':
    model = PseTNet_model()
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(out.shape)