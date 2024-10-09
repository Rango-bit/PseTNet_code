import torch.nn as nn
import torch.nn.functional as F

from model.PseTNet.text_part.text_prompt import CustomCLIP, text_params
from model.PseTNet.image_part import weight_module

# text modules
class Text_Conv(nn.Module):
    def __init__(self, in_channel, text_last_dim=512 * 2,
                 down_factor: int = None, top_feature=False):
        super(Text_Conv, self).__init__()
        if down_factor:
            self.squeeze_c = in_channel // down_factor
        else:
            self.squeeze_c = in_channel
        if top_feature:
            self.mid_channel = self.squeeze_c * 4
        else:
            self.mid_channel = self.squeeze_c

        self.weight_module = weight_module((1, 1, int(4096/in_channel)))

        self.text_down = nn.Linear(text_last_dim, text_last_dim // 2)
        self.conv_squeeze = nn.Conv2d(in_channel, self.squeeze_c, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.squeeze_c)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_expand = nn.Conv2d(self.mid_channel, in_channel, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, text_features):
        if len(text_features.shape) == 3:
            text_features = text_features.unsqueeze(dim=0)

        text_features = self.weight_module(text_features, x)

        text_features = self.text_down(text_features)
        text_features = text_features.permute(0, 1, 3, 2).contiguous()

        text_features = text_features.view((self.mid_channel, self.squeeze_c, 1, 1))
        x = self.bn1(self.conv_squeeze(x))
        x = self.relu1(x)
        x = F.conv2d(x, text_features, stride=1, padding=0)
        x = self.relu2(self.bn2(self.conv_expand(x)))

        return x


class Text_Encoder(nn.Module):
    def __init__(self, keywords, n_vector, clip_params_path=''):
        super(Text_Encoder, self).__init__()
        self.text_features = CustomCLIP(n_vector, keywords, clip_params_path)
        text_state_dict = text_params(clip_params_path)
        self.missing_keys, self.unexpect_keys = self.text_features.load_state_dict(text_state_dict, strict=False)
        print('miss key:', self.missing_keys)
        print('unexpect key:', self.unexpect_keys)

    def forward(self):
        text_mid_features = self.text_features()
        return text_mid_features


class Double_TextConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, text_len, dim, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(Double_TextConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.LayerNorm([text_len, dim]),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.LayerNorm([text_len, dim]),
            nn.GELU()
        )


class Text_Projection(nn.Module):
    def __init__(self, input_text_len=77, text_len=64, input_dim=512):
        super(Text_Projection, self).__init__()
        self.text_project = nn.Sequential(
            nn.Conv1d(input_text_len, text_len, kernel_size=1, stride=1),
            nn.LayerNorm([text_len, 512]),
            nn.GELU(),
            nn.Linear(input_dim, 512),
            nn.LayerNorm([text_len, 512]),
            nn.LeakyReLU(),
        )
        self.text_down_1 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            Double_TextConv(in_channels=1, out_channels=4, text_len=text_len // 2, dim=input_dim // 2)
        )
        self.text_down_2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            Double_TextConv(in_channels=4, out_channels=16, text_len=text_len // 4, dim=input_dim // 4)
        )
        self.text_down_3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            Double_TextConv(in_channels=16, out_channels=64, text_len=text_len // 8, dim=input_dim // 8)
        )
        self.text_down_4 = nn.Sequential(
            Double_TextConv(in_channels=64, out_channels=64, text_len=text_len // 8, dim=input_dim // 8)
        )

    def forward(self, text_features):
        text_features_set = []
        text_features_1 = self.text_project(text_features)
        text_features_set.append(text_features_1)

        text_features_1 = text_features_1.unsqueeze(dim=0)
        text_features_2 = self.text_down_1(text_features_1)
        text_features_set.append(text_features_2)

        text_features_3 = self.text_down_2(text_features_2)
        text_features_set.append(text_features_3)

        text_features_4 = self.text_down_3(text_features_3)
        text_features_set.append(text_features_4)

        text_features_5 = self.text_down_4(text_features_4)
        text_features_set.append(text_features_5)

        return text_features_set