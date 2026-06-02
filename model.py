import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2, padding_mode='replicate')
        self.bn1 = nn.BatchNorm1d(channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x += residual
        x = self.silu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, input_size=3, num_class=1):
        super(ResNet, self).__init__()
        # 第一层卷积块
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=5, padding=2, padding_mode='replicate'),  # 保持序列长度
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # 长度缩减为250
        )

        # 第二层卷积块
        self.conv2 = nn.Sequential(
            ResidualBlock(128),
            nn.MaxPool1d(kernel_size=2, stride=2)  # 长度缩减为125
        )

        # 第三层卷积块
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2, padding_mode='replicate'),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # 长度缩减为62
        )

        # 第四层卷积块
        self.conv4 = nn.Sequential(
            ResidualBlock(256),
            nn.MaxPool1d(kernel_size=2, stride=2)  # 长度缩减为31
        )

        # 第五层卷积块
        self.conv5 = nn.Sequential(
            ResidualBlock(256),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )

        # 自适应池化到固定长度，兼容不同输入序列长度（如 600/700/800/900/1000）
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),  # 较高丢弃率防止过拟合
            nn.Linear(256, 512),  # AdaptiveAvgPool1d(1) 后为 (B,256,1) -> 256 维
            nn.SiLU(),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_class),
        )

    def forward(self, x):
        # x: (batch_size, 1, input_length)
        x = self.conv1(x)  # (batch_size, 64, 250)
        x = self.conv2(x)  # (batch_size, 128, 125)
        x = self.conv3(x)  # (batch_size, 256, 63)
        x = self.conv4(x)  # (batch_size, 256, 32)
        x = self.conv5(x)
        x = self.adaptive_pool(x)  # (batch_size, 256, 1)
        x = self.classifier(x)  # (batch_size,)
        return x


# ----- 对比模型（输入 (B, C, L)，与 ResNet 相同） -----

import math
import torch.nn.init as init


class CNNsimple(nn.Module):
    """CNNsimple-AdaPool"""

    def __init__(self, input_size=3, num_class=1):
        super(CNNsimple, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=5, padding=2, padding_mode='replicate'),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2, padding_mode='replicate'),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2, padding_mode='replicate'),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=5, padding=2, padding_mode='replicate'),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=5, padding=2, padding_mode='replicate'),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_class),
        )

    def forward(self, x):
        x = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
        x = self.adaptive_pool(x)
        return self.classifier(x)


class BiLSTM_GAMP(nn.Module):
    """BiLSTM-GAMP"""

    def __init__(self, input_size=3, hidden_size=64, num_layers=1, num_class=1):
        super(BiLSTM_GAMP, self).__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc_combined = nn.Sequential(
            nn.Linear(2 * hidden_size, 128),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_class),
        )
        self._init_weights()

    def forward(self, x):
        x = x.transpose(1, 2)
        b = x.size(0)
        h0 = torch.zeros(1, b, self.hidden_size, device=x.device)
        c0 = torch.zeros(1, b, self.hidden_size, device=x.device)
        out1, (h1, c1) = self.lstm1(x, (h0, c0))
        out2, _ = self.lstm2(out1, (h1, c1))
        feat = torch.cat([out2.mean(dim=1), out2.max(dim=1).values], dim=1)
        return self.fc_combined(self.dropout(feat))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a=0.1)
                if m.bias is not None:
                    m.bias.data.zero_()


def _sinusoidal_pos_emb(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    emb = torch.zeros(seq_len, d_model)
    emb[:, 0::2] = torch.sin(position * div_term)
    emb[:, 1::2] = torch.cos(position * div_term)
    return emb


class Transformer_CLS(nn.Module):
    """Transformer-CLS"""

    def __init__(
        self,
        input_size=3,
        num_class=1,
        num_layers=2,
        d_model=128,
        num_heads=4,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=1024,
    ):
        super(Transformer_CLS, self).__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_size))
        self.input_proj = nn.Linear(input_size, d_model)
        self.register_buffer('position_emb', _sinusoidal_pos_emb(max_seq_len + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(d_model, 128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_class),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        b, seq_len, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.input_proj(x)
        x = x + self.position_emb[: seq_len + 1].unsqueeze(0).to(x.dtype)
        x = self.transformer_encoder(x)
        return self.classifier(x[:, 0, :])
