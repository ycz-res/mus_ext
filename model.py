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

