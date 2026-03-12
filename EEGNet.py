import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from utils import init_eegnet_weights

class EEGNet(nn.Module):

    def __init__(self,
                 n_classes: int,
                 Chans: int,
                 Samples: int,
                 kernLength: int,
                 F1: int,
                 D: int,
                 F2: int,
                 dropoutRate:  float,
                 is_init = True):
        super(EEGNet, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropoutRate

        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLength // 2 - 1,
                          self.kernLength - self.kernLength // 2, 0,
                          0)),  # left, right, up, bottom
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLength),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            # DepthwiseConv2d
            nn.Conv2d(in_channels=self.F1,
                      out_channels=self.F1 * self.D,
                      kernel_size=(self.Chans, 1),
                      groups=self.F1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            # SeparableConv2d
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F1 * self.D,
                      kernel_size=(1, 16),
                      stride=1,
                      groups=self.F1 * self.D,
                      bias=False),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F2,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropoutRate))

        # 注意：这里把 classifier 单独定义为 Linear，便于后续 expand
        self.feature_dim = self.F2 * (self.Samples // (4 * 8))  # Samples after pooling
        self.classifier = nn.Linear(in_features=self.feature_dim,
                                    out_features=self.n_classes,
                                    bias=True)
        # self.classifier_block = nn.Sequential(
        #     nn.Linear(in_features=self.F2 * (self.Samples // (4 * 8)),
        #             out_features=self.n_classes, bias=True))

        if is_init:
            self.apply(init_eegnet_weights)
            
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """仅返回 backbone 的特征向量（flatten 后）"""
        out = self.block1(x)
        out = self.block2(out)
        out = out.reshape(out.size(0), -1)
        return out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        # out = self.classifier(feat)
        return feat
        
# ====== 测试 ======
if __name__ == "__main__":
    batch_size = 4
    Samples = 1001   # 时间点
    Chans = 22      # 通道数
    n_classes = 2   # 假设要分4类

    # 随机造一个输入 [batch, Samples, Chans]
    x = torch.randn(batch_size, Samples, Chans)

    # 转换为 EEGNet 需要的 [batch, 1, Chans, Samples]
    x = x.permute(0, 2, 1)   # [32, 32, 128]
    x = x.unsqueeze(1)       # [32, 1, 32, 128]

    # 初始化模型
    model = EEGNet(
        n_classes=n_classes,
        Chans=Chans,
        Samples=Samples,
        kernLength=64,
        F1=16,
        D=2,
        F2=32,
        dropoutRate=0.5
    )

    # 前向传播
    y = model(x)
    print("输入 shape:", x.shape)
    print("输出 shape:", y.shape)  # 应该是 [32, n_classes]
    print(model.feature_dim)

    