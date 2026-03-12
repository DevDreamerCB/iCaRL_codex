import torch.nn as nn
import torch

class Network(nn.Module):

    def __init__(self, numclass, feature_extractor, is_init=True):
        super(Network, self).__init__()
        # 分离特征提取器和全连接层
        self.feature = feature_extractor
        self.fc = nn.Linear(feature_extractor.feature_dim, numclass, bias=True)
        # nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
        # nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, input, return_feat=False):
        feature = self.feature(input)
        logit = self.fc(feature)
        if return_feat:
            return feature, logit
        else:
            return logit

    def Incremental_learning(self, numclass):
        old_weight = self.fc.weight.detach().clone()
        old_bias = self.fc.bias.detach().clone()
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        with torch.no_grad():
            self.fc = nn.Linear(in_feature, numclass, bias=True)
            self.fc.weight[:out_feature] = old_weight
            self.fc.bias[:out_feature] = old_bias
    
    def feature_extractor(self,inputs):
        return self.feature(inputs)

if __name__ == "__main__":
    pass
    # feature_extractor = EEGNet(n_classes=2,Chans=22,Samples=1001,kernLength=64,F1=16,D=2,F2=32,dropoutRate=0.5)
    # # network = Network(numclass = 2, feature_extractor=feature_extractor)

    # # X = torch.randn(size=(4,1,22,1001))
    # # print(network(X).shape)

    # import torch
    # import torch.nn as nn
    # # 假设你已有 Network 类和 feature_extractor 实例
    # model = Network(numclass=2, feature_extractor=feature_extractor)
    # # 随机输入，shape 根据你的数据调整
    # dummy = torch.randn(4, 1, 22, 1001)  # 举例：batch=4, 1通道, 22x112（按你数据改）
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # # 前向、反向
    # optimizer.zero_grad()
    # out = model(dummy)            # (4, 2)
    # loss = out.softmax(dim=1)[:,0].mean()   # 简单损失示例
    # loss.backward()

    # # 检查梯度
    # for name, p in model.named_parameters():
    #     print(name, "requires_grad:", p.requires_grad, " grad_none:", p.grad is None,
    #         " grad_norm:", None if p.grad is None else p.grad.norm().item())
