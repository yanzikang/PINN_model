# 我来写一些PINN可以用的即插即用的模块

import torch
import torch.nn as nn

# MLP模块
class MLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=512, output_dim=1, num_layers=8):
        super(MLP, self).__init__()
        # 初始层
        self.ini_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        # 中间层
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        
        # 输出层
        self.out_net = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, input_tensor):
        ini_shape = x.shape
        y = self.ini_net(input_tensor)
        y = self.net(y)
        y = self.out_net(y)
        return y.view(ini_shape)

if __name__ == "__main__":
    model = MLP(input_dim=4, hidden_dim=512, output_dim=1, num_layers=8) # (输入维度，隐藏层维度，输出维度，网络层数)
    x = torch.randn(100)
    y = torch.randn(100)
    z = torch.randn(100)
    t = torch.randn(100)
    input_tensor = torch.stack([x, y, z, t], dim=-1)
    output_tensor = model(input_tensor)
    print(f"输入数据维度{input_tensor.shape}")
    print(f"输出数据维度{output_tensor.shape}")
