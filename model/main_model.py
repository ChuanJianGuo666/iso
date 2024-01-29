import torch
from torch import nn
from model.TCN_layers import TemporalConvNet
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool


class TrafficPredictionModel(nn.Module):
    def __init__(self, num_node_features, num_tcn_features, num_classes, pre_leg):
        super(TrafficPredictionModel, self).__init__()
        # GCN层
        self.gcn1 = GCNConv(num_node_features, pre_leg)  # batch_sum*pre_leg
        self.gcn2 = GCNConv(pre_leg, pre_leg)   # batch_sum*pre_leg
        self.gcn3 = GCNConv(pre_leg, pre_leg)   # batch_sum*pre_leg,经过pooling层后维度为[batch, pre_leg]

        # TCN层
        self.tcn = TemporalConvNet(48, [num_tcn_features, num_tcn_features, num_tcn_features, num_tcn_features], kernel_size=3)  # 经过squeeze后输出为[num_tcn_features, pre_leg]

        # 线性层
        self.fc1 = nn.Linear(pre_leg*3, 24)
        self.fc2 = nn.Linear(24, 12)
        self.fc3 = nn.Linear(12, num_classes)  # num_tcn_features//2*1

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):
        x = self.relu(self.gcn1(x, edge_index))
        x = self.relu(self.gcn2(x, edge_index))
        x = self.relu(self.gcn3(x, edge_index))

        # 三种池化方法
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)

        # 拼接池化结果
        x = torch.cat([x_mean, x_max, x_sum], dim=1)
        x = x.unsqueeze(0)

        tcn_out = self.tcn(x)
        tcn_out = tcn_out.squeeze(0)

        out = self.relu(self.fc1(tcn_out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        out = out.T
        out = out.squeeze(0)
        return out