import re
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import torch
from torch_geometric.data import Data

A_PATH = r'C:\Users\言射耳总\PycharmProjects\pytorch_learning\ISO-DTGCN\dataset\邻接矩阵1.csv'
X_path = r'C:\Users\言射耳总\PycharmProjects\pytorch_learning\ISO-DTGCN\dataset\flow_28days.csv'
id_path = r'C:\Users\言射耳总\PycharmProjects\pytorch_learning\ISO-DTGCN\dataset\id_list.csv'
A = np.loadtxt(A_PATH, delimiter=",")
X = np.loadtxt(X_path, delimiter=",")

A = csr_matrix(A)  # 替换为您的稀疏矩阵 A
X = np.array(X)    # 替换为您的特征矩阵 X

# 假设 subgraph_lists 是包含个路段编号列表的列表
subgraph_lists = []
window_size = 10
with open(id_path, 'r') as file:
    for line in file:
        line = line.strip()
        fields = re.split(r'\t|,', line)
        clean_fields = [field.strip() for field in fields if field.strip()]
        int_data = [int(x) for x in clean_fields]
        subgraph_lists.append(int_data)  # 替换为您的列表

# print(len(subgraph_lists))
data_list = []
window_size = 10  # 滑动窗口的长度
num_features = X.shape[1]  # 特征矩阵的列数
specific_node_index = 62

for i, subgraph_nodes in enumerate(subgraph_lists):
    # 从 A 中提取子图的边
    subgraph_edges = A[subgraph_nodes, :][:, subgraph_nodes].tocoo()

    # 转换边索引为 PyTorch 张量
    edge_index_np = np.vstack((subgraph_edges.row, subgraph_edges.col))
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    # 计算特征矩阵的列索引
    start_col_idx = i % num_features
    end_col_idx = start_col_idx + window_size
    next_col_idx = end_col_idx % num_features  # 紧随窗口的下一列

    # 检查是否需要从头开始填充
    if end_col_idx > num_features:
        # 从特征矩阵的开始处补充
        cols_to_wrap = end_col_idx - num_features
        x_subgraph = np.hstack((X[subgraph_nodes, start_col_idx:num_features],
                                X[subgraph_nodes, 0:cols_to_wrap]))
    else:
        # 直接提取特征子矩阵
        x_subgraph = X[subgraph_nodes, start_col_idx:end_col_idx]

    y_subgraph = X[specific_node_index, next_col_idx]

    x = torch.tensor(x_subgraph, dtype=torch.float)
    y = torch.tensor(y_subgraph, dtype=torch.float)

    # 创建 Data 对象
    data = Data(x=x, edge_index=edge_index, y=y)
    data_list.append(data)
torch.save(data_list, '../dataset/graph_data_list1.pt')