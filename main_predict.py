from model.main_model import TrafficPredictionModel
import os, torch
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from util.index import compute_mae, compute_rmse, compute_wmape, compute_r2
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


pre_day = 1
pre_leg = 10
batch_size = 48
lr = 0.0002
num_node_features = 10
num_classes = 1
num_tcn_features = 48
data_list = torch.load('./dataset/graph_data_list1.pt')
# 计算各个子集的大小
train_size = int(0.7143 * len(data_list))  # 20天的训练集
test_size = pre_day * 144  # 一天的测试集
# 按时间顺序划分数据集
test_data = data_list[train_size:test_size + train_size]
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 实例化
model = TrafficPredictionModel(num_node_features, num_tcn_features, num_classes, pre_leg).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss().to(device)

path = './save_model/ours_2024_01_19_09_59_27_combine/model_dict_checkpoint_170_168.37612043.pth'
checkpoint = torch.load(path)
model.load_state_dict(checkpoint, strict=True)
model.eval()
true_values, predicted_values = [], []
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        true_values.append(data.y.cpu().numpy())
        predicted_values.append(output.cpu().numpy())

# 转换为一维数组
true_values = np.concatenate(true_values)
predicted_values = np.concatenate(predicted_values)

# 计算评估指标
mae = compute_mae(true_values, predicted_values)
rmse = compute_rmse(true_values, predicted_values)
wmape = compute_wmape(true_values, predicted_values)
r2 = compute_r2(true_values, predicted_values)

# 输出评估指标
print(f'RMSE: {rmse}, R2: {r2}, MAE: {mae}, WMAPE: {wmape}')

# 绘制预测结果和真实值的比较图
plt.figure(figsize=(10, 6))
plt.plot(true_values, label='True Values')
plt.plot(predicted_values, label='Predicted Values', alpha=0.7)
plt.title('Comparison of True and Predicted Values')
plt.xlabel('Sample')
plt.ylabel('Traffic Flow')
plt.legend()
plt.show()

# 将数据组织成 CSV 格式需要的形式
import csv
rows = [
    ["ISO-DTGCN(combine)"] + predicted_values.tolist()
]

# 写入 CSV 文件
csv_file_path = r'C:\Users\言射耳总\Desktop\论文\论文\单步预测结果.csv'
with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(rows)