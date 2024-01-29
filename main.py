from model.main_model import TrafficPredictionModel
from util.earlystopping import EarlyStopping
import os, time, torch
from torch_geometric.loader import DataLoader
from util.index import compute_mae, compute_rmse, compute_wmape, compute_r2
import numpy as np
import csv
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epoch_num = 500
pre_day = 1
pre_leg = 10
batch_size = 48
lr = 0.0002
num_node_features = 10
num_classes = 1
num_tcn_features = 48  # TCN层的特征数量，可以调整,但要与batch_保持一致
global_start_time = time.time()
model_type = 'ours'
TIMESTAMP = str(time.strftime("%Y_%m_%d_%H_%M_%S"))
save_dir = './save_model/' + model_type + '_' + TIMESTAMP
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


data_list = torch.load('./dataset/graph_data_list1.pt')

# 计算各个子集的大小
train_size = int(0.7143 * len(data_list))  # 20天的训练集
test_size = pre_day * 144  # 一天的测试集
val_size = len(data_list) - train_size - test_size  # 验证集

# 按时间顺序划分数据集
train_data = data_list[:train_size]
test_data = data_list[train_size:test_size + train_size]
val_data = data_list[train_size + test_size:]

# 创建 DataLoader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)  # 对于时间序列数据，通常不打乱
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

model = TrafficPredictionModel(num_node_features, num_tcn_features, num_classes, pre_leg).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss().to(device)
temp_time = time.time()
early_stopping = EarlyStopping(patience=100, verbose=True)

rmse_list = []
mae_list = []
wmape_list, r2_list = [], []
# 训练和验证过程
for epoch in range(epoch_num):  # 100个训练周期
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

    model.eval()
    true_values, predicted_values = [], []
    with torch.no_grad():
        val_loss = 0
        for data in val_loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch)
            val_loss += criterion(output, data.y).item()
            true_values.append(data.y.cpu().numpy())
            predicted_values.append(output.cpu().numpy())
        val_loss /= len(val_loader)
        true_values = np.concatenate(true_values)
        predicted_values = np.concatenate(predicted_values)
        rmse = compute_rmse(true_values, predicted_values)
        mae = compute_mae(true_values, predicted_values)
        wmape = compute_wmape(true_values, predicted_values)
        r2 = compute_r2(true_values, predicted_values)
        rmse_list.append(rmse)
        mae_list.append(mae)
        wmape_list.append(wmape)
        r2_list.append(r2)
        # print(r2_list)
        print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')

    if epoch > 0:
        # early stopping
        model_dict = model.state_dict()
        early_stopping(val_loss, model_dict, model, epoch, save_dir)
        # 保存验证过程中的RMSE\MAE下降情况
        with open('./dataset/validation_metrics_combined.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'RMSE', 'MAE', 'WMAPE', 'R2', 'R2'])
            for i in range(len(rmse_list)):
                writer.writerow([i + 1, rmse_list[i], mae_list[i], wmape_list[i], r2_list[i]])
        if early_stopping.early_stop:
            print("Early Stopping")
            break
    # 每10个epoch打印一次训练时间
    if epoch % 10 == 0:
        print("time for 10 epoches:", round(time.time() - temp_time, 2))
        temp_time = time.time()
global_end_time = time.time() - global_start_time
print("global end time:", global_end_time)