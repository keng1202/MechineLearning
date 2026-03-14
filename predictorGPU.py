import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

input_size = 20
lr = 1e-5
epoch = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def load_dataset(path):
    data = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            data.append([float(x) for x in row[: input_size + 1]])
    array = np.asarray(data, dtype=np.float32)
    x = torch.from_numpy(array[:, :input_size]).to(device)
    y = torch.from_numpy(array[:, input_size]).to(device)
    return x, y


def evaluate(model, x_data, y_data):
    model.eval()
    with torch.no_grad():
        pred = model(x_data).squeeze(1)
        mse = torch.mean((pred - y_data) ** 2)
    return float(mse.item())


def save_model(model, path="modelGPU.csv"):
    layer = model[0]
    weight = layer.weight.detach().cpu().numpy().reshape(-1)
    bias = layer.bias.detach().cpu().numpy().reshape(-1)
    params = np.concatenate([weight, bias])
    np.savetxt(path, params, fmt="%.10f")


x_train, y_train = load_dataset("training.csv")
x_test, y_test = load_dataset("testing.csv")

model = nn.Sequential(nn.Linear(input_size, 1)).to(device)

# Try warm-start from model.csv if shape matches input_size + bias.
try:
    params = np.loadtxt("modelGPU.csv", dtype=np.float32)
    if params.shape[0] == input_size + 1:
        with torch.no_grad():
            model[0].weight.copy_(torch.from_numpy(params[:input_size]).view(1, -1).to(device))
            model[0].bias.copy_(torch.from_numpy(params[input_size:]).to(device))
except Exception:
    pass

optimizer = torch.optim.SGD(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# ... 前面代碼保持不變 ...

mse_history = []
print_interval = 1000 # 每 1000 次記錄一次，節省時間

for k in range(epoch):
    model.train()
    optimizer.zero_grad()
    out = model(x_train).squeeze(1)
    loss = criterion(out, y_train)
    loss.backward()
    optimizer.step()

    # 優化：不需要每輪都 evaluate，也不需要每輪都 print 到螢幕（IO 很慢）
    if k % print_interval == 0 or k == epoch - 1:
        mse = evaluate(model, x_test, y_test)
        mse_history.append(mse)
        print(f"epoch: {k}, testing MSE: {mse:.6f}")

save_model(model)

# 繪圖優化
plt.figure(figsize=(10, 5))
# 根據記錄間隔調整 X 軸
x_axis = [i * print_interval for i in range(len(mse_history))]
plt.plot(x_axis, mse_history, color="tab:blue", linewidth=1.5)

plt.yscale('log') # 建議加上這行，能更清楚看到 MSE 下降的過程
plt.title("MSE Curve by Epoch (Log Scale)")
plt.xlabel("Epoch")
plt.ylabel("Testing MSE (Log)")
plt.grid(True, which="both", linestyle="--", alpha=0.5) # 對數座標建議加這行 grid
plt.tight_layout()
plt.savefig("mse_curve_GPU.png", dpi=150)
