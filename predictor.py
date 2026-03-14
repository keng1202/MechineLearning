import csv
import matplotlib.pyplot as plt
import numpy as np

input_size = 50
lr = 0.00001  # learning rate
epoch = 1000


def load_dataset(path):
    data = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            data.append([float(x) for x in row[: input_size + 1]])
    return data


def evaluate(weight, dataset):
    total_err = 0.0
    for row in dataset:
        output = sum(weight[j] * row[j] for j in range(input_size)) + weight[input_size]
        err = row[input_size] - output
        total_err += err**2
    return total_err / len(dataset)


def save_model(weight, path="model.csv"):
    with open(path, "w", newline="") as fout:
        for w in weight:
            fout.write(str(w) + "\n")


train_par = load_dataset("training.csv")
test_par = load_dataset("testing.csv")

# kind of pretrain, if there is a model.csv, then use it directly
try:
    weight = np.loadtxt("model.csv").astype(float).tolist()
    if len(weight) != input_size + 1:
        raise ValueError("model length mismatch")
except Exception:
    weight = [1.0 for _ in range(input_size + 1)]


# start training and collect testing MSE by epoch
mse_history = []
for k in range(epoch):
    for row in train_par:
        output = sum(weight[j] * row[j] for j in range(input_size)) + weight[input_size]
        err = row[input_size] - output

        # cal gradient = err*parameter, then, change weight
        for j in range(input_size):
            gradient = err * row[j]
            weight[j] = weight[j] + gradient * lr
        weight[input_size] = weight[input_size] + err * lr

    mse = evaluate(weight, test_par)
    mse_history.append(mse)
    print(f"epoch: {k}, testing MSE: {mse:.6f}")


save_model(weight)

# draw and save mse curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, epoch + 1), mse_history, color="tab:blue", linewidth=1.5)
plt.title("MSE Curve by Epoch")
plt.xlabel("Epoch")
plt.ylabel("Testing MSE")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mse_curve.png", dpi=150)
print("MSE curve saved to mse_curve.png")
