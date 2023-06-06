import matplotlib.pyplot as plt
import numpy as np


def draw_acc_loss_by_epoch(file_path, ylabel):
    """
    draw acc/loss curve by epoch.
    :param file_path: acc file or loss file path.
    :param ylabel: the name of ylabel.
    :return:
    """
    file = open(file_path, 'r')
    datas = []
    i = 0
    for line in file:
        line = line.strip('\n')
        line = line.split(' ')
        datas.append(line[0])
        i += 1
    file.close()

    datas = np.array(datas, dtype=float)

    X = np.linspace(0, i - 1, i)
    Y = datas

    plt.figure(figsize=(8, 6))
    plt.title("Train Result")
    plt.xlabel("Train Epoch")
    plt.ylabel(ylabel)
    plt.plot(X, Y)
    plt.show()


if __name__ == "__main__":
    draw_acc_loss_by_epoch(file_path="save11/loss.txt", ylabel="Train Loss")
