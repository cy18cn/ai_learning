import numpy as np
from data.data_reader import load_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from pathlib import Path

file_name = "mse"


def target_func(x, w, b):
    return w * x + b


def load_data(file):
    data = np.load(file)
    return data["x"], data["label"]


def create_sample_data(w, b, n):
    file = Path(file_name)
    if file.exists():
        x, y = load_data(file)
    else:
        x = np.linspace(0, 1, num=n)
        noise = np.random.uniform(-0.5, 0.5, size=n)
        y = target_func(x, w, b) + noise
        np.savez(file_name, data=x, label=y)
    return x, y


# 单样本： loss = (z - y) ^ 2
# m个样本多样本： loss = sum(zi-yi) / (2m)
def loss_func(y, z, count):
    c = (z - y) ** 2
    return c.sum()/count/2


def show_result(ax, x, y, a, loss, title):
    ax.scatter(x, y)
    ax.plot(x, a, 'r')
    titles = str.format("{0} Loss={1:01f}", title, loss)
    ax.set_title(titles)


def show_cost_for_4b(x, y, m, w, b):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    a1 = w * x + b - 1
    loss1 = loss_func(y, a1, m)
    show_result(ax1, x, y, a1, loss1, 'z=2x+2')

    a2 = w * x + b - 0.5
    loss2 = loss_func(y, a2, m)
    show_result(ax2, x, y, a2, loss2, 'z=2x+2.5')

    a3 = w * x + b
    loss3 = loss_func(y, a3, m)
    show_result(ax3, x, y, a3, loss3, 'z=2x+3')

    a4 = w * x + b + 0.5
    loss4 = loss_func(y, a4, m)
    show_result(ax4, x, y, a4, loss4, 'z=2x+3')

    plt.show()


def show_all_4b(x, y, m, w, b):
    plt.scatter(x, y)

    a1 = w * x + b - 1
    plt.plot(x, a1)

    a2 = w * x + b - 0.5
    plt.plot(x, a2)

    a3 = w * x + b
    plt.plot(x, a3)

    a4 = w * x + b + 0.5
    plt.plot(x, a4)

    plt.show()


# 显示b变化时，loss的变化
def calculate_cost_b(x, y, m, w, b):
    range_b = np.arange(b - 1, b + 1, 0.05)
    losses = []
    for i in range(len(range_b)):
        z = x * w + range_b[i]
        loss = loss_func(y, z, m)
        losses.append(loss)

    plt.title("Loss according to b")
    plt.xlabel("b")
    plt.ylabel("loss")
    plt.plot(range_b, losses, 'x')
    plt.show()


# 显示w变化时，loss的变化
def calculate_cost_w(x, y, m, w, b):
    range_w = np.arange(w - 1, w + 1, 0.05)
    losses = []
    for i in range(len(range_w)):
        z = range_w[i] * x + b
        loss = loss_func(y, z, m)
        losses.append(loss)

    plt.title("Loss according to w")
    plt.xlabel("w")
    plt.ylabel("Loss")
    plt.plot(range_w, losses, 'o')
    plt.show()


#
def calculate_cost_wb(x, y, m, w, b):
    range_w = np.arange(w - 10, w + 10, 0.1)
    range_b = np.arange(b - 10, b + 10, 0.1)
    losses = np.zeros((len(range_w), len(range_b)))
    for i in range(len(range_w)):
        for j in range(len(range_b)):
            z = range_w[i] * x + range_b[j]
            loss = loss_func(y, z, m)
            losses[i, j] = loss

    print(losses)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(range_w, range_b, losses)
    plt.show()


def show_3d_surface(x, y, m, w, b):
    fig = plt.figure()
    ax = Axes3D(fig)

    m_x = x.reshape(m, 1)
    m_y = y.reshape(m, 1)

    len1 = 50
    len2 = 50
    len = len1 * len2
    m_w = np.linspace(w - 2, w + 2, len1)
    m_b = np.linspace(b - 2, b + 2, len2)

    m_w, m_b = np.meshgrid(m_w, m_b)

    m = m_x.shape[0]
    z = np.dot(m_x, m_w.ravel().reshape(1, len)) + m_b.ravel().reshape(1, len)
    loss1 = (z - m_y) ** 2
    loss2 = loss1.sum(axis=0, keepdims=True)/2/m
    loss3 = loss2.reshape(len1, len2)
    ax.plot_surface(m_w, m_b, loss3, norm=LogNorm(), cmap='rainbow')
    plt.show()


def draw_contour(x, y, m, w, b):
    X = x.reshape(m, 1)
    Y = y.reshape(m, 1)

    len1 = 50
    len2 = 50
    len = len1 * len2
    W = np.linspace(w - 2, 2 + 2, len1)
    B = np.linspace(b - 2, b + 2, len2)
    W, B = np.meshgrid(W, B)
    loss = np.zeros((len1, len2))

    m = X.shape[0]
    Z = np.dot(X, W.ravel().reshape(1, len)) + B.ravel().reshape(1, len)
    loss1 = (Z - Y) ** 2
    loss2 = loss1.sum(axis=0, keepdims=True) / 2 / m
    loss3 = loss2.reshape(len1, len2)
    plt.contour(W, B, loss3, levels=np.logspace(-5, 5, 50), norm=LogNorm(), cmap=plt.cm.jet)
    plt.show()


if __name__ == '__main__':
    w = 2
    b = 3
    m = 50
    x, y = create_sample_data(w, b, m)
    plt.scatter(x, y)
    plt.show()

    show_cost_for_4b(x, y, m, w, b)
    show_all_4b(x, y, m, w, b)

    calculate_cost_b(x, y, m, w, b)
    calculate_cost_w(x, y, m, w, b)

    calculate_cost_wb(x, y, m, w, b)

    show_3d_surface(x, y, m, w, b)

    draw_contour(x, y, m, w, b)