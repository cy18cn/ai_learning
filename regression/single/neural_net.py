import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from regression.single.training_history import TrainingHistory


class NeuralNet(object):
    def __init__(self, w, b, params):
        self.params = params
        self.w = w
        self.b = b

    def forward_batch(self, batch_x):
        return np.dot(batch_x, self.w) + self.b

    def backward_batch(self, batch_x, batch_y, batch_z):
        m = batch_x.shape[0]
        dz = batch_z - batch_y
        db = dz.sum(axis=0, keepdims=True) / m
        dw = np.dot(batch_x.T, dz) / m
        return dw, db

    def update(self, dw, db):
        self.w = self.w - self.params.eta * dw
        self.b = self.b - self.params.eta * db

    def inference(self, batch_x):
        return self.forward_batch(batch_x)

    def check_loss(self, data_reader):
        x, y = data_reader.get_whole_samples()
        m = x.shape[0]
        z = self.forward_batch(x)
        return ((y - z) ** 2).sum() / 2 / m

    def train(self, data_reader):
        loss_history = TrainingHistory()
        # batch_size默认为全量数据
        if self.params.batch_size == -1:
            self.params.batch_size = data_reader.num_train

        # 每一轮的迭代次数
        max_iteration = int(data_reader.num_train / self.params.batch_size)
        for epoch in range(self.params.max_epoch):
            print("epoch=%d" % epoch)
            data_reader.shuffle()
            for iteration in range(max_iteration):
                batch_x, batch_y = data_reader.get_batch_samples(self.params.batch_size, iteration)
                batch_z = self.forward_batch(batch_x)
                dw, db = self.backward_batch(batch_x, batch_y, batch_z)
                self.update(dw, db)

                if iteration % 2 == 0:
                    loss = self.check_loss(data_reader)
                    print(epoch, iteration, loss)
                    loss_history.add(epoch * max_iteration + iteration, loss, self.w, self.b)
                    if loss < self.params.eps:
                        break

            if loss < self.params.eps:
                break

            loss_history.show_history(self.params)
            print(self.w, self.b)
        self.show_contour(data_reader, loss_history, self.params.batch_size)

    def show_contour(self, data_reader, loss_history, batch_size):
        latest_loss, latest_iteration, latest_w, latest_b = loss_history.get_latest()
        len1 = 50
        len2 = 50
        # w坐标向量 [1, 2, 3]
        w = np.linspace(latest_w - 1, latest_w + 1, len1)
        # b坐标向量 [4, 5]
        b = np.linspace(latest_b - 1, latest_b + 1, len2)
        # 从坐标向量中返回坐标矩阵： w, b在坐标系中共有6个点(1,4) (2,4) (3,4) (1,5) (2,5) (3,5)
        # 返回坐标矩阵： [[1, 2, 3], [1, 2, 3]], [[4, 4, 4], [5, 5, 5]]
        w, b = np.meshgrid(w, b)

        len = len1 * len2
        x, y = data_reader.get_whole_samples()
        m = x.shape[0]
        # ravel 扁平化 w.ravel() [1, 2, 3, 1, 2, 3]
        z = np.dot(x, w.ravel().reshape(1, len)) + b.ravel().reshape(1, len)
        loss = (z - y) ** 2
        loss = loss.sum() / 2 / m
        loss = loss.reshape(len1, len2)
        plt.contour(w, b, loss, levels=np.logspace(-5, 5, 100), norm=LogNorm(), cmap=plt.cm.jet)

        #
        w_history = loss_history.w_history
        b_history = loss_history.b_history
        plt.plot(w_history, b_history)
        plt.xlabel("w")
        plt.ylabel("b")
        plt.title(str.format("batchsize={0}, iteration={1}, eta={2}, w={3:.3f}, b={4:.3f}",
                             batch_size, latest_iteration, self.params.eta, latest_w, latest_b))

        plt.axis([latest_w - 1, latest_w + 1, latest_b - 1, latest_b + 1])
        plt.show()

