from pathlib import Path
import numpy as np


class DataReader(object):
    def __init__(self, data_file):
        self.data_file = data_file
        self.num_train = 0
        self.xtrain = None
        self.ytrain = None

    # read data from file
    def read_data(self):
        train_file = Path(self.data_file)
        if train_file.exists():
            data = np.load(train_file)
            self.xtrain = data["data"]
            self.ytrain = data["label"]
            self.num_train = self.xtrain.shape[0]
        else:
            raise Exception("Cannot find train file!")

    def get_train_sample(self, iteration):
        return self.xtrain[iteration], self.ytrain[iteration]

    def get_batch_samples(self, batch_size, iteration):
        start = batch_size * iteration
        end = start + batch_size

        return self.xtrain[start:end, :], self.ytrain[start:end, :]

    def get_whole_samples(self):
        return self.xtrain, self.ytrain

    def shuffle(self):
        seed = np.random.randint(0, 100)
        np.random.seed(seed)
        # 随机排列一个序列，或者数组
        xp = np.random.permutation(self.xtrain)
        np.random.seed(seed)
        yp = np.random.permutation(self.ytrain)
        self.xtrain = xp
        self.ytrain = yp
