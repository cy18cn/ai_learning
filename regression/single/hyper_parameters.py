class HyperParameters(object):
    def __init__(self, input_size, output_size, eta=0.1, max_epoch=10000, batch_size=5, eps=0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.eta = eta
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.eps = eps

    def string(self):
        return str.format("bz:{0},eta:{1}", self.batch_size, self.eta)