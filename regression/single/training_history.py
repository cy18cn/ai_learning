import matplotlib.pyplot as plt


class TrainingHistory(object):
    def __init__(self):
        self.iteration = []
        self.loss_history = []
        self.w_history = []
        self.b_history = []

    def add(self, iteration, loss, w, b):
        self.iteration.append(iteration)
        self.loss_history.append(loss)
        self.w_history.append(w)
        self.b_history.append(b)

    def show_history(self, params, xmin=None, xmax=None, ymin=None, ymax=None):
        plt.plot(self.iteration, self.loss_history)
        title = params.string()
        plt.title(title)
        plt.xlabel("iteration")
        plt.ylabel("loss")
        if xmin is not None and ymin is not None:
            plt.axis([xmin, xmax, ymin, ymax])
        plt.show()

        return title

    def get_latest(self):
        count = len(self.loss_history)
        return self.loss_history[count - 1], \
               self.iteration[count - 1], \
               self.w_history[count - 1], \
               self.b_history[count - 1]
