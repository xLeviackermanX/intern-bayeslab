class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, loss_function):
        self.generator = generator
        self.loss_function = loss_function

    def __call__(self, x, y, norm):

        x = self.generator(x)

        loss = self.loss_function(x.contiguous().view(-1, x.size(-1)),
                                  y.contiguous().view(-1)) / norm

        return loss.data * norm