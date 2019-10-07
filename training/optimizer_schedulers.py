class NoamOpt:
    "Wraps the Noam rate scheduler around the optimizer"
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        print rate
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Adjust rate"
        if step is None:
            step = self._step
        return self.factor * (min(step ** (-0.5), step * self.warmup ** (-1.5)))

class StepOpt:
    "Wraps a simple step-based rate scheduler around the optimizer"
    def __init__(self, initial_lr, optimizer, lr_decay, steps_between_lr_decay):
        self.optimizer = optimizer
        self._step = 0
        self._rate = initial_lr
        self._lr_decay = lr_decay
        self._steps_between_lr_decay = steps_between_lr_decay

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self):
        steps_between_learning_rate_decay = self._steps_between_lr_decay
        if self._step % steps_between_learning_rate_decay == 0:
            print "lowering lr"
            return self._rate * self._lr_decay
        return self._rate