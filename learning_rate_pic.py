import math
from util_train import *
from util_normal import *
import matplotlib.pyplot as plt

class lr_scheduler():
    def __init__(self, optimizer,restart_num,start=0.1, last=1e-4,  last_epoch=-1):
        self.optimizer = optimizer
        self.linear=restart_num[1]
        self.restart_num=restart_num

        self.start = start
        self.last = last


    def step(self, epoch):

        for num in range(len(self.restart_num)):
            if epoch == self.restart_num[num] and epoch != self.restart_num[-1]:
                lr = self.start * 0.9 ** num
                break

            elif epoch < self.restart_num[num]:
                top = self.start * 0.9 ** (num - 1)

                looplast = top * self.last
                b =self.restart_num[num - 1]
                lastnum = self.restart_num[num] - 1

                w = math.pi / (lastnum - b)
                c = (top + looplast) / 2
                a = top - c
                lr = a * math.cos(w * (epoch - b)) + c

                break
            # elif epoch >= self.restart_num[-1]:
            #     # lr = self.optimizer.param_groups[0]['lr']
            #     return lr
            #     break
        # print(lr)
        return lr

hyparam = open_yaml('./hyparam.yaml')
aa=lr_scheduler(None, hyparam['restart_num'])
at=[]

for x in range(260):
    at.append(aa.step(x))

plt.plot(at)
plt.show()