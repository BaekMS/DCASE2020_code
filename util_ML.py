from util_normal import *
import numpy as np
from copy import deepcopy
import random
import math
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
import feature_extraction


class Loading_in_Batch():
    def __init__(self, train_data, feature_dir, specaug=False, load_in_batch=False, mean=None):
        self.train_data=train_data
        self.feature_dir=feature_dir

        self.specaugTF = specaug
        self.specaug_func = specaugment(data_shape=(3, 256, 423), T=115, WF=100, FF=60, mean=mean)
        self.load_in_batch=load_in_batch
        # self.numberlist=np.arange(len(train_ans))
        # self.split_set=None
        # self.data_shape=np.load(self.feature_dir + self.train_list[0]).shape
        #
        # temp=[]
        # res=int(len(train_ans)/batch_size)
        # rem=len(train_ans)%batch_size
        # for i in range(res):
        #     if i==0:
        #         if i<rem:
        #             temp.append(batch_size+1)
        #         else:
        #             temp.append(batch_size)
        #     else:
        #         if i<rem:
        #             temp.append(temp[i-1]+batch_size+1)
        #         else:
        #             temp.append(temp[i-1]+batch_size)
        #
        # self.batchlist=temp


    def making_data(self, index):
    #
    #     # data = np.zeros((self.batchlist[index], *self.data_shape), dtype=np.float32)
    #     # ans = np.zeros((self.batchlist[index]), dtype=np.long)
    #     #
    #     # for num, i in enumerate(self.split_set[index]):
    #     #
    #     #     b = np.load(self.feature_dir + self.train_list[i])
    #     #     ajp = self.train_ans[i]
    #     #     data[num, :, :, :] = b
    #     #     ans[num] = ajp
    #
        # data = np.load(self.feature_dir + self.train_data[0][index])[ :, :, 4:-4]

        if index>=len(self.train_data[1]):
            index=index-len(self.train_data[1])
            spec=True
        else:
            spec=False

        if self.load_in_batch:
            # data = np.load(self.feature_dir + self.train_data[0][index])[:,:,:-8]
            data = np.load(self.feature_dir + self.train_data[0][index])
        else:
            # data=self.train_data[0][index,:,:,:-8]
            data=self.train_data[0][index,:,:,:]
            # data = self.train_data[0][index]
        # print(data)

        if spec:
            # print(data.shape)

            data=self.specaug_func.augment(data)

        # data=(data-data.mean())/data.std()
        # print(data.shape)
        # data=data.squeeze(0)

        # feature_extraction.showing(data, 44100, 1024)
        # feature_extraction.showing(specaugment(data), 44100, 1024)

        # data=np.concatenate((data, specaugment(data)), axis=0)
        ans = self.train_data[1][index]

        ans = torch.nn.functional.one_hot(ans, num_classes=10).type(torch.float32)

        return data, ans

    def __getitem__(self, index):
        # print('tt')
        return self.making_data(index)

    def __len__(self):
        if self.specaugTF:
            return 2*len(self.train_data[1])
        else:
            return len(self.train_data[1])

class lr_scheduler():
    def __init__(self, optimizer,restart_num,start=0.1, last=1e-4,  last_epoch=-1):
        self.optimizer = optimizer
        self.linear=restart_num[1]
        self.restart_num=restart_num

        self.start = start
        self.last = last




    def step(self, epoch):
        # print(epoch)
        # exit()
        # if epoch<self.linear:
        #     m=((1e-5)-(1e-3))/(0-self.linear)
        #     n=(1e-5)-m*0
        #     lr=m*epoch+n
        #     # print(lr)
        #     # exit()
        #     # lr=1e-5
        # else:
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
            elif epoch >= self.restart_num[-1]:
                lr = self.optimizer.param_groups[0]['lr']

                break
        self.optimizer.param_groups[0]['lr'] = lr

class specaugment():
    def __init__(self, data_shape=(1,256,423), T=213, WF=100, FF=60, mean=None):
        self.data_shape=data_shape
        self.T=T
        self.WF=WF # 서로 바꾸는 범위
        self.FF=FF
        self.mean=mean

    def augment(self, data):

        W = random.randint(10, self.WF) # 범위 바꾸기
        start=random.randint(0, self.data_shape[-1]-W-1)


        temp = data[:, :, start:start+W] # data 떼어놓기
        # print(temp.shape)
        # print(W, start)
        move = random.randint(int(W*0.1), int(W*0.9)) # 서로 바꿀 중심, 범위 바꾸기
        # print(data[:, :, start:start + move].shape)
        # print(temp[:, :, -move:].shape)
        data[:, :, start:start + move] = temp[:, :, -move:]
        data[:, :, start + move:start+W] = temp[:, :, 0:-move]
        if self.mean==None:
            mid = data.mean(axis=(1,2), keepdims=True)
        else:
            mid=self.mean

        # num = random.randint(1, 3)
        # for k in range(num):
        # t = random.randint(10, self.T)
        t0 = random.randint(0, data.shape[-1] - self.T)
        data[:, :, t0:t0 + self.T] = mid

        num = random.randint(1, 2)
        for k in range(num):
            f = random.randint(40, self.FF)
            f0 = random.randint(0, data.shape[-2] - f)
            data[:, f0:f0 + f, :] = mid
        # print(data.shape)
        # feature_extraction.showing(data, 44100, 1024)
        return data



def mixup(train_set, ans):

    num_of_data=int(ans.shape[0]/2)
    # print(num_of_data)


    alpha = torch.ones((num_of_data, 1, 1, 1), dtype=torch.float32) * 0.2
    # print(alpha)
    # exit()

    train_set=torch.cat((alpha*train_set[:num_of_data,:,:,:]+(1-alpha)*train_set[-num_of_data:,:,:,:], (1-alpha)*train_set[:num_of_data,:,:,:]+alpha*train_set[-num_of_data:,:,:,:]), dim=0)
    alpha=alpha.view(-1,1)

    ans=torch.cat((alpha*ans[:num_of_data,:]+(1-alpha)*ans[-num_of_data:,:], (1-alpha)*ans[:num_of_data,:]+alpha*ans[-num_of_data:,:]), dim=0)

    return train_set, ans

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

        self.size_average = size_average

    def forward(self, logpt, target, mix_up=False, log_softmax=True):
        if mix_up==False:
            if logpt.dim()>2:
                logpt = logpt.view(logpt.size(0),logpt.size(1),-1)  # N,C,H,W => N,C,H*W
                logpt = logpt.transpose(1,2)    # N,C,H*W => N,H*W,C
                logpt = logpt.contiguous().view(-1,logpt.size(2))   # N,H*W,C => N*H*W,C
            target = target.view(-1,1)



            # logpt=

            logpt = logpt.gather(1,target)
            logpt = logpt.view(-1)
            if log_softmax:
                pt = logpt.data.exp()
                loss = -1 * self.alpha * (1 - pt) ** self.gamma * logpt
            else:
                aa = logpt.log()
                loss = -1 * self.alpha * (1 - logpt.data) ** self.gamma * aa

            # print(logpt)

            # print(aa)
            # print(logpt)
            # exit()
            #




            # print(loss.mean())
            # exit()

            if self.size_average: return loss.mean()
            else: return loss.sum()

        else:

            # if input.dim() > 2:
            #     input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            #     input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            #     input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C


            kk=target.nonzero(as_tuple=True)
            loss=-1*target[kk[0], kk[1]]*self.alpha*(1-logpt[kk[0], kk[1]].data.exp())**self.gamma*logpt[kk[0], kk[1]]


            if self.size_average:
                return loss.sum()/target.shape[0]
            else:
                return loss.sum()

