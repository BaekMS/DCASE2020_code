from util_normal import *
from util_ML import *
import transformer_trident
import torch
import pandas as pd
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from time import time
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import sys
from tqdm import tqdm
import math

def train(loader, model, optimizer, mode, hyparam,total_batch=100, loss_func=None):
    # print('ttt')
    # exit()
    if mode == "train":
        model.train()
    elif mode == "eval":
        model.eval()

    total_ans=None
    total_result=None
    total_loss=0
    loss_list=[]
    for batch_num, (train_data, ans) in tqdm(enumerate(loader), total=total_batch, desc=mode):
        # print(train_data.shape, ans.shape)
        # exit()

        if mode=='train':
            # print(train_data[-1])
            mix_size=int(hyparam['batch_size']/2)
            # print(mix_size)
            # exit()
            new_t, new_a=mixup(train_data[mix_size:,:,:,:], ans[4:,:])
            train_data[mix_size:,:,:,:]=new_t
            ans[mix_size:,:]=new_a
            # print(train_data[-1])
            # exit()
            # train_data=torch.cat((train_data, new_t), dim=0)
            # ans=torch.cat((ans, new_a), dim=0)



        train_data = normalize(train_data, hyparam).to(hyparam['device'])
        # if train_data.shape[1]==1:
        #     train_data = normalize(train_data).to(hyparam['device'])
        #     tgt=None
        # else:
        #     # tgt=normalize(train_data[:, 1,:,:]).to(hyparam['device'])
        #     # train_data=normalize(train_data[:, 0,:,:]).to(hyparam['device'])
        #     tgt=train_data[:,1,:,:].to(hyparam['device'])
        #     train_data=train_data[:, 0,:,:].to(hyparam['device'])

        if mode == "train":
            # output = model(train_data, tgt)
            output = model(train_data)

        elif mode=="eval":
            with torch.no_grad():
                # output = model(train_data, tgt)
                output = model(train_data)

        batch_size_now = train_data.shape[0]
        ans = ans.to(hyparam['device'])
        # print(output.shape)
        # print(ans.shape)
        # exit()
        # exit()
        if loss_func==None:
            loss=F.nll_loss(output, ans)
        else:
            if mode=='train':
                loss=loss_func(output, ans,mix_up=True, log_softmax=True)
            else:
                loss=loss_func(output, ans, log_softmax=True)

        if mode == "train":

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print(loss)
        # exit(
        if mode=='train':
            ans = ans.to('cpu').max(dim=1)[1]
        else:
            ans=ans.to('cpu')
        # print(output.shape)
        # print(ans.max(dim=1))
        # exit()
        if total_result==None:

            total_result=output.max(dim=1)[1].to('cpu')==ans
            total_ans=ans
        else:
            total_result=torch.cat((total_result, output.max(dim=1)[1].to('cpu')==ans), dim=0)
            total_ans=torch.cat((total_ans, ans), dim=0)

        total_loss += loss.item() * batch_size_now
        # print(total_loss)

    accuracy, total_result=train_result(total_result, total_ans)

    return total_loss, accuracy, total_result

def data_prepare(hyparam):
    # af=[0,0,0]
    # print(min(af))
    # exit()
    # hyparam['mean']=torch.zeros((3,1,1), dtype=torch.float32)
    # hyparam['std'] = torch.zeros(( 3, 1, 1), dtype=torch.float32)

    record_param = {'accu_list_train': [], 'accu_list_eval': [], 'loss_list_train': [], 'loss_list_eval': []}
    train_data, eval_data = csv_reader(hyparam['train_metadata'])
    train_list = train_data[0][0:3]

    if hyparam['specaugment']:
        train_size = len(train_data[0]) * 2
        for x in range(len(train_data[2])):
            train_data[2][x] *= 2
    else:
        train_size = len(train_data[0])

    if hyparam['load_batch'] == False:
        print('Data Loading...\n')
        train_data[0] = feature_load(train_data[0], hyparam['feature_dir'])
    else:

        calc_mean_std(train_data[0], hyparam)
        # print(hyparam['mean'])
        # print(hyparam['std'])
        # exit()

    train_loader = Loading_in_Batch(train_data, hyparam['feature_dir'], specaug=hyparam['specaugment'],
                                    load_in_batch=hyparam['load_batch'], mean=hyparam['mean'])
    train_loader = DataLoader(train_loader, batch_size=hyparam['batch_size'], num_workers=4, pin_memory=True,
                              drop_last=hyparam['droplast'], shuffle=True)

    if hyparam['droplast']:
        train_size = train_size // hyparam['batch_size']

    # eval_data[0] = feature_load(eval_data[0], hyparam['feature_dir'])[:,:,:,:-8]
    eval_data[0] = feature_load(eval_data[0], hyparam['feature_dir'])
    eval_data[0]=torch.tensor(eval_data[0])
    # print(eval_data[0].shape)
    eval_loader = TensorDataset(eval_data[0], eval_data[1])
    eval_loader = DataLoader(eval_loader, batch_size=hyparam['batch_size'], shuffle=False, pin_memory=True,
                             num_workers=4)
    return record_param, train_list, train_loader, eval_loader, train_size, train_data, eval_data

def train_prepare(hyparam, train_continue=False, model_dir=None):

    record_param, train_list, train_loader, eval_loader, train_size, train_data, eval_data=data_prepare(hyparam)
    model = __import__(hyparam['model']).Model().to(hyparam['device'])
    record_dir, result_text, writer = save_record(model, train_list, hyparam)

    if hyparam['learning_rate_scheduler']:
        optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
        learning_rate_scheduler = lr_scheduler(optimizer, hyparam['restart_num'], start=0.1, last=1e-4)
        # optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.00005)
        loss_func = FocalLoss()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        # loss_func=None
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.00005)
        loss_func = None
        # loss_func=torch.nn.CrossEntropyLoss()

    if train_continue:

        data = torch.load(hyparam['model_dir'] + 'current_model.pth')
        model.load_state_dict(data['model_state_dict'])

        optimizer.load_state_dict(data['optimizer_state_dict'])

        start_epoch=data['epoch']
        record_param=data['param']
       # result_text=record_dir


    else:
        start_epoch=-1


    error=False
    for epoch in range(start_epoch+1, hyparam['epoch']):

        print("\n", epoch, "epoch")
        if hyparam['learning_rate_scheduler']:
            learning_rate_scheduler.step(epoch)
            print("learning rate :", str(optimizer.param_groups[0]['lr']), '\n')

        if epoch<10:
            start = time()

        ############ train
        # try:
        #     loss, accuracy, each_accu = train(train_loader, model, optimizer, 'train', hyparam, loss_func)
        # except:
        #     error=True
        #     break
        # print(math.ceil(train_size / hyparam['batch_size']))
        # print(train_size)
        # exit()

        loss, accuracy, each_accu = train(train_loader, model, optimizer, 'train', hyparam,total_batch=math.ceil(train_size/hyparam['batch_size']), loss_func=loss_func)

        record_param['loss_list_train'].append(loss/train_size)
        record_param['accu_list_train'].append(accuracy/train_size)
        print_each_acc(result_text, epoch, each_accu, train_data[2])

        ############ evaluation
        # try:
        #     loss, accuracy, each_accu = train(eval_loader, model, optimizer, 'eval', hyparam, loss_func)
        # except:
        #     error=True
        #     break
        loss, accuracy, each_accu = train(eval_loader, model, optimizer, 'eval', hyparam,total_batch=math.ceil(eval_data[1].shape[0]/hyparam['batch_size']), loss_func=loss_func)

        record_param['loss_list_eval'].append(loss/len(eval_data[1]))
        record_param['accu_list_eval'].append(accuracy/len(eval_data[1]))
        print_each_acc(result_text, epoch, each_accu, eval_data[2])

        # if epoch <10:
        #     print("Training Time: %3.3f sec" % (time() - start))
        log_saving(record_dir, record_param, epoch, model, writer, optimizer, result_text, hyparam['restart_num'])

    if error:
        train_prepare(hyparam, train_continue, record_dir)
    else:
        log_saving(record_dir, record_param, epoch, model, writer, optimizer, result_text, hyparam['restart_num'], end=True)
        writer.close()

def temp_train_prepare(hyparam):
    train_data, eval_data = csv_reader(hyparam['temp_metadata'])
    train_data[0] = feature_load(train_data[0], hyparam['feature_dir'])
    eval_data[0] =torch.from_numpy(feature_load(eval_data[0], hyparam['feature_dir']))


    train_loader = Loading_in_Batch(train_data, hyparam['feature_dir'], specaug=hyparam['specaugment'],
                                    load_in_batch=False)
    train_loader = DataLoader(train_loader, batch_size=hyparam['batch_size'], shuffle=True)

    eval_loader = TensorDataset(eval_data[0], eval_data[1])
    eval_loader = DataLoader(eval_loader, batch_size=hyparam['batch_size'], shuffle=False)

    model = __import__(hyparam['model']).Model().to(hyparam['device'])

    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.00005, momentum=0.9)
    for epoch in range(hyparam['epoch']):
        print("\n", epoch, "번째 ")

        loss, accuracy, each_accu = train(train_loader, model, optimizer, 'train', hyparam)
        print(loss / train_data[1].shape[0], accuracy / train_data[1].shape[0])
        print(each_accu)
        print('\n')

        loss, accuracy, each_accu = train(eval_loader, model, optimizer, 'eval', hyparam)
        print(loss / eval_data[1].shape[0], accuracy / eval_data[1].shape[0])
        print(each_accu)
        print('\n')

def continue_training(model_dir,hyparam):

    data=torch.load(model_dir+'current_model.pth')
    model=__import__(hyparam['model']).Model().to(hyparam['device'])
    model.load_state_dict(data['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.00005)
    optimizer.load_state_dict(data['optimizer_state_dict'])

    return model, optimizer, data['epoch'], data['param'], model_dir, model_dir+'result.txt', SummaryWriter(model_dir + "/board")



class batch_size_tester():
    def __init__(self, hyparam, max_size=256):
        self.hyparam=hyparam
        self.start_num=0
        self.train_data, _= csv_reader(hyparam['train_metadata'])
        self.train_data[0] = feature_load(self.train_data[0][0:max_size], hyparam['feature_dir'])
        if hyparam['specaugment']:
            self.whole_size=2*self.train_data[1].shape[0]
        else:
            self.whole_size =  self.train_data[1].shape[0]
        self.train_data[1]=self.train_data[1][0:max_size]
        self.max_size=max_size
        self.mode = 'train'

    def forward(self):
        maxi=0

        for k in range(2):
            torch.cuda.empty_cache()
            model = __import__(self.hyparam['model']).Model().to(self.hyparam['device'])
            optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.00005)

            if k==1:
                self.start_num = maxi+1


            while True:
                # print(maxi, self.start_num)
                if k==0:
                    batch_size=2**self.start_num
                else:
                    batch_size = self.start_num
                if batch_size>self.max_size:
                    break
                tratra=[self.train_data[0][0:batch_size], self.train_data[1][0:batch_size]]
                train_loader = Loading_in_Batch(tratra, self.hyparam['feature_dir'], specaug=self.hyparam['specaugment'],
                                            load_in_batch=False)
                # print(self.start_num)
                # print(batch_size)
                loss_func = FocalLoss()
                train_loader = DataLoader(train_loader, batch_size=batch_size, drop_last=self.hyparam['droplast'], shuffle=True)
                # train(train_loader, model, optimizer, self.mode, self.hyparam, loss_func=loss_func)
                try:
                    train(train_loader, model, optimizer, self.mode, self.hyparam, loss_func=loss_func)

                except:
                    print('This is it\n')

                    break

                maxi=batch_size
                print('batchsize :', str(maxi), str(self.whole_size % maxi))
                self.start_num+=1





