import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from time import time, sleep
import matplotlib.pyplot as plt
from datetime import datetime
import random
from sklearn.preprocessing import minmax_scale
from time import time
import pandas as pd
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import shutil

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available()==True:
    torch.cuda.manual_seed_all(0)
    device='cuda'
else:
    device='cpu'


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

        self.size_average = size_average

    def forward(self, input, target, mix_up=False):
        if mix_up==False:
            if input.dim()>2:
                input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
                input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
                input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
            target = target.view(-1,1)


            logpt = F.log_softmax(input, dim=-1)
            # target=target.view
            # print(logpt.shape)
            # print(target.shape)
            # exit()
            # print(logpt.shape)
            # print(target.shape)
            # exit()
            logpt = logpt.gather(1,target)

            logpt = logpt.view(-1)

            pt = logpt.data.exp()



            loss = -1 *self.alpha* (1-pt)**self.gamma * logpt

            if self.size_average: return loss.mean()
            else: return loss.sum()

        else:

            if input.dim() > 2:
                input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
                input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
                input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
            # target = target.view(-1,1)

            # print(target)

            logpt = F.log_softmax(input, dim=-1)
            # print(logpt.shape)
            # exit()

            kk=target.nonzero(as_tuple=True)

            # logpt = logpt.gather(1,target)



            loss=-1*target[kk[0], kk[1]]*self.alpha*(1-logpt[kk[0], kk[1]].data.exp())**self.gamma*logpt[kk[0], kk[1]]


            if self.size_average:
                return loss.sum()/target.shape[0]
            else:
                return loss.sum()

class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()


        self.avp1 = nn.AvgPool2d((1, 4), stride=(1, 4), padding=(0, 0))
        self.bn1=nn.BatchNorm2d(1)
        self.cv1=nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=(1,2), padding=(2,0), dilation=(2,1))
        self.bn2=nn.BatchNorm2d(32)
        self.cv2=nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=(1,2), padding=(2,1), dilation=(2,1))


        self.avp2 = nn.AvgPool2d((1, 4), stride=(1, 4), padding=(0, 0))

        self.bn3 = nn.BatchNorm2d(64)
        self.cv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=(1, 2), padding=(2, 0),
                             dilation=(2, 1))
        self.bn4=nn.BatchNorm2d(128)
        self.cv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=(1, 2), padding=(2, 1),
                             dilation=(2, 1))
        # cnn bias = False
        # nn.init.kaiming_uniform_(self.cv1.weight, nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.cv2.weight, nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.cv3.weight, nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.cv4.weight, nonlinearity='relu')
        #
        # for p in self.bn2.parameters():
        #     p.requires_grad=False
        # for p in self.bn4.parameters():
        #     p.requires_grad=False

    def forward(self, x):
        # print(x.shape)
        avg1=self.avp1(x).to(device)
        # print(avg1.shape)
        # exit()

        pad1=(0,0, 0,0, 31,32, 0,0)
        avg1=F.pad(avg1, pad1, 'constant', 0)


        x=self.bn1(x)
        x=F.relu(x)
        x=self.cv1(x)


        # x = (x - x.mean(dim=(2,3),keepdim=True))/x.std(dim=(2,3),keepdim=True)
        x=self.bn2(x)
        x=F.relu(x)
        x=self.cv2(x)
        # print(avg1)
        # print(x)
        # exit()


        x+=avg1


        avg2=self.avp2(x).to(device)

        pad2 = (0, 0, 0, 0, 96, 96, 0, 0)
        avg2 = F.pad(avg2, pad2, 'constant', 0)


        x = self.bn3(x)
        x = F.relu(x)
        x = self.cv3(x)
        x=self.bn4(x)
        # x = (x - x.mean(dim=(2,3),keepdim=True))/x.std(dim=(2,3),keepdim=True)

        x = F.relu(x)
        x = self.cv4(x)


        x+=avg2

        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.cv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=(1, 1), padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.mp1 = nn.MaxPool2d((4, 10), stride=(1, 2))

        self.dp1 = nn.Dropout(0.3)

        self.cv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.mp2 = nn.MaxPool2d((6, 6), stride=(3, 3))
        self.dp2 = nn.Dropout(0.3)

        self.gru=nn.GRU(input_size=1984, hidden_size=128, num_layers=2, dropout=0.3, batch_first=True)

        self.fc1=nn.Linear(128, 64)
        self.fc2=nn.Linear(64, 29)
        self.dp3=nn.Dropout(0.3)
        # self.dp4=nn.Dropout(0.3)

    def forward(self, x):
        x = self.cv1(x)
        x = self.bn1(x)
        x = self.mp1(x)
        x = F.relu(x)
        x = self.dp1(x)

        x = self.cv2(x)
        x = self.bn2(x)
        x = self.mp2(x)
        x = F.relu(x)
        x = self.dp2(x)
        x=x.reshape(-1, x.shape[1]*x.shape[2], x.shape[-1]) # batch, channel*freq, time
        x=x.permute(0,2,1)
        # print(x.shape)
        # exit()
        x=self.gru(x)
        # print(x[0].shape)
        # exit();
        x=x[0][:,-1,:]

        x=self.fc1(x)
        # print(x.shape)
        # exit()

        x = F.relu(x)
        x = self.dp3(x)
        x=self.fc2(x)
        # print(x.shape)
        #
        #
        #
        # x = self.bn1(x)
        # x = self.cv1(x)
        #
        # x = self.bn2(x)
        # x = self.cv2(x)
        # x = self.bn3(x)
        #
        # x=self.avp(x)
        # x=x.view(-1, 29)


        x = F.log_softmax(x, dim=-1)
        return x

def new_folder(model_dir):
    now = datetime.now()
    model_dir = model_dir + "/" + now.strftime("%Y_%m_%d_%H_%M_%S")
    os.mkdir(model_dir)
    os.mkdir(model_dir+'/board')

    result_text = open(model_dir + '/result.txt', 'w')
    result_text.close()
    result_text = model_dir + '/result.txt'

    memo = open(model_dir + '/memo.txt', 'w')
    memo.close()
    return model_dir, result_text

def log_saving(model_dir, param, epoch, model, writer,optimizer,result_text, end=False):
    write_file = open(result_text, 'a')

    print("Accuracy(train, eval, KOR) : %3.3f %3.3f %3.3f" % (
        param['accu_list_train'][epoch] * 100, param['accu_list_eval'][epoch] * 100, param['accu_list_KOR'][epoch] * 100))
    write_file.write("\nAccuracy(train, eval) : %3.3f %3.3f \n" % (
        param['accu_list_train'][epoch] * 100, param['accu_list_eval'][epoch] * 100))
    print("Max accuracy(eval) : %d 번째, %3.3f" % (
        param['accu_list_eval'].index(max(param['accu_list_eval'])),
        max(param['accu_list_eval']) * 100))
    write_file.write("Max accuracy(eval) : %d 번째, %3.3f\n\n" % (
        param['accu_list_eval'].index(max(param['accu_list_eval'])),
        max(param['accu_list_eval']) * 100))
    print("Max accuracy(KOR) : %d 번째, %3.3f\n" % (
        param['accu_list_KOR'].index(max(param['accu_list_KOR'])) ,
        max(param['accu_list_KOR']) * 100))
    write_file.write("Max accuracy(KOR) : %d 번째, %3.3f\n\n" % (
        param['accu_list_KOR'].index(max(param['accu_list_KOR'])),
        max(param['accu_list_KOR']) * 100))

    print("Loss(train, eval, KOR) : %3.3f %3.3f %3.3f" % (
        param['loss_list_train'][epoch], param['loss_list_eval'][epoch], param['loss_list_KOR'][epoch]))
    write_file.write("Loss(train, eval) : %3.3f %3.3f\n" % (
        param['loss_list_train'][epoch], param['loss_list_eval'][epoch]))
    print("Min loss(eval): %d 번째, %3.3f" % (
        param['loss_list_eval'].index(min(param['loss_list_eval'])) , min(param['loss_list_eval'])))
    write_file.write("Min loss(eval): %d 번째, %3.3f\n\n" % (
        param['loss_list_eval'].index(min(param['loss_list_eval'])), min(param['loss_list_eval'])))
    print("Min loss(KOR): %d 번째, %3.3f\n" % (
        param['loss_list_KOR'].index(min(param['loss_list_KOR'])) , min(param['loss_list_KOR'])))
    write_file.write("Min loss(KOR: %d 번째, %3.3f\n\n" % (
        param['loss_list_KOR'].index(min(param['loss_list_KOR'])), min(param['loss_list_KOR'])))

    write_file.close()


    writer.add_scalars('Accuracy', {'Accuracy/Train': param['accu_list_train'][-1], 'Accuracy/Val':param['accu_list_eval'][-1], 'Accuracy/KOR':param['accu_list_KOR'][-1]}, epoch+1)
    writer.add_scalars('Loss', {'Loss/Train': param['loss_list_train'][-1], 'Loss/Val': param['loss_list_eval'][-1], 'Loss/KOR': param['loss_list_KOR'][-1]}, epoch+1)

    if param['loss_list_eval'][-1] == min(param['loss_list_eval']):
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'param': param
                    }, model_dir + "/best_loss_model.pth")

    if param['accu_list_eval'][-1] == max(param['accu_list_eval']):
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'param': param
                    }, model_dir + "/best_accu_model.pth")

    if (epoch % 10 == 1) or end ==True:

        for name, parameter in model.named_parameters():
            writer.add_histogram(name, parameter.clone().detach().cpu().data.numpy(), epoch+1)

    if (epoch % 10 == 1) or end ==True:
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'param': param
                    }, model_dir + "/current_model.pth")

    fig1 = plt.figure()
    epo = np.arange(0, epoch + 1, 1)
    a1 = fig1.add_subplot(2, 1, 1)
    a1.plot(epo, np.array(param['accu_list_train']), epo, np.array(param['accu_list_eval']), epo, np.array(param['accu_list_KOR']))
    a1.set_title("Accuracy")
    a1.legend(['Train', 'Eval', 'KOR'])
    a1.set_xlabel('Epochs')
    a1.set_ylabel('Accuracy')
    a1.grid(axis='y', linestyle='dashed')


    a2 = fig1.add_subplot(2, 1, 2)
    a2.plot(epo, np.array(param['loss_list_train']), epo, np.array(param['loss_list_eval']), epo, np.array(param['loss_list_KOR']))
    a2.set_title("Loss", y=1.08)
    a2.legend(['Train', 'Eval', 'KOR'])
    a2.set_ylabel('Loss')
    a2.set_xlabel('Epochs')
    a2.grid(axis='y', linestyle='dashed')
    fig1.tight_layout()
    fig1.savefig(model_dir + '/accu_loss.png', dpi=300)

    plt.close(fig1)

def model_save(model, writer, train_list, train_ans, feature_dir):
    train_data=[]
    for i in train_list:
        data=np.load(feature_dir+i)
        data = (data - np.mean(data)) / np.std(data)
        train_data.append([data])
    train_data=torch.tensor(train_data).to(device)
    writer.add_graph(model, train_data)
    # exit()

def data_load(train_x, feature_dir):
    train_set = []
    for i in train_x:
        if 'nohash' in i or 'KOR' in i:
            data = np.load(feature_dir + i)
            # data=(data-np.mean(data))/np.std(data)
            train_set.append([data])


    return np.array(train_set)

def mixup(num, ans, value, accum):
    alpha=random.uniform(0.55, 0.9)

    while True:
        if ans==0:
            mix=random.randint(0, accum[0]-1)
        else:
            mix=random.randint(accum[ans-1]-1, accum[ans]-1)

        if mix!=num:
            break

    result=alpha*value[num]+(1-alpha)*value[mix]
    return result

def run(model, loader, optimizer, mode, size):
    word_list = (
    'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin', 'nine',
    'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'two', 'up', 'wow', 'yes', 'zero')

    word_accu_list = {'bed':0, 'bird':0, 'cat':0, 'dog':0, 'down':0, 'eight':0, 'five':0, 'four':0, 'go':0, 'happy':0, 'house':0, 'left':0, 'marvin':0, 'nine':0,
    'no':0, 'off':0, 'on':0, 'one':0, 'right':0, 'seven':0, 'sheila':0, 'six':0, 'stop':0, 'three':0, 'two':0, 'up':0, 'wow':0, 'yes':0, 'zero':0}

    if mode == "train":
        model.train()
    elif mode == "eval":
        model.eval()

    total_loss=0
    accuracy=0

    for train_set, ans in loader:

        train_set=train_set.to(device)

        ans = ans.to(device)
        if mode == "train":
            output = model(train_set)
        elif mode=="eval":
            with torch.no_grad():
                output = model(train_set)

        batch_now = train_set.shape[0]
        loss = F.nll_loss(output, ans)

        ans = ans.cpu().data.numpy()

        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        result = torch.max(output.data, 1)[1].cpu().data.numpy()
        for num, ansans in enumerate(ans):
            if ansans==result[num]:
                word_accu_list[word_list[ansans]]+=1

        accuracy += sum(ans == result) / len(ans) * batch_now / size
        total_loss+=loss.detach().cpu().data.numpy()*batch_now/size


    return total_loss, accuracy, word_accu_list

def augrun(model, loader, optimizer, mode, size, feature_dir, train_list, loss_func, value, load_in_batch):
    word_list = (
    'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin', 'nine',
    'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'two', 'up', 'wow', 'yes', 'zero')

    word_accu_list = {'bed':0, 'bird':0, 'cat':0, 'dog':0, 'down':0, 'eight':0, 'five':0, 'four':0, 'go':0, 'happy':0, 'house':0, 'left':0, 'marvin':0, 'nine':0,
    'no':0, 'off':0, 'on':0, 'one':0, 'right':0, 'seven':0, 'sheila':0, 'six':0, 'stop':0, 'three':0, 'two':0, 'up':0, 'wow':0, 'yes':0, 'zero':0}

    if mode == "train":
        model.train()
    elif mode == "eval":
        model.eval()

    total_loss=0
    accuracy=0

    for train_x, ans in loader:

        if train_x.shape[0]%2==0:
            train_x=train_x.split(int(train_x.shape[0]/2))
            ans=ans.split(int(ans.shape[0]/2))
        else:
            train_x = train_x.split(int(train_x.shape[0] / 2)+1)
            ans = ans.split(int(ans.shape[0] / 2)+1)

        train_x=list(train_x)

        for i in range(2):
            train_set = []
            train_num = train_x[i].tolist()

            if load_in_batch == True:
                for kp in train_num:
                    # a = (train_list[i]).split('_')[1]
                    # if a == 'specaug':
                    #     continue
                    # elif a == 'mixup':
                    #     continue
                    # else:

                    data = np.load(feature_dir + train_list[kp])

                    data = (data - np.mean(data)) / np.std(data)
                    train_set.append([data])
            else:
                train_set = np.take(value, train_num, axis=0)

            train_set=torch.tensor(train_set)
            train_set=torch.cat((train_set, specaugment(train_set)), dim=0)
            train_set=Variable(train_set).to(device)
            # train_set=torch.tensor(train_set).to(device)
            ansans=torch.cat((ans[i], ans[i]), dim=0).to(device)
            # ansans = ans[i].to(device)
            if mode == "train":
                output = model(train_set)
            elif mode=="eval":
                with torch.no_grad():
                    output = model(train_set)

            batch_now = train_set.shape[0]
            # loss = F.nll_loss(output, ans)

            loss=loss_func(output, ansans)
            ansans = ansans.cpu().data.numpy()

            if mode == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            result = torch.max(output.data, 1)[1].cpu().data.numpy()
            for num, anan in enumerate(ansans):
                if anan==result[num]:
                    word_accu_list[word_list[anan]]+=1

            accuracy += sum(ansans == result) / (size*2)
            total_loss+=loss.detach().cpu().data.numpy()*batch_now/(size*2)


    return total_loss, accuracy, word_accu_list

def specaugment(ori):
    # data=deepcopy(ori)

    T=int(ori.shape[2]*0.8/3)
    WF=int(ori.shape[2]*0.2)
    F=int(ori.shape[1]*0.8/3)
    mid=ori.mean()

    W = random.randint(1, WF)
    move = random.randint(1, ori.shape[-1]-2*W-3)

    temp = ori[:, :, W:-W]

    ori[:,:, W:W+move] = temp[:,:, -move:]
    ori[:,:, W+ move:-W] = temp[:, :, 0:-move]

    k= random.randint(1,3)
    for x in range(k):
        t = random.randint(5, T)

        t0 = random.randint(0, ori.shape[2] - t - 1)
        ori[:,:,t0:t0+t]=0

    f = random.randint(5, F)
    f0 = random.randint(0, ori.shape[1] - f - 1)
    ori[:,f0 :f0+f, :] = 0

    return ori

class Loading_in_Batch():
    def __init__(self, train_data,train_ans, feature_dir, specaug=True, load_in_batch=True):

        self.train_data=train_data
        self.ans=train_ans
        self.feature_dir=feature_dir
        self.specaug=specaug
        self.load_in_batch=load_in_batch



    def making_data(self, index):

        if index>=len(self.ans):
            index=index-len(self.ans)
            spec=True
        else:
            spec=False

        if self.load_in_batch:
            data = np.load(self.feature_dir + self.train_data[index])

            data=data[np.newaxis]
            # print(data.shape)
        else:
            data=self.train_data[index,:,:,:]
        # print(data)
        if spec:
            # print(data.shape)
            data=specaugment(data)

        data=(data-data.mean())/data.std()
        # print(data.shape)
        # data=data.squeeze(0)

        # feature_extraction.showing(data, 44100, 1024)
        # feature_extraction.showing(specaugment(data), 44100, 1024)

        # data=np.concatenate((data, specaugment(data)), axis=0)
        ans = self.ans[index]


        return data, ans

    def __getitem__(self, index):



        return self.making_data(index)

    def __len__(self):
        if self.specaug:
            return 2*len(self.ans)
        else:
            return len(self.data)

def train(feature_dir,model_dir, batch_size=128, total_epoch=500, temp=False, load_in_batch=True, augment=False):
    word_list = (
        'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin', 'nine',
        'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'two', 'up', 'wow', 'yes', 'zero')

    print('Loading...')

    if temp:
        train = feature_dir + "metadata/train_set_divp_temp.csv"
    else:
        train= feature_dir + "metadata/train_set_divp.csv"


    df = pd.read_csv(train)
    train_list = df['train_set'].values.tolist() # 파일 이름
    train_ans = df['answer'].values.tolist()


    train_people_num = [0 for _ in range(29)] # 각 단어 사람수

    for i in train_ans:
        train_people_num[i]+=2



    # train_data=torch.from_numpy(train_data)

    if load_in_batch:
        train_data = train_list
    else:
        train_data = data_load(train_list, feature_dir)
    train_data = Loading_in_Batch(train_data,train_ans, feature_dir, load_in_batch=load_in_batch)
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # print(train_data)
    # exit()


    if temp:
        eval = feature_dir + "metadata/eval_set_divp_temp.csv"
    else:
        eval = feature_dir + "metadata/eval_set_divp.csv"


    df = pd.read_csv(eval)
    eval_list=df['eval_set'].values.tolist()

    eval_ans=df['answer'].values.tolist()

    eval_people_num = [0 for _ in range(29)]
    for i in eval_ans:
        eval_people_num[i]+=1

    eval_data = torch.from_numpy(data_load(eval_list, feature_dir))
    eval_data=(eval_data-eval_data.mean(dim=(-1,-2), keepdim=True))/eval_data.std(dim=(-1,-2), keepdim=True)
    eval_ans = torch.tensor(eval_ans)

    eval_data =TensorDataset(eval_data, eval_ans)
    eval_data = DataLoader(eval_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


    # KOR data

    if temp:
        KOR_data = feature_dir + "metadata/KOR_data_temp.csv"
    else:
        KOR_data = feature_dir + "metadata/KOR_data.csv"


    df=pd.read_csv(KOR_data)
    KOR_list=df['eval_set'].values.tolist()
    KOR_ans = df['answer'].values.tolist()

    KOR_people_num = [0 for _ in range(29)]
    for i in KOR_ans:
        KOR_people_num[i] += 1

    KOR_data = torch.from_numpy(data_load(KOR_list, feature_dir))
    KOR_data = (KOR_data - KOR_data.mean(dim=(-1, -2), keepdim=True)) / KOR_data.std(dim=(-1, -2), keepdim=True)
    # print(KOR_data.shape)
    KOR_ans = torch.tensor(KOR_ans)

    KOR_data = TensorDataset(KOR_data, KOR_ans)
    KOR_data = DataLoader(KOR_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)



    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.000001)
    # loss_func=FocalLoss(gamma=2, alpha=0.25)


    param = {'accu_list_train': [], 'accu_list_eval': [], 'accu_list_KOR': [], 'loss_list_train': [], 'loss_list_eval': [], 'loss_list_KOR': [] }


    if temp==False:
        model_dir, result_text = new_folder(model_dir)
        writer = SummaryWriter(model_dir + "/board")
        shutil.copy('./main-gru.py', model_dir)
        model_save(model, writer, train_list[0:3], train_ans[0:3], feature_dir)

    first=True



    for epoch in range(total_epoch):

        print("\n", epoch, "번째 ", '\n')
        if temp == False:
            write_file = open(result_text, 'a')
            write_file.write('epoch ' + str(epoch) + '\n')

        start = time()

        loss, accuracy, each_accu = run(model, train_data, optimizer, "train", 2*len(train_ans))

        print("train each accu")

        for i in range(29):
            tempt = each_accu[word_list[i]] / train_people_num[i] * 100

            print("%s : %3.2f  " % (word_list[i], tempt), end='')
            if (i != 0) and (i % 10 == 0):
                print('')

            write_file.write("%s : %3.2f  " % (word_list[i], tempt))
        write_file.write('\n')
        write_file.close()
        print('')

        param['loss_list_train'].append(loss)
        param['accu_list_train'].append(accuracy)

        loss, accuracy, each_accu = run(model, eval_data, optimizer, "eval", len(eval_ans))

        print("\neval each accu")
        write_file = open(result_text, 'a')
        for i in range(29):
            tempt = each_accu[word_list[i]] / train_people_num[i] * 100

            print("%s : %3.2f  " % (word_list[i], tempt), end='')
            if (i != 0) and (i % 10 == 0):
                print('')
            write_file.write("%s : %3.2f  " % (word_list[i], tempt))
        write_file.write('\n')
        write_file.close()
        print('')

        param['loss_list_eval'].append(loss)
        param['accu_list_eval'].append(accuracy)

        loss, accuracy, each_accu = run(model, KOR_data, optimizer, "eval", len(KOR_ans))

        print("\n\nKOR each accu")
        write_file = open(result_text, 'a')

        for i in range(29):
            tempt = each_accu[word_list[i]] / train_people_num[i] * 100

            print("%s : %3.2f  " % (word_list[i], tempt), end='')
            if (i != 0) and (i % 10 == 0):
                print('')
            write_file.write("%s : %3.2f  " % (word_list[i], tempt))
        write_file.write('\n')
        write_file.close()
        print('')

        param['loss_list_KOR'].append(loss)
        param['accu_list_KOR'].append(accuracy)


        print("\n\nTraining Time: %3.3f sec\n" % (time() - start))
        log_saving(model_dir, param, epoch, model, writer, optimizer, result_text)
    log_saving(model_dir, param, epoch, model, writer, optimizer, result_text, end=True)
    writer.close()

def main():

    feature_dir = "D:/Graduate/feature/speech_commands/"
    feature_dir="../feature/practice/"
    batch_size=350

    record_dir="./record"
    epoch=1000
    load_in_batch=False
    augment=True

    while True:
        print("숫자 입력(1: temp testing, 2: new training, 3: end)")
        a = input()
        a = int(a)

        if a==1:
            train(feature_dir, record_dir, batch_size=batch_size, total_epoch=epoch, temp=True, load_in_batch=load_in_batch, augment=augment)
        elif a==2:
            train(feature_dir, record_dir, batch_size=batch_size, total_epoch=epoch, temp=False, load_in_batch=load_in_batch, augment=augment)
        elif a==3:
            exit()


if __name__=="__main__":
    main()