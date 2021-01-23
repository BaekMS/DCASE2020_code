import Trident_Resnet

from util_train import *
from util_normal import *
import feature_extraction

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from sklearn.ensemble import VotingClassifier

# import json

def main():
    device = randomseed()



    global hyparam

    hyparam=open_yaml('./hyparam.yaml')
    hyparam['device']=device


    train_data, eval_data = csv_reader(hyparam['train_metadata'])
    calc_mean_std(train_data[0], hyparam)

    eval_data[0] = feature_load(eval_data[0], hyparam['feature_dir'])
    eval_data[0] = torch.tensor(eval_data[0])



    # eval_data[0]=(eval_data[0]-eval_data[0].mean(dim=(-1,-2), keepdim=True))/eval_data[0].std(dim=(-1,-2), keepdim=True)
    # print(eval_data[0].shape)
    eval_loader = TensorDataset(eval_data[0], eval_data[1])
    eval_loader = DataLoader(eval_loader, batch_size=hyparam['batch_size'], shuffle=False, pin_memory=True,
                             num_workers=4)

    model_dir='./record/2020_10_18_13_05_21/'
    number=[62,125,253]
    weight = [70.317, 70.586, 70.350]
    weight_list=[[1.3, 1, 1.3], [70.216, 70.350, 70.115], [1.2, 0.9, 1.2], [1.25, 0.9, 1.2], [1.1,0.9, 1.1]]
    # weight=[1,1,1]
    # number=[62]
    result=[]
    model = __import__(hyparam['model']).Model().to(hyparam['device']).eval()
    for num in number:
        print(num)


        total=None
        anan=None



        if num==None:
            data = torch.load(model_dir+'best_accu_model.pth')
        else:
            data = torch.load(model_dir + 'model_' + str(num) + '.pth')

        model.load_state_dict(data['model_state_dict'])
        # print(model.resmodule[0].cv21.weight.shape)
        # # print(model)
        # exit()

        for train_data, ans in eval_loader:
            train_data = normalize(train_data, hyparam).to(hyparam['device'])

            with torch.no_grad():
                output=model(train_data).to('cpu').exp()

            if total==None:
                total=output
                anan=ans
            else:
                total=torch.cat((total, output), dim=0)
                anan=torch.cat((anan, ans), dim=0)


        if torch.all(anan.eq(eval_data[1])):
            print('good')
        else:
            print('jenjang')

        result.append(total)
        # model=model.to('cpu')

    # weight=[0.93, 1.07, 1]
    for weight in weight_list:
        tx = (result[0] * weight[0] + result[1] * weight[1] + result[2] * weight[2]) / (weight[0] + weight[1] + weight[2])
        # tx=(result[0]*68.430+result[1]*70.249+result[2]*69.542)/(70.249+69.542+68.430)

        # for kj in range(3):
        #     tx = result[kj]
        #     # tx=result[0]

            # exit()

            # tx=result[0]
        total_result = tx.max(dim=1)[1].eq(eval_data[1])

        total_result=total_result.sum().item()
        print('\n\n')
        # print('weighted')
        print(weight)
        print(total_result)
        print(total_result/eval_data[1].shape[0]*100)
        print('\n\n')

    for jj in range(3):
        result[jj]=result[jj].max(dim=-1)[1].unsqueeze(1)

    tf=torch.cat((result[0], result[1], result[2]), dim=-1)
    tf=torch.mode(tf, dim=-1)[0]

    total_result = tf.eq(eval_data[1])
    total_result = total_result.sum().item()
    print('\n\n\n\n')
    print(total_result)
    print(total_result / eval_data[1].shape[0] * 100)
    print('\n\n\n\n')
    # print(tf)
    # print(tf.shape)

if __name__=="__main__":

    main()