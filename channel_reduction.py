import Trident_Resnet

from util_train import *
from util_normal import *
import feature_extraction
from scipy import sparse
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from sklearn.ensemble import VotingClassifier
import numpy as np
# import json
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import pairwise
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
exit()

def plotting(data, inin, outout, maxi, mini):
    save_loc = './testing/'
    fig=plt.figure()
    ax=fig.add_subplot(111)
    # ax.set_title('yeah')
    plt.imshow(data, cmap ='Greys')
    ax.set_aspect('equal')
    plt.clim(mini, maxi)

    plt.colorbar(orientation='vertical')
    # plt.pcolor(X, Y, f(data), cmap=cm, vmin=-4, vmax=4)

    # plt.colorbar()
    # plt.show()
    fig.savefig(save_loc + str(inin)+'_'+str(outout)+'.png', dpi=300)
    plt.close()

def main():
    device = randomseed()
    save_loc='./testing/'


    global hyparam

    hyparam=open_yaml('./hyparam.yaml')
    hyparam['device']=device


    # train_data, eval_data = csv_reader(hyparam['train_metadata'])
    # calc_mean_std(train_data[0], hyparam)
    #
    # eval_data[0] = feature_load(eval_data[0], hyparam['feature_dir'])
    # eval_data[0] = torch.tensor(eval_data[0])
    #
    #
    #
    # # eval_data[0]=(eval_data[0]-eval_data[0].mean(dim=(-1,-2), keepdim=True))/eval_data[0].std(dim=(-1,-2), keepdim=True)
    # # print(eval_data[0].shape)
    # eval_loader = TensorDataset(eval_data[0], eval_data[1])
    # eval_loader = DataLoader(eval_loader, batch_size=hyparam['batch_size'], shuffle=False, pin_memory=True,
    #                          num_workers=4)

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
        # print(model.weight)
        for name, p in model.named_parameters():
            # print(p, name)
            if 'cv' in name and 'bias' not in name:
                # print(name)
                kp = p.clone().to('cpu').detach()
                shaping=kp.shape
                kp = kp.reshape(kp.shape[0], kp.shape[1], -1).numpy()
                last=None
                for ii in range(kp.shape[1]):
                    res= pairwise.cosine_similarity(kp[:, ii,:])
                    try:
                        last += res
                    except:
                        last=res

                last/=ii+1
                last[abs(last)<0.8]=0
                # b=abs(last)>0.9
                # print(b)
                # exit()
                fig = plt.figure()
                ax = fig.add_subplot(111)
                # ax.set_title('yeah')
                plt.imshow(last, cmap='bwr')
                ax.set_aspect('equal')
                plt.clim(-1, 1)

                plt.colorbar(orientation='vertical')
                # plt.pcolor(X, Y, f(data), cmap=cm, vmin=-4, vmax=4)

                # plt.colorbar()
                # plt.show()
                now = datetime.now()

                fig.savefig(save_loc + name + '.png')
                plt.close()
        exit()
                # print(kp.shape)
                # exit()
            # exit()
        print(model.parameters)
        exit()
        channel_data=model.resmodule[0].cv11.weight.to('cpu').detach().numpy()
        kp=model.resmodule[0].cv11.weight.to('cpu').detach()

        a_sparse = pairwise.cosine_similarity(kp[:, :])
        print(a_sparse)
        print(kp.shape)
        exit()
        cos=torch.nn.CosineSimilarity(dim = 3)
        print(cos(kp, kp).shape)
        exit()
        channel_data=np.reshape(channel_data, (32, 27))
        print(channel_data.shape)
        a_sparse=pairwise.cosine_similarity(channel_data[:,:])
        # print(res)
        # exit()
        # a_sparse=sparse.csr_matrix(channel_data[:,0,:], dtype=np.float32).toarray()
        # print(a_sparse)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax.set_title('yeah')
        plt.imshow(a_sparse, cmap='viridis')
        ax.set_aspect('equal')
        plt.clim(-1,1)

        plt.colorbar(orientation='vertical')
        # plt.pcolor(X, Y, f(data), cmap=cm, vmin=-4, vmax=4)

        # plt.colorbar()
        plt.show()
        now = datetime.now()

        fig.savefig(save_loc + now.strftime("%Y_%m_%d_%H_%M_%S")+'.png')
        plt.close()
        print(a_sparse.shape)
        exit()
        # exit()
        # print(channel_data.max())
        # print(channel_data.min())

        # exit()
        for inin in range(channel_data.shape[1]):
            for outout in range(channel_data.shape[0]):
                plotting(channel_data[outout, inin,:,:], inin, outout, channel_data.max(),channel_data.min())
        exit()

        print(channel_data)


        # print(model.resmodule[0].cv21.weight)
        # torch.save({'channel':model.resmodule[0].cv21.weight.to('cpu')
        #             }, save_loc + "channel.pth")
        # print(model)
        exit()

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