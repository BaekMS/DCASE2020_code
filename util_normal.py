import yaml
import pandas as pd
import random
import numpy as np
import torch
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
import shutil
import matplotlib.pyplot as plt

def plot_numpy(data):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_title('yeah')
    plt.imshow(data)
    ax.set_aspect('equal')
    plt.colorbar(orientation='vertical')
    plt.show()

def randomseed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available() == True:
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        device = 'cuda'

    else:
        device = 'cpu'
    # print(torch.randint(low=0,high=1000, size=(5,5)))
    # print(torch.randint(low=0,high=1000, size=(5,5)))
    # exit()
    return device

def open_yaml(yaml_file):
    jj=open(yaml_file)
    vt=yaml.load(jj, Loader=yaml.FullLoader)
    return vt

def train_result(total_result, total_ans):
    accuracy = total_result.float().sum().item()
    total_result = ((total_result == True).nonzero(as_tuple=True))[0]
    total_result = total_ans[total_result].type(torch.float32)
    total_result = (torch.histc(total_result, bins=10, min=0, max=9).type(torch.long)).tolist()
    return accuracy, total_result

def csv_reader(metadata):
    train_data=[]
    eval_data=[]

    df = pd.read_csv(metadata[0])
    train_list = df['train_set'].values.tolist()
    train_ans =df['answer'].values.tolist()


    train_each_scene_num=[]
    for i in range(10):
        train_each_scene_num.append(train_ans.count(i))
    train_ans=torch.tensor(train_ans)

    train_data.append(train_list)
    train_data.append(train_ans)
    train_data.append(train_each_scene_num)

    df = pd.read_csv(metadata[1])
    eval_list = df['eval_set'].values.tolist()
    eval_ans = df['answer'].values.tolist()

    eval_each_scene_num = []
    for i in range(10):
        eval_each_scene_num.append(eval_ans.count(i))
    eval_ans=torch.tensor(eval_ans)

    eval_data.append(eval_list)
    eval_data.append(eval_ans)
    eval_data.append(eval_each_scene_num)

    return train_data, eval_data

def feature_load(data_list, feature_dir):
    # sample=np.load(feature_dir+data_list[0])[ :, :, 4:-4]
    sample=np.load(feature_dir+data_list[0])
    train_value=np.zeros((data_list.__len__(), *sample.shape), dtype=np.float32)
    # train_value=[]
    for num, data in enumerate(data_list):

        # train_value[num]=np.load(feature_dir + data)[ :, :, 4:-4]
        train_value[num]=np.load(feature_dir + data)
        # train_value.append(np.load(feature_dir + data))

    # train_value=torch.from_numpy(train_value)
    #print(train_value.shape)
    return train_value

def normalize(data, hyparam):
    # print(hyparam['mean'])
    # exit()
    return (data - hyparam['mean']) / hyparam['std']

def print_each_acc(result_text, epoch, each_accu, each_num):
    write_file = open(result_text, 'a')
    write_file.write('epoch ' + str(epoch) + '\n')

    for i in range(10):
        tp = each_accu[i] / each_num[i] * 100
        memory = "%d : %3.2f  " % (i, tp)
        print(memory, end='')
        write_file.write(memory)

    print('')
    write_file.write('\n')
    write_file.close()
def calc_mean_std(data_list, hyparam):
    # print(data_list)
    temp_sum=torch.zeros((3, 256, 423), dtype=torch.float32)
    for data in data_list:
        data_v = torch.from_numpy(np.load(hyparam['feature_dir'] + data))
        temp_sum+=data_v
        # hyparam[]
        # print(data_v.shape)
        # exit()
    hyparam['mean']=temp_sum.mean(dim=(1,2), keepdim=True).unsqueeze(0)/len(data_list)
    hyparam['std'] = temp_sum.std(dim=(1, 2), keepdim=True).unsqueeze(0)/len(data_list)
    # print(hyparam['mean'].shape)
    # exit()

def log_saving(record_dir, record_param, epoch, model, writer, optimizer, result_text,restart_num, end=False, temp=False):
    write_file=open(result_text, 'a')
    # print(epoch)
    print("\nAccuracy(train, eval) : %3.3f %3.3f" % (
        record_param['accu_list_train'][epoch] * 100, record_param['accu_list_eval'][epoch] * 100))
    write_file.write("\nAccuracy(train, eval) : %3.3f %3.3f \n" % (
        record_param['accu_list_train'][epoch] * 100, record_param['accu_list_eval'][epoch] * 100))

    print("Max accuracy(eval) : %d epoch, %3.3f\n" % (
        record_param['accu_list_eval'].index(max(record_param['accu_list_eval'])) ,
        max(record_param['accu_list_eval']) * 100))
    write_file.write("Max accuracy(eval) : %d epoch, %3.3f\n\n" % (
        record_param['accu_list_eval'].index(max(record_param['accu_list_eval'])),
        max(record_param['accu_list_eval']) * 100))

    print("Loss(train, eval) : %3.3f %3.3f" % (
        record_param['loss_list_train'][epoch], record_param['loss_list_eval'][epoch]))
    write_file.write("Loss(train, eval) : %3.3f %3.3f\n" % (
        record_param['loss_list_train'][epoch], record_param['loss_list_eval'][epoch]))

    print("Min loss(eval): %d epoch, %3.3f\n" % (
        record_param['loss_list_eval'].index(min(record_param['loss_list_eval'])) , min(record_param['loss_list_eval'])))
    write_file.write("Min loss(eval): %d epoch, %3.3f\n\n" % (
        record_param['loss_list_eval'].index(min(record_param['loss_list_eval'])) , min(record_param['loss_list_eval'])))

    write_file.close()

    if temp ==False:
        writer.add_scalars('Accuracy', {'Accuracy/Train': record_param['accu_list_train'][-1], 'Accuracy/Val':record_param['accu_list_eval'][-1]}, epoch)
        writer.add_scalars('Loss', {'Loss/Train': record_param['loss_list_train'][-1], 'Loss/Val': record_param['loss_list_eval'][-1]}, epoch)

        if (epoch % 10 == 0) or end ==True or epoch<10 or epoch!=0:
            for name, parameter in model.named_parameters():
                writer.add_histogram(name, parameter.clone().detach().cpu().data.numpy(), epoch)

        for x in restart_num:
            if epoch>=(x-6) and epoch<=(x-1):
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'param': record_param
                            }, record_dir + "/model_"+str(epoch)+".pth")

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'param': record_param
                    }, record_dir + "/current_model.pth")
        if record_param['loss_list_eval'][-1]==min(record_param['loss_list_eval']):
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'param': record_param
                        }, record_dir + "/best_loss_model.pth")

        if record_param['accu_list_eval'][-1]==max(record_param['accu_list_eval']):
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'param': record_param
                        }, record_dir + "/best_accu_model.pth")
        # learn_rate=[0,4,15,32,60,160,300,570]

        # learn_rate=[0, 50, 100, 150, 200, 300, 570]
        # for x in range(len(learn_rate)):
        #     if (epoch>=learn_rate[x]) and (epoch<learn_rate[x+1]):
        #
        #         lossloss=record_param['loss_list_eval'][learn_rate[x]:]
        #         accucu=record_param['accu_list_eval'][learn_rate[x]:]
        #
        #         if min(lossloss)==lossloss[-1]:
        #             torch.save({'epoch': epoch,
        #                         'model_state_dict': model.state_dict(),
        #                         'optimizer_state_dict': optimizer.state_dict(),
        #                         'param': record_param
        #                         }, record_dir + "/loss_good_model"+str(learn_rate[x])+".pth")
        #         if max(accucu)==accucu[-1]:
        #             torch.save({'epoch': epoch,
        #                         'model_state_dict': model.state_dict(),
        #                         'optimizer_state_dict': optimizer.state_dict(),
        #                         'param': record_param
        #                         }, record_dir + "/acc_good_model" + str(learn_rate[x]) + ".pth")

        fig1 = plt.figure(figsize=(7,4))
        epo = np.arange(0, epoch + 1, 1)
        a1 = fig1.add_subplot(2, 1, 1)
        a1.plot(epo, np.array(record_param['accu_list_train']), epo, np.array(record_param['accu_list_eval']))
        a1.set_title("Accuracy")
        a1.legend(['Train', 'Eval'])
        a1.set_xlabel('Epochs')
        a1.set_ylabel('Accuracy')
        a1.grid(axis='y', linestyle='dashed')

        a2 = fig1.add_subplot(2, 1, 2)
        a2.plot(epo, np.array(record_param['loss_list_train']), epo, np.array(record_param['loss_list_eval']))
        a2.set_title("Loss")
        a2.legend(['Train', 'Eval'])
        a2.set_ylabel('Loss')
        a2.set_xlabel('Epochs')
        a2.grid(axis='y', linestyle='dashed')
        fig1.tight_layout()
        fig1.savefig(record_dir + '/accu_loss.png', dpi=300)
        plt.close(fig1)
        fig1 = plt.figure(figsize=(7,4))
        epo = np.arange(0, epoch + 1, 1)
        a1 = fig1.add_subplot(2, 1, 1)
        a1.plot(epo, np.array(record_param['accu_list_train']), epo, np.array(record_param['accu_list_eval']))
        a1.set_title("Accuracy")
        a1.legend(['Train', 'Eval'])
        a1.set_xlabel('Epochs')
        a1.set_ylabel('Accuracy')
        a1.grid(axis='y', linestyle='dashed')

        a2 = fig1.add_subplot(2, 1, 2)
        a2.plot(epo, np.array(record_param['loss_list_train']), epo, np.array(record_param['loss_list_eval']))
        a2.set_title("Loss")
        a2.legend(['Train', 'Eval'])
        a2.set_ylabel('Loss')
        a2.set_xlabel('Epochs')
        a2.grid(axis='y', linestyle='dashed')
        fig1.tight_layout()
        fig1.savefig(record_dir + '/accu_loss.png', dpi=300)
        plt.close(fig1)

        # fig2=plt.figure()
        # epo = np.arange(0, epoch + 1, 1)
        # a1 = fig2.add_subplot(1, 1, 1)
        # a1.set_title("Accuracy")
        # a1.set_xlabel('Epochs')
        # a1.set_ylabel('Accuracy')
        # a1.grid(axis='y', linestyle='dashed')
        # fig2.tight_layout()
        #
        # for i in range(10):
        #     a1.plot(epo, np.array(each_accu[i]))
        # a1.legend([str(i) for i in range(10)])
        # fig2.savefig(model_dir + '/each_accu_loss.png', dpi=300)
        # plt.close(fig2)

def model_save(model, writer, train_list, hyparam):
    return
    # train_data=feature_load(train_list, hyparam['feature_dir']).to(hyparam['device'])
    # writer.add_graph(model, train_data, tar)

def new_folder(record_dir):
    now = datetime.now()
    model_dir = record_dir + now.strftime("%Y_%m_%d_%H_%M_%S")
    os.mkdir(model_dir)
    os.mkdir(model_dir+'/board')
    os.mkdir(model_dir+'/pyfile')
    result_text=open(model_dir+'/result.txt', 'w')
    result_text.close()
    result_text=model_dir+'/result.txt'

    memo = open(model_dir + '/description.txt', 'w')
    memo.close()

    return model_dir, result_text

def save_record(model, train_list, hyparam):
    record_dir, result_text = new_folder(hyparam['record_dir'])
    writer = SummaryWriter(record_dir + "/board")
    shutil.copy('./hyparam.yaml', record_dir)

    for pyfile in os.listdir('./'):

        if '.py' in pyfile:

            shutil.copy('./'+pyfile, record_dir+'/pyfile')

    model_save(model, writer, train_list[0:3], hyparam)
    fig1 = plt.figure(figsize=(7, 4))
    fig1.savefig(record_dir + '/accu_loss.png', dpi=300)
    plt.close(fig1)
    return record_dir, result_text, writer