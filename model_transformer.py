import torch
from torch import nn
import torch.nn.functional as F
import custom_transformer
import util_normal
from feature_extraction import showing
import Trident_Resnet
import small_trident


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.main_model1=nn.Transformer(d_model=256, nhead=16, dim_feedforward=2048, num_encoder_layers=4, num_decoder_layers=4)
        self.main_model2=nn.Transformer(d_model=256, nhead=16, dim_feedforward=2048, num_encoder_layers=4, num_decoder_layers=4)
        self.main_model3 = nn.Transformer(d_model=256, nhead=16, dim_feedforward=2048, num_encoder_layers=4,
                                          num_decoder_layers=4)
        # self.ttt=torch.zeros((256,1), dtype=torch.float32)
        # self.main_model=custom_transformer.Transformer(d_model=256, nhead=8, dim_feedforward=2048, num_encoder_layers=4, num_decoder_layers=4)
        # self.big_model=Trident_Resnet.Model()
        # self.big_model=small_trident.Model()
        # self.bnx=nn.BatchNorm2d(1)

        # self.mp1=nn.MaxPool2d(kernel_size=(1,29), stride=(1,23))
        # self.fc=nn.Linear(256, 10)
        # self.fc1=nn.Linear(110336, 10)
        self.mp1=nn.MaxPool2d(kernel_size=(3,3), stride=(2,1))
        self.cv1=nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3,1), stride=(2,1))
        self.avg1=nn.AvgPool2d(kernel_size=(63,8))
        # self.bn1=nn.BatchNorm2d(64)
        # self.mp1 = nn.MaxPool2d(kernel_size=5, stride=2)
        #
        # self.cv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        # self.mp2 = nn.MaxPool2d(kernel_size=5, stride=2)
        #
        # self.cv3 = nn.Conv2d(in_channels=64, out_channels=10, kernel_size=3, stride=1)
        # self.mp3 = nn.MaxPool2d(kernel_size=5, stride=2)
        #
        # self.avp=nn.AvgPool2d(27,49)
        # # self.fc1=nn.Linear(256,10)
        # self.av1=nn.AvgPool2d(83,141)
        # self.tp=torch.zeros((71,256,256,26)).to(device)
        #
        self._reset_parameters()



    def forward(self, x, tar):
        # a = x[1, 0, :, :].detach().cpu().numpy()
        # util_normal.plot_numpy(a)

        torch.cuda.empty_cache()


        # x=x.squeeze(1)
        x=x.squeeze(1).permute(2,0,1)
        # x=self.main_model(x, x).permute(1,2,0).unsqueeze(1)
        # # print(x.shape)
        # x=self.bnx(x)
        # x=self.big_model(x)
        # print(x.shape)
        # exit()
        ttt=torch.zeros((10,x.shape[1],256), dtype=torch.float32).to('cuda')
        # ttt[:,:,0:10]=0.1
        # # print(ttt.shape)
        # # # print(x.shape)
        # # # exit()
        # # # # tar=tar.unsqueeze(0).permute(0,1,2)
        # # # # print(tar.shape)
        # # # # print(x.shape)
        # # # # exit()
        # # # # print(x)
        ttt=self.main_model1(x[0:144,:,:], ttt)
        ttt = self.main_model2(x[144:288, :, :], ttt)
        # # print(ttt.shape)
        # # print(x[0:216,:,:].shape)
        # # exit()

        x=self.main_model3(x[288:,:,:], ttt).permute(1,2,0).unsqueeze(1)
        x = self.mp1(x)
        x=F.relu(x)
        # print(x.shape)
        x=self.cv1(x)
        x=self.avg1(x).reshape(-1,10)
        # print(x.shape)
        # exit()
        # x=self.fc(x)
        # print(x.shape)
        # exit()
        # x=x.permute(1,2,0).unsqueeze(1)
        # # # print(x.shape)
        # # # exit()
        # x=self.bnx(x)
        # x=self.big_model(x)
        x = F.log_softmax(x, dim=-1)
        # print(x.shape)
        # exit()
        # print(x.shape)
        # exit()
        # if tar==None:
        #     x=self.main_model(x, x)
        # else:
        #     tar=tar.squeeze(1)
        #     tar=tar.permute(1,0,2)
        #     x=self.main_model(x, tar)
        #
        # x=x.permute(1,0,2)
        # x=self.fc(x)
        # # x=self.mp1(x).squeeze(1)
        # # print(x.shape)
        # # exit()
        # # x=self.fc(x)
        # # x=F.log_softmax(x, dim=-1)
        # x=F.softmax(x,dim=2)
        # x=x.mean(dim=1)
        # print(x)
        # print(x.shape)
        # exit()
        # x=x.sum(dim=1)
        # print(x)
        # # x=self.mp1(x)
        # print(x.shape)
        # exit()
        #
        # x=self.fc1(x)
        # print(x.shape)
        # exit()
        # x=self.fc1(x)
        # print(x.shape)
        # exit()
        # x=self.cv1(x)
        # x=self.mp1(x)
        # x=F.relu(x)
        # x=self.cv2(x)
        # x=self.mp2(x)
        # x=F.relu(x)
        # x=self.cv3(x)
        # x=self.mp3(x)
        #
        # x=self.avp(x).reshape(-1, 10)
        # x
        # print(x.shape)
        # exit()
        # # x=self.big_model(x)
        # # print(x.shape)
        # # exit()
        # a = x[1, 0, :, :].detach().cpu().numpy()
        # util_normal.plot_numpy(a)
        # exit()
        #
        # x=self.mp1(x)
        # # print(x.shape)
        # # exit()
        #
        # # util_normal.plot_numpy(a)
        # # exit()
        # # print(a)
        # # showing(a, 44100, 1024)
        # # a = x[1, 0, :, :].detach().cpu().numpy()
        # # util_normal.plot_numpy(a)
        # # exit()
        # x=self.cv1(x)
        # # a = x[1, 0, :, :].detach().cpu().numpy()
        # # util_normal.plot_numpy(a)
        # x=F.relu(x)
        # # a = x[1, 0, :, :].detach().cpu().numpy()
        # # util_normal.plot_numpy(a)
        # # x = F.relu(x)
        # print(x.shape)
        # exit()
        # x=self.av1(x).reshape(-1, 10)
        # # print(x)
        # x=self.fc1(x)
        # print(x)
        # print(x.shape)
        # exit()

        # x=x.unsqueeze(1)
        # x=self.cv1(x)
        # x=self.av1(x).reshape(-1, 10)
        # print(x.shape)
        # exit()
        # x=x.mean(dim=-1


        # x=F.log_softmax(x, dim=1)
        return x
    def _reset_parameters(self):


        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)