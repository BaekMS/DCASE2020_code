import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt




def plotting(data, inin, outout, maxi, mini):
    save_loc = './testing/'
    print(data.to('cpu'))
    numpp=data.to('cpu').clone().detach().numpy()
    # print(numpp.shape)
    # exit()
    maxi=numpp.max()
    mini=numpp.min()
    for j in range(numpp.shape[0]):
        fig=plt.figure()
        ax=fig.add_subplot(111)
        # ax.set_title('yeah')
        plt.imshow(numpp[j,:,:], cmap ='Greys')
        ax.set_aspect('equal')
        plt.clim(mini, maxi)

        plt.colorbar(orientation='vertical')
    # plt.pcolor(X, Y, f(data), cmap=cm, vmin=-4, vmax=4)

    # plt.colorbar()
    #     plt.show()
    #     exit()
        fig.savefig(save_loc + str(j)+'_'+'.png', dpi=300)
        plt.close()
    exit()
class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        # torch.cuda.empty_cache()

        self.avp1 = nn.AvgPool2d((1, 2), stride=(1, 2), padding=(0, 1))
        self.bn11=nn.BatchNorm2d(3)
        self.cv11=nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=(1,2), padding=(2,1), dilation=(2,1), bias=False)
        self.bn12=nn.BatchNorm2d(32, affine=False)
        self.cv12=nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=(1,1), padding=(2,1), dilation=(2,1), bias=False)
        self.bn13 = nn.BatchNorm2d(32, affine=False)
        self.cv13 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=(1, 1), padding=(2, 1),
                              dilation=(2, 1), bias=False)

        self.avp2 = nn.AvgPool2d((1, 2), stride=(1, 2), padding=(0, 0))
        self.bn21 = nn.BatchNorm2d(32)
        self.cv21 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=(1, 2), padding=(2, 1),
                             dilation=(2, 1), bias=False)
        self.bn22 = nn.BatchNorm2d(64, affine=False)
        self.cv22 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=(1, 1), padding=(2, 1),
                             dilation=(2, 1), bias=False)
        self.bn23 = nn.BatchNorm2d(64, affine=False)
        self.cv23 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=(1, 1), padding=(2, 1),
                              dilation=(2, 1), bias=False)

        self.avp3 = nn.AvgPool2d((1, 2), stride=(1, 2), padding=(0, 0))
        self.bn31 = nn.BatchNorm2d(64)
        self.cv31 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=(1, 2), padding=(2, 1),
                              dilation=(2, 1), bias=False)
        self.bn32 = nn.BatchNorm2d(128, affine=False)
        self.cv32 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=(1, 1), padding=(2, 1),
                              dilation=(2, 1), bias=False)
        self.bn33 = nn.BatchNorm2d(128, affine=False)
        self.cv33 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=(1, 1), padding=(2, 1),
                              dilation=(2, 1), bias=False)

        self.avp4 = nn.AvgPool2d((1, 2), stride=(1, 2), padding=(0, 1))
        self.bn41 = nn.BatchNorm2d(128)
        self.cv41 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=(1, 2), padding=(2, 1),
                              dilation=(2, 1), bias=False)
        self.bn42 = nn.BatchNorm2d(256, affine=False)
        self.cv42 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=(1, 1), padding=(2, 1),
                              dilation=(2, 1), bias=False)
        self.bn43 = nn.BatchNorm2d(256, affine=False)
        self.cv43 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=(1, 1), padding=(2, 1),
                              dilation=(2, 1), bias=False)



    def forward(self, x):
        # print(x.shape)
        avg1=self.avp1(x)
        pad1=(0,0, 0,0, 0,29, 0,0)

        avg1=F.pad(avg1, pad1, 'constant', 0)

        x=self.bn11(x)
        x=F.relu(x)
        x=self.cv11(x)
        x=self.bn12(x)
        x = F.relu(x)
        x=self.cv12(x)

        x = self.bn13(x)
        x = F.relu(x)
        x = self.cv13(x)

        # plotting(x[0], None, None, None, None)

        x+=avg1


        avg2 = self.avp2(x)
        pad2 = (0, 0, 0, 0, 0, 32, 0, 0)

        avg2 = F.pad(avg2, pad2, 'constant', 0)

        x = self.bn21(x)
        x = F.relu(x)
        x = self.cv21(x)
        x = self.bn22(x)
        x = F.relu(x)
        x = self.cv22(x)
        x = self.bn23(x)
        x = F.relu(x)
        x = self.cv23(x)
        x += avg2

        avg3 = self.avp3(x)
        pad3 = (0, 0, 0, 0, 0, 64, 0, 0)

        avg3 = F.pad(avg3, pad3, 'constant', 0)
        x = self.bn31(x)
        x = F.relu(x)
        x = self.cv31(x)
        x = self.bn32(x)
        x = F.relu(x)
        x = self.cv32(x)
        x = self.bn33(x)
        x = F.relu(x)
        x = self.cv33(x)

        x += avg3

        avg4 = self.avp4(x)
        pad4 = (0, 0, 0, 0, 0, 128, 0, 0)
        avg4 = F.pad(avg4, pad4, 'constant', 0)
        x = self.bn41(x)
        x = F.relu(x)
        x = self.cv41(x)
        x = self.bn42(x)
        x = F.relu(x)
        x = self.cv42(x)
        x = self.bn43(x)
        x = F.relu(x)
        x = self.cv43(x)
        # print(x.shape)
        # print(avg4.shape)
        # exit()
        x += avg4

        # p
        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.resmodule = nn.ModuleList([ResBlock() for i in range(3)])
        self.bn1=nn.BatchNorm2d(256)
        self.cv1=nn.Conv2d(in_channels=256, out_channels=768, kernel_size=1)

        self.bn2=nn.BatchNorm2d(768)
        self.cv2=nn.Conv2d(in_channels=768, out_channels=10, kernel_size=1)
        self.bn3=nn.BatchNorm2d(10)

        self.avp = nn.AvgPool2d((256, 27))

        self._reset_parameters()

    def forward(self, x):
        # print(x.shape)
        # exit()
        x=torch.cat((self.resmodule[0](x[:,:,0:64,:]), self.resmodule[1](x[:,:,64:128,:]), self.resmodule[2](x[:,:,128:256,:])), dim=2)


        x=self.bn1(x)


        x=F.relu(x)

        x=self.cv1(x)

        x=self.bn2(x)


        x=self.cv2(x)

        x=self.bn3(x)
        # print(x.shape)
        # exit()


        x= self.avp(x)

        x = x.view(-1, 10)


        x=F.log_softmax(x, dim=1)

        return x

    def _reset_parameters(self):


        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p)