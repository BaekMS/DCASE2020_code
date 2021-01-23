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


        # self.main_model=custom_transformer.Transformer(d_model=256, nhead=8, dim_feedforward=2048, num_encoder_layers=4, num_decoder_layers=6)
        # self.freq_model=custom_transformer.Transformer(d_model=432, nhead=8, dim_feedforward=2048, num_encoder_layers=4, num_decoder_layers=6)

        self.big_model=small_trident.Model()
        # self.bnx=nn.BatchNorm2d(3)


        self._reset_parameters()



    def forward(self, x, tar):

        # x=x.squeeze(1).permute(2,0,1)
        # pn=x.norm(dim=(2), keepdim=True)
        # pn=x/pn
        # # print(pn.norm(dim=2))
        # # exit()
        # pp=self.main_model(pn, x).permute(1,2,0).unsqueeze(1)
        #
        #
        #
        # x=x.permute(2,1,0)
        # # px=self.freq_model(x,x).permute(1,0,2).unsqueeze(1)
        # x=x.permute(1,0,2).unsqueeze(1)
        # x=torch.cat((x, pp), dim=1)


        # util_normal.plot_numpy(pp[:,0,:].permute(1,0).to('cpu').detach().numpy())
        # util_normal.plot_numpy(x[:,0,:].permute(1,0).to('cpu').detach().numpy())

        # print(pp.shape)
        # pp=pp-pp.min(dim=1, keepdim=True)[0]
        # pp=pp/pp.max(dim=1, keepdim=True)[0]
        # # print(pp)
        # # exit()
        # # util_normal.plot_numpy(pp[:, 0, :].permute(1, 0).to('cpu').detach().numpy())
        #
        # x*=pp
        # util_normal.plot_numpy(x[:, 0, :].permute(1, 0).to('cpu').detach().numpy())
        # # print(x.shape)
        # exit()
        # x=x.permute(1,0,2).unsqueeze(1)
        # x=self.bnx(x)
        # print(x.shape)
        x=self.big_model(x)
        x = F.log_softmax(x, dim=-1)



        return x

    def _reset_parameters(self):


        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)