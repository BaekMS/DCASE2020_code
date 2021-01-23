import copy
from typing import Optional, Any

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module, MultiheadAttention, ModuleList, Dropout, Linear, LayerNorm
# from .activation import MultiheadAttention
# from .container import ModuleList
from torch.nn.init import xavier_uniform_
# from .dropout import Dropout
# from .linear import Linear
# from .normalization import LayerNorm


class Model(nn.Module):
    def __init__(self, enco_d_model: int = 256, deco_d_model=10, enco_nhead: int = 8, deco_nhead=5, num_encoder_layers: int = 4,
                                                                                num_decoder_layers: int = 4, enco_dim_feedforward: int = 2048,deco_dim_feedforward=2048, dropout: float = 0.1,
                                                                                                                                                           activation: str = "relu", custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None) -> None:
        super(Model, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(enco_d_model, enco_nhead, enco_dim_feedforward, dropout, activation)
            encoder_norm = LayerNorm(enco_d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(deco_d_model, deco_nhead, deco_dim_feedforward, dropout, activation)
            decoder_norm = LayerNorm(deco_d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.enco_d_model = enco_d_model
        self.enco_nhead = enco_nhead
        self.deco_d_model = deco_d_model
        self.deco_nhead = deco_nhead
        self.fc1=nn.Linear(256,10)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:



        src=src.squeeze(1).permute(2,0,1)
        if tgt==None:
            tgt=torch.zeros((1, src.shape[1], 10), dtype=torch.float32).to('cuda')
        for s in range(4):
            if s==0:
                memory = self.encoder(src[0:108*(s+1),:,:], mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            else:
                memory = self.encoder(torch.cat((memory,src[108*s:108 * (s + 1), :, :]), dim=0), mask=src_mask,
                                      src_key_padding_mask=src_key_padding_mask)
            d_memory=self.fc1(memory)

            tgt = self.decoder(tgt, d_memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)



        output=F.log_softmax(tgt.squeeze(0), dim=-1)
        # print(output.shape)
        # exit()

        return output

class TransformerEncoder(Module):

    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        output = src


        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDecoder(Module):
        r"""TransformerDecoder is a stack of N decoder layers

        Args:
            decoder_layer: an instance of the TransformerDecoderLayer() class (required).
            num_layers: the number of sub-decoder-layers in the decoder (required).
            norm: the layer normalization component (optional).

        Examples::
            >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
            >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
            >>> memory = torch.rand(10, 32, 512)
            >>> tgt = torch.rand(20, 32, 512)
            >>> out = transformer_decoder(tgt, memory)
        """
        __constants__ = ['norm']

        def __init__(self, decoder_layer, num_layers, norm=None):
            super(TransformerDecoder, self).__init__()
            self.layers = _get_clones(decoder_layer, num_layers)
            self.num_layers = num_layers
            self.norm = norm

        def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
            r"""Pass the inputs (and mask) through the decoder layer in turn.

            Args:
                tgt: the sequence to the decoder (required).
                memory: the sequence from the last layer of the encoder (required).
                tgt_mask: the mask for the tgt sequence (optional).
                memory_mask: the mask for the memory sequence (optional).
                tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
                memory_key_padding_mask: the mask for the memory keys per batch (optional).

            Shape:
                see the docs in Transformer class.
            """
            output = tgt

            for mod in self.layers:
                output = mod(output, memory, tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask)

            if self.norm is not None:
                output = self.norm(output)

            return output

class TransformerEncoderLayer(Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:



        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))