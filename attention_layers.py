import torch.nn as nn
import numpy as np
from collections import defaultdict
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch import Tensor

from layers import *


#    Self-Attention-GAN, slightly adapted https://github.com/heykeetae/Self-Attention-GAN/tree/master
class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim, query_key_out_channels=None, out_dim=None):
        super(SelfAttention,self).__init__()

        if query_key_out_channels == None:
            query_key_out_channels = in_dim//8

        if out_dim == None:
            out_dim = in_dim

        #self.chanel_in = in_dim
        #self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = query_key_out_channels , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = query_key_out_channels , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = out_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out #,attention
    

class AttentionResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k = 3, n_layers = 2, pooling = 'max'):
        super(AttentionResBlock, self).__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for ix in range(n_layers):
            self.convs.append(nn.Conv2d(in_channels, out_channels, (k, k), 
                                        stride = (1, 1), padding = ((k + 1) // 2 - 1, (k + 1) // 2 - 1)))
            self.norms.append(nn.Sequential(nn.InstanceNorm2d(out_channels), nn.Dropout2d(0.1)))
            
            in_channels = out_channels
    
        self.activation = nn.ELU()

        self.initial_attention = SelfAttention(in_channels)
        
    def forward(self, x, return_unpooled = False):
        x = self.initial_attention(x)

        xs = [self.norms[0](self.convs[0](x))]
        
        for ix in range(1, len(self.norms)):
            xs.append(self.norms[ix](self.convs[ix](xs[-1])) + xs[-1])
            
        x = self.activation(torch.cat(xs, dim = 1))
        
        return x