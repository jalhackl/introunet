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
from attention_layers import *


class NestedUNetLSTM(nn.Module):
    def __init__(self, num_classes, input_channels=2, filter_multiplier = 1, hidden_dim=4, n_layers=1, create_gru=True, bidirectional=True, polymorphisms=128,
                 deep_supervision=False, small = False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        nb_filter = list(map(int, [u * filter_multiplier for u in nb_filter]))

        self.deep_supervision = deep_supervision
        self.small = small

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True)

        self.dropout = nn.Dropout2d(0.1)

        self.conv0_0 = ResBlock(input_channels, nb_filter[0] // 2)
        self.conv1_0 = ResBlock(nb_filter[0], nb_filter[1] // 2)
        self.conv2_0 = ResBlock(nb_filter[1], nb_filter[2] // 2)
        self.conv3_0 = ResBlock(nb_filter[2], nb_filter[3] // 2)

        #new
        #self.out = nn.Sequential(nn.Linear(128, 128), nn.LayerNorm((128,)), nn.ReLU(),
        #                    nn.Linear(128, 3), nn.Softmax(dim = -1))


        self.out = nn.Sequential(nn.Linear(polymorphisms, 256),  nn.LayerNorm((256,)), nn.Linear(256, polymorphisms))

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        if create_gru ==True:
            self.out_lstm = nn.GRU(2, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=bidirectional)
        else:
            self.out_lstm = nn.LSTM(2, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=bidirectional)


        if not self.small:
            self.conv4_0 = ResBlock(nb_filter[3], nb_filter[4] // 2)
            self.conv0_4 = ResBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0] // 2)

            self.conv2_2 = ResBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2] // 2)
            self.conv1_3 = ResBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1] // 2)
            self.conv3_1 = ResBlock(nb_filter[3]+nb_filter[4], nb_filter[3] // 2)

        self.conv0_1 = ResBlock(nb_filter[0]+nb_filter[1], nb_filter[0] // 2)
        self.conv1_1 = ResBlock(nb_filter[1]+nb_filter[2], nb_filter[1] // 2)
        self.conv2_1 = ResBlock(nb_filter[2]+nb_filter[3], nb_filter[2] // 2)

        self.conv0_2 = ResBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0] // 2)
        self.conv1_2 = ResBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1] // 2)

        self.conv0_3 = ResBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0] // 2)


        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        new_input = input


        new_input = new_input.transpose(1,0)
        new_input_conv = new_input[0:2]
        new_input_seq = new_input[2]
        new_input_conv = new_input_conv.transpose(1,0)


        x0_0 = self.conv0_0(new_input_conv)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        if not self.small:
            x4_0 = self.conv4_0(self.pool(x3_0))
            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
            x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            if not self.small:

                output = torch.squeeze(self.final(x0_4))


                oldshape = output.shape

                new_input_seq = torch.flatten(new_input_seq, start_dim=0, end_dim=1 )


                output_rows = torch.flatten(output, start_dim=0, end_dim=1 )


                final_out_lstm = torch.stack((output_rows, new_input_seq), axis =2 )


                outputt_lstm, (hx,ha) = self.out_lstm(final_out_lstm)

                outputt_lstm = outputt_lstm.reshape(outputt_lstm.shape[0], outputt_lstm.shape[1], 2, self.hidden_dim)

                outputt_lstm = outputt_lstm[..., -1:]
                outputt_lstm = torch.squeeze(outputt_lstm)
                outputt_lstm = torch.mean(outputt_lstm, -1)

                final_output = self.out(outputt_lstm)

                outputt5 = torch.reshape(final_output, oldshape)
                output = outputt5

            else:
                output = torch.squeeze(self.final(x0_3))
            return output



class NestedUNetLSTM_fwbw(nn.Module):
    def __init__(self, num_classes, input_channels=2, filter_multiplier = 1, hidden_dim=4, n_layers=1, create_gru=True, bidirectional=True, polymorphisms=128, 
                 deep_supervision=False, small = False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        nb_filter = list(map(int, [u * filter_multiplier for u in nb_filter]))

        self.deep_supervision = deep_supervision
        self.small = small

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True)

        self.dropout = nn.Dropout2d(0.1)

        self.conv0_0 = ResBlock(input_channels, nb_filter[0] // 2)
        self.conv1_0 = ResBlock(nb_filter[0], nb_filter[1] // 2)
        self.conv2_0 = ResBlock(nb_filter[1], nb_filter[2] // 2)
        self.conv3_0 = ResBlock(nb_filter[2], nb_filter[3] // 2)



        self.out = nn.Sequential(nn.Linear(polymorphisms, 256),  nn.LayerNorm((256,)), nn.Linear(256, polymorphisms))

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        if create_gru ==True:
            self.out_lstm = nn.GRU(3, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=bidirectional)
        else:
            self.out_lstm = nn.LSTM(3, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=bidirectional)


        if not self.small:
            self.conv4_0 = ResBlock(nb_filter[3], nb_filter[4] // 2)
            self.conv0_4 = ResBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0] // 2)

            self.conv2_2 = ResBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2] // 2)
            self.conv1_3 = ResBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1] // 2)
            self.conv3_1 = ResBlock(nb_filter[3]+nb_filter[4], nb_filter[3] // 2)

        self.conv0_1 = ResBlock(nb_filter[0]+nb_filter[1], nb_filter[0] // 2)
        self.conv1_1 = ResBlock(nb_filter[1]+nb_filter[2], nb_filter[1] // 2)
        self.conv2_1 = ResBlock(nb_filter[2]+nb_filter[3], nb_filter[2] // 2)

        self.conv0_2 = ResBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0] // 2)
        self.conv1_2 = ResBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1] // 2)

        self.conv0_3 = ResBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0] // 2)


        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        new_input = input

        new_input = new_input.transpose(1,0)
        new_input_conv = new_input[0:2]
        new_input_seq = new_input[2:4]

        new_input_conv = new_input_conv.transpose(1,0)


        x0_0 = self.conv0_0(new_input_conv)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        if not self.small:
            x4_0 = self.conv4_0(self.pool(x3_0))
            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
            x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            if not self.small:

                output = torch.squeeze(self.final(x0_4))

                oldshape = output.shape

                new_input_seq = torch.permute(new_input_seq, (1,2,3,0))

                new_input_seq = torch.flatten(new_input_seq, start_dim=0, end_dim=1 )

                output_rows = torch.flatten(output, start_dim=0, end_dim=1 )


                output_rows = torch.unsqueeze(output_rows, -1)

                final_out = torch.cat((output_rows, new_input_seq), axis =-1 )

                final_out_lstm = final_out

                outputt_lstm, (hx,ha) = self.out_lstm(final_out_lstm)

                outputt_lstm = outputt_lstm.reshape(outputt_lstm.shape[0], outputt_lstm.shape[1], 2, self.hidden_dim)

                outputt_lstm = outputt_lstm[..., -1:]
                outputt_lstm = torch.squeeze(outputt_lstm)
                outputt_lstm = torch.mean(outputt_lstm, -1)

                final_output = self.out(outputt_lstm)


                outputt5 = torch.reshape(final_output, oldshape)
                output = outputt5

            else:
                output = torch.squeeze(self.final(x0_3))
            return output






#This version accepts a second vector (e.g. positions) as input which is processed via a FNN
class NestedUNetExtraPos(nn.Module):
    def __init__(self, num_classes, input_channels=3, filter_multiplier = 1, polymorphisms=128,
                 deep_supervision=False, small = False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        nb_filter = list(map(int, [u * filter_multiplier for u in nb_filter]))

        self.deep_supervision = deep_supervision
        self.small = small

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True)

        self.dropout = nn.Dropout2d(0.1)

        self.conv0_0 = ResBlock(input_channels, nb_filter[0] // 2)
        self.conv1_0 = ResBlock(nb_filter[0], nb_filter[1] // 2)
        self.conv2_0 = ResBlock(nb_filter[1], nb_filter[2] // 2)
        self.conv3_0 = ResBlock(nb_filter[2], nb_filter[3] // 2)

        #new
        #self.out = nn.Sequential(nn.Linear(polymorphisms, 128), nn.LayerNorm((128,)), nn.ReLU(),
        #                    nn.Linear(128, 3), nn.Softmax(dim = -1))

        self.out = nn.Sequential(nn.Linear(polymorphisms, 512),  nn.ReLU(),
                        nn.Linear(512, polymorphisms), nn.Sigmoid())

        if not self.small:
            self.conv4_0 = ResBlock(nb_filter[3], nb_filter[4] // 2)
            self.conv0_4 = ResBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0] // 2)

            self.conv2_2 = ResBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2] // 2)
            self.conv1_3 = ResBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1] // 2)
            self.conv3_1 = ResBlock(nb_filter[3]+nb_filter[4], nb_filter[3] // 2)

        self.conv0_1 = ResBlock(nb_filter[0]+nb_filter[1], nb_filter[0] // 2)
        self.conv1_1 = ResBlock(nb_filter[1]+nb_filter[2], nb_filter[1] // 2)
        self.conv2_1 = ResBlock(nb_filter[2]+nb_filter[3], nb_filter[2] // 2)

        self.conv0_2 = ResBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0] // 2)
        self.conv1_2 = ResBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1] // 2)

        self.conv0_3 = ResBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0] // 2)


        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        new_input = input

        new_input = new_input.transpose(1,0)
        new_input_conv = new_input[0:2]
        new_input_seq = new_input[2]
        new_input_conv = new_input_conv.transpose(1,0)


        x0_0 = self.conv0_0(new_input_conv)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        if not self.small:
            x4_0 = self.conv4_0(self.pool(x3_0))
            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
            x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            if not self.small:

                output = torch.squeeze(self.final(x0_4))
                oldshape = output.shape

                new_input_seq = torch.flatten(new_input_seq, start_dim=0, end_dim=1 )

                output_rows = torch.flatten(output, start_dim=0, end_dim=1 )

                outputt4 = self.out(output_rows)
                outputt5 = torch.reshape(outputt4, oldshape)

                output = outputt5

            else:
                output = torch.squeeze(self.final(x0_3))
            return output



    


#additional attention layer
class NestedUNetAttention(nn.Module):
    def __init__(self, num_classes, input_channels=3, filter_multiplier = 1, 
                 deep_supervision=False, small = False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        nb_filter = list(map(int, [u * filter_multiplier for u in nb_filter]))

        self.deep_supervision = deep_supervision
        self.small = small

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True)

        self.dropout = nn.Dropout2d(0.1)

        #additional attention layer

        self.self_attention = SelfAttention(input_channels)

        self.conv0_0 = ResBlock(input_channels, nb_filter[0] // 2)
        self.conv1_0 = ResBlock(nb_filter[0], nb_filter[1] // 2)
        self.conv2_0 = ResBlock(nb_filter[1], nb_filter[2] // 2)
        self.conv3_0 = ResBlock(nb_filter[2], nb_filter[3] // 2)
        
        if not self.small:
            self.conv4_0 = ResBlock(nb_filter[3], nb_filter[4] // 2)
            self.conv0_4 = ResBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0] // 2)
            
            self.conv2_2 = ResBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2] // 2)
            self.conv1_3 = ResBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1] // 2)
            self.conv3_1 = ResBlock(nb_filter[3]+nb_filter[4], nb_filter[3] // 2)

        self.conv0_1 = ResBlock(nb_filter[0]+nb_filter[1], nb_filter[0] // 2)
        self.conv1_1 = ResBlock(nb_filter[1]+nb_filter[2], nb_filter[1] // 2)
        self.conv2_1 = ResBlock(nb_filter[2]+nb_filter[3], nb_filter[2] // 2)

        self.conv0_2 = ResBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0] // 2)
        self.conv1_2 = ResBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1] // 2)

        self.conv0_3 = ResBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0] // 2)
        

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x_attention = self.self_attention(input)

        x0_0 = self.conv0_0(x_attention)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        if not self.small:
            x4_0 = self.conv4_0(self.pool(x3_0))
            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
            x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            if not self.small:
                output = torch.squeeze(self.final(x0_4))
            else:
                output = torch.squeeze(self.final(x0_3))
            return output
        


#additional attention layer
class NestedUNetAttentionBlock(nn.Module):
    def __init__(self, num_classes, input_channels=3, filter_multiplier = 1, 
                 deep_supervision=False, small = False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        nb_filter = list(map(int, [u * filter_multiplier for u in nb_filter]))

        self.deep_supervision = deep_supervision
        self.small = small

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True)

        self.dropout = nn.Dropout2d(0.1)

        #additional attention layer

        self.self_attention = SelfAttention(input_channels)

        self.conv0_0 = ResBlock(input_channels, nb_filter[0] // 2)
        self.conv1_0 = AttentionResBlock(nb_filter[0], nb_filter[1] // 2)
        self.conv2_0 = AttentionResBlock(nb_filter[1], nb_filter[2] // 2)
        self.conv3_0 = AttentionResBlock(nb_filter[2], nb_filter[3] // 2)
        
        if not self.small:
            self.conv4_0 = AttentionResBlock(nb_filter[3], nb_filter[4] // 2)
            self.conv0_4 = ResBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0] // 2)
            
            self.conv2_2 = ResBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2] // 2)
            self.conv1_3 = ResBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1] // 2)
            self.conv3_1 = ResBlock(nb_filter[3]+nb_filter[4], nb_filter[3] // 2)

        self.conv0_1 = ResBlock(nb_filter[0]+nb_filter[1], nb_filter[0] // 2)
        self.conv1_1 = ResBlock(nb_filter[1]+nb_filter[2], nb_filter[1] // 2)
        self.conv2_1 = ResBlock(nb_filter[2]+nb_filter[3], nb_filter[2] // 2)

        self.conv0_2 = ResBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0] // 2)
        self.conv1_2 = ResBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1] // 2)

        self.conv0_3 = ResBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0] // 2)
        

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        #x_attention = self.self_attention(input)

        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        if not self.small:
            x4_0 = self.conv4_0(self.pool(x3_0))
            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
            x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            if not self.small:
                output = torch.squeeze(self.final(x0_4))
            else:
                output = torch.squeeze(self.final(x0_3))
            return output