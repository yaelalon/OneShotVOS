import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, args, input_size, hidden_size, kernel_size, padding):
        super(ConvLSTMCell,self).__init__()
        self.use_gpu = args.use_gpu
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + 2*hidden_size, 4 * hidden_size, kernel_size, padding=padding)

    def forward(self, input_, prev_state_spatial, hidden_state_temporal):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state_spatial is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if self.use_gpu:
                prev_state_spatial = (
                    Variable(torch.zeros(state_size)).cuda(),
                    Variable(torch.zeros(state_size)).cuda()
                )
            else:
                prev_state_spatial = (
                    Variable(torch.zeros(state_size)),
                    Variable(torch.zeros(state_size))
                )
                
        if hidden_state_temporal is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if self.use_gpu:
                hidden_state_temporal = Variable(torch.zeros(state_size)).cuda()
            else:
                hidden_state_temporal = Variable(torch.zeros(state_size))


        prev_hidden_spatial, prev_cell_spatial = prev_state_spatial

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_hidden_spatial, hidden_state_temporal], 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = f.tanh(cell_gate)
        # compute current cell and hidden state
        cell = (remember_gate * prev_cell_spatial) + (in_gate * cell_gate)
        hidden = out_gate * f.tanh(cell)

        state = [hidden,cell]

        return state

        
class ConvLSTMCellMask(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, args, input_size, hidden_size, kernel_size, padding):
        super(ConvLSTMCellMask,self).__init__()
        self.use_gpu = args.use_gpu
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_GS_hidden = args.use_GS_hidden
        self.Gates = nn.Conv2d(input_size + 2*hidden_size + 1, 4 * hidden_size, kernel_size, padding=padding)
        
        if args.use_GS_hidden:
            self.hidden_conv = nn.Conv2d(2*hidden_size,hidden_size,kernel_size,padding=padding)
            self.mask_conv = nn.Conv2d(2,1,kernel_size,padding=padding)
        
    def get_hidden(self,hidden,batch_size,spatial_size):
        # The function convets the hidden tensor to torch of zeros in xase it's None
        if hidden is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if self.use_gpu:
                hidden = Variable(torch.zeros(state_size)).cuda()
            else:
                hidden = Variable(torch.zeros(state_size))
        return hidden
            
    def forward(self, input_, prev_mask, prev_state_spatial, hidden_state_temporal,mask_first,hideen_temporal_first):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state_spatial is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if self.use_gpu:
                prev_state_spatial = (
                    Variable(torch.zeros(state_size)).cuda(),
                    Variable(torch.zeros(state_size)).cuda()
                )
            else:
                prev_state_spatial = (
                    Variable(torch.zeros(state_size)),
                    Variable(torch.zeros(state_size))
                )
        prev_hidden_spatial, prev_cell_spatial = prev_state_spatial
                
        hidden_state_temporal = self.get_hidden(hidden_state_temporal,batch_size,spatial_size)
        hideen_temporal_first = self.get_hidden(hideen_temporal_first,batch_size,spatial_size)
        
        if self.use_GS_hidden: # Evaluate hidden also from first
           temporal_comb = torch.cat([hidden_state_temporal,hideen_temporal_first],1)
           hidden_state_temporal = self.hidden_conv(temporal_comb)
           mask_comb = torch.cat([prev_mask,mask_first],1)           
           prev_mask = self.mask_conv(mask_comb) 
           del hideen_temporal_first, mask_first

           
        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_mask, prev_hidden_spatial, hidden_state_temporal], 1)
        del prev_hidden_spatial, hidden_state_temporal, prev_mask, input_
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = f.tanh(cell_gate)
        # compute current cell and hidden state
        cell = (remember_gate * prev_cell_spatial) + (in_gate * cell_gate)
        hidden = out_gate * f.tanh(cell)
        del cell_gate, out_gate, remember_gate, in_gate, gates, stacked_inputs

        state = [hidden,cell]

        return state
