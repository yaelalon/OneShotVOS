from torch.autograd import Variable
import torch
import os
import numpy as np
import pickle
from collections import OrderedDict

def make_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def check_parallel(encoder_dict,decoder_dict):
	# check if the model was trained using multiple gpus
    trained_parallel = False
    for k, v in encoder_dict.items():
        if k[:7] == "module.":
            trained_parallel = True
        break;
    if trained_parallel:
        # create new OrderedDict that does not contain "module."
        new_encoder_state_dict = OrderedDict()
        new_decoder_state_dict = OrderedDict()
        for k, v in encoder_dict.items():
            name = k[7:]  # remove "module."
            new_encoder_state_dict[name] = v
        for k, v in decoder_dict.items():
            name = k[7:]  # remove "module."
            new_decoder_state_dict[name] = v
        encoder_dict = new_encoder_state_dict
        decoder_dict = new_decoder_state_dict

    return encoder_dict, decoder_dict

def get_base_params(args, model):
    b = []
    if 'vgg' in args.base_model:
        b.append(model.base.features)
    else:
        b.append(model.base.conv1)
        b.append(model.base.bn1)
        b.append(model.base.layer1)
        b.append(model.base.layer2)
        b.append(model.base.layer3)
        b.append(model.base.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def get_skip_params(model):
    b = []

    b.append(model.sk2.parameters())
    b.append(model.sk3.parameters())
    b.append(model.sk4.parameters())
    b.append(model.sk5.parameters())
    b.append(model.bn2.parameters())
    b.append(model.bn3.parameters())
    b.append(model.bn4.parameters())
    b.append(model.bn5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i

def merge_params(params):
    for j in range(len(params)):
        for i in params[j]:
            yield i

def get_optimizer(optim_name, lr, parameters, weight_decay = 0, momentum = 0.9):
    if optim_name == 'sgd':
        opt = torch.optim.SGD(filter(lambda p: p.requires_grad, parameters),
                                lr=lr, weight_decay = weight_decay,
                                momentum = momentum)
    elif optim_name =='adam':
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, parameters), lr=lr, weight_decay = weight_decay)
    elif optim_name =='rmsprop':
        opt = torch.optim.RMSprop(filter(lambda p: p.requires_grad, parameters), lr=lr, weight_decay = weight_decay)
    return opt

def save_checkpoint(args, encoder, decoder, enc_opt, dec_opt):
    torch.save(encoder.state_dict(), os.path.join('../models',args.model_name,'encoder.pt'))
    torch.save(decoder.state_dict(), os.path.join('../models',args.model_name,'decoder.pt'))
    torch.save(enc_opt.state_dict(), os.path.join('../models',args.model_name,'enc_opt.pt'))
    torch.save(dec_opt.state_dict(), os.path.join('../models',args.model_name,'dec_opt.pt'))
    # save parameters for future use
    pickle.dump(args, open(os.path.join('../models',args.model_name,'args.pkl'),'wb'))
    
def save_checkpoint_prev_mask(args, model_dir, epoch, encoder, decoder, enc_opt, dec_opt, isbest):
    if isbest: # Best model
        epoch_result_path = os.path.join(model_dir) 
        pickle.dump(args, open(os.path.join(model_dir,'args.pkl'),'wb'))
    else:
        epoch_result_path = os.path.join(model_dir,'epoch%d_results' %epoch)
        
    if not os.path.isdir(epoch_result_path):
        os.mkdir(epoch_result_path)  
        
    torch.save(encoder.state_dict(), os.path.join(epoch_result_path,'encoder.pt'))
    torch.save(decoder.state_dict(), os.path.join(epoch_result_path,'decoder.pt'))
    torch.save(enc_opt.state_dict(), os.path.join(epoch_result_path,'enc_opt.pt'))
    torch.save(dec_opt.state_dict(), os.path.join(epoch_result_path,'dec_opt.pt'))
    # save parameters for future use
    
def save_checkpoint_prev_inference_mask(args, encoder, decoder, enc_opt, dec_opt):
    torch.save(encoder.state_dict(), os.path.join('../models',args.model_name + '_prev_inference_mask','encoder.pt'))
    torch.save(decoder.state_dict(), os.path.join('../models',args.model_name + '_prev_inference_mask','decoder.pt'))
    torch.save(enc_opt.state_dict(), os.path.join('../models',args.model_name + '_prev_inference_mask','enc_opt.pt'))
    torch.save(dec_opt.state_dict(), os.path.join('../models',args.model_name + '_prev_inference_mask','dec_opt.pt'))
    # save parameters for future use
    pickle.dump(args, open(os.path.join('../models',args.model_name + '_prev_inference_mask','args.pkl'),'wb'))

def load_checkpoint(model_name,use_gpu=True):
    if use_gpu:
        encoder_dict = torch.load(os.path.join('../models',model_name,'encoder.pt'))
        decoder_dict = torch.load(os.path.join('../models',model_name,'decoder.pt'))
        enc_opt_dict = torch.load(os.path.join('../models',model_name,'enc_opt.pt'))
        dec_opt_dict = torch.load(os.path.join('../models',model_name,'dec_opt.pt'))
    else:
        encoder_dict = torch.load(os.path.join('../models',model_name,'encoder.pt'), map_location=lambda storage, location: storage)
        decoder_dict = torch.load(os.path.join('../models',model_name,'decoder.pt'), map_location=lambda storage, location: storage)
        enc_opt_dict = torch.load(os.path.join('../models',model_name,'enc_opt.pt'), map_location=lambda storage, location: storage)
        dec_opt_dict = torch.load(os.path.join('../models',model_name,'dec_opt.pt'), map_location=lambda storage, location: storage)
    # save parameters for future use
    args = pickle.load(open(os.path.join('../models',model_name,'args.pkl'),'rb'))

    return encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, args

def batch_to_var(args, inputs, targets):
    """
    Turns the output of DataLoader into data and ground truth to be fed
    during training
    """
    x = Variable(inputs,requires_grad=False)
    y_mask = Variable(targets[:,:,:-1].float(),requires_grad=False)
    sw_mask = Variable(targets[:,:,-1],requires_grad=False)

    if args.use_gpu:
        return x.cuda(), y_mask.cuda(), sw_mask.cuda()
    else:
        return x, y_mask, sw_mask
        
def batch_to_var_test(args, inputs):
    """
    Turns the output of DataLoader into data and ground truth to be fed
    during training
    """
    x = Variable(inputs,requires_grad=False)

    if args.use_gpu:
        return x.cuda()
    else:
        return x

def get_skip_dims(model_name):
    if model_name == 'resnet50' or model_name == 'resnet101':
        skip_dims_in = [2048,1024,512,256,64]
    elif model_name == 'resnet34':
        skip_dims_in = [512,256,128,64,64]
    elif model_name =='vgg16':
        skip_dims_in = [512,512,256,128,64]

    return skip_dims_in

def init_visdom(args,viz):

    # initialize visdom figures

    lot = viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1,4)).cpu(),
        opts=dict(
            xlabel='Iteration',
            ylabel='Loss',
            title='Training Losses',
            legend=['iou','total']
        )
    )

    elot = {}
    # epoch losses

    elot['iou'] = viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1,2)).cpu(),
        opts=dict(
            xlabel='Epoch',
            ylabel='Loss',
            title='sIoU Loss',
            legend = ['train','val']
        )
    )

    elot['total'] = viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1,2)).cpu(),
        opts=dict(
            xlabel='Epoch',
            ylabel='Loss',
            title='Total Loss',
            legend = ['train','val']
        )
    )

    mviz_pred = {}
    for i in range(args.maxseqlen):
        mviz_pred[i] = viz.heatmap(X=np.zeros((args.imsize,args.imsize)),
                                   opts=dict(title='Pred mask t'))

    mviz_true = {}
    for i in range(args.maxseqlen):
        mviz_true[i] = viz.heatmap(X=np.zeros((args.imsize,args.imsize)),
                                   opts=dict(title='True mask t'))


    image_lot = viz.image(np.ones((3,args.imsize,args.imsize)),
                        opts=dict(title='image'))


    return lot, elot, mviz_pred, mviz_true, image_lot

def outs_perms_to_cpu(args,outs,true_perm,h,w):
    # ugly function that turns contents of torch variables to numpy
    # (used for display during training)

    out_masks = outs
    y_mask_perm = true_perm[0]

    y_mask_perm = y_mask_perm.view(y_mask_perm.size(0),y_mask_perm.size(1),h,w)
    out_masks = out_masks.view(out_masks.size(0),out_masks.size(1),h,w)
    out_masks = out_masks.view(out_masks.size(0),out_masks.size(1),h,w)


    out_masks = out_masks.cpu().numpy()
    y_mask_perm = y_mask_perm.cpu().numpy()


    return out_masks, y_mask_perm

import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

def test(args, encoder, decoder, x, prev_hidden_temporal_list):

    """
    Runs forward, computes loss and (if train mode) updates parameters
    for the provided batch of inputs and targets
    """

    T = args.maxseqlen
    hidden_spatial = None
    hidden_temporal_list = []

    out_masks = []

    encoder.eval()
    decoder.eval()

    feats = encoder(x)
    
    # loop over sequence length and get predictions
    for t in range(0, T):
        #prev_hidden_temporal_list is a list with the hidden state for all instances from previous time instant
        #If this is the first frame of the sequence, hidden_temporal is initialized to None. Otherwise, it is set with the value from previous time instant.
        if prev_hidden_temporal_list is not None:
            hidden_temporal = prev_hidden_temporal_list[t]
            if args.only_temporal:
                hidden_spatial = None
        else:
            hidden_temporal = None
            
        #The decoder receives two hidden state variables: hidden_spatial (a tuple, with hidden_state and cell_state) which refers to the
        #hidden state from the previous object instance from the same time instant, and hidden_temporal which refers to the hidden state from the same
        #object instance from the previous time instant.
        out_mask, hidden = decoder(feats, hidden_spatial, hidden_temporal)
        hidden_tmp = []
        for ss in range(len(hidden)):
            hidden_tmp.append(hidden[ss][0].data)
        hidden_spatial = hidden
        hidden_temporal_list.append(hidden_tmp)

        upsample_match = nn.UpsamplingBilinear2d(size=(x.size()[-2], x.size()[-1]))
        out_mask = upsample_match(out_mask)
        out_mask = out_mask.view(out_mask.size(0), -1)

        # get predictions in list to concat later
        out_masks.append(out_mask)

        del hidden_temporal, hidden_tmp, out_mask

    # concat all outputs into single tensor to compute the loss
    t = len(out_masks)
    out_masks = torch.cat(out_masks,1).view(out_masks[0].size(0),len(out_masks), -1)
    out_masks = torch.sigmoid(out_masks)
    outs = out_masks.data
    
    del feats, x
    return outs, hidden_temporal_list

def get_prev_mask(prev_mask,x,feats,t):
    mask_lstm = []
    maxpool = nn.MaxPool2d((2, 2),ceil_mode=True)
    prev_mask_instance = prev_mask[:,t,:]
    prev_mask_instance = prev_mask_instance.view(prev_mask_instance.size(0),1,x.data.size(2),-1)
    prev_mask_instance = maxpool(prev_mask_instance)
    for ii in range(len(feats)):
        prev_mask_instance = maxpool(prev_mask_instance)
        mask_lstm.append(prev_mask_instance)
        
    mask_lstm = list(reversed(mask_lstm))
    return mask_lstm
    
def test_prev_mask(args, encoder, decoder, x, prev_hidden_temporal_list, hideen_temporal_first_list, prev_mask, mask_first):

    """
    Runs forward, computes loss and (if train mode) updates parameters
    for the provided batch of inputs and targets
    """

    T = args.maxseqlen
    hidden_spatial = None
    hidden_temporal_list = []

    out_masks = []

    encoder.eval()
    decoder.eval()
    encoder.train(False)
    decoder.train(False)

    feats = encoder(x)
    
    # loop over sequence length and get predictions
    for t in range(0, T):
        #prev_hidden_temporal_list is a list with the hidden state for all instances from previous time instant
        #If this is the first frame of the sequence, hidden_temporal is initialized to None. Otherwise, it is set with the value from previous time instant.
        hideen_temporal_first = None
        if prev_hidden_temporal_list is not None:
            hidden_temporal = prev_hidden_temporal_list[t]
            if args.use_GS_hidden:
                hideen_temporal_first = hideen_temporal_first_list[t]
        else:
            hidden_temporal = None
            hideen_temporal_first = None

        mask_lstm =  get_prev_mask(prev_mask,x,feats,t)
        if args.use_GS_hidden:
            mask_lstm_first = get_prev_mask(mask_first,x,feats,t)
        else:
            mask_lstm_first = None
        
        #The decoder receives two hidden state variables: hidden_spatial (a tuple, with hidden_state and cell_state) which refers to the
        #hidden state from the previous object instance from the same time instant, and hidden_temporal which refers to the hidden state from the same
        #object instance from the previous time instant.
        out_mask, hidden = decoder(args,feats, mask_lstm, mask_lstm_first, hidden_spatial, hidden_temporal, hideen_temporal_first)

        hidden_tmp = []
        for ss in range(len(hidden)):
            hidden_tmp.append(hidden[ss][0].data)
        hidden_spatial = hidden
        hidden_temporal_list.append(hidden_tmp)

        upsample_match = nn.UpsamplingBilinear2d(size=(x.size()[-2], x.size()[-1]))
        out_mask = upsample_match(out_mask)
        out_mask = out_mask.view(out_mask.size(0), -1)

        # get predictions in list to concat later
        out_masks.append(out_mask)
        
        del mask_lstm, mask_lstm_first,hidden_temporal, hidden_tmp, out_mask

    # concat all outputs into single tensor to compute the loss
    t = len(out_masks)
    out_masks = torch.cat(out_masks,1).view(out_masks[0].size(0),len(out_masks), -1)

    out_masks = torch.sigmoid(out_masks)
    outs = out_masks.data

    return outs, hidden_temporal_list
