import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from args import get_parser
from utils.utils import batch_to_var, batch_to_var_test, make_dir, outs_perms_to_cpu, load_checkpoint, check_parallel, test, test_prev_mask
from modules.model import RSISMask, FeatureExtractor
#from test import test, test_prev_mask
from dataloader.dataset_utils import sequence_palette
from PIL import Image
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize
from scipy.misc import toimage
#import scipy
from dataloader.dataset_utils import get_dataset
from torchvision import models
import torch
import numpy as np
from torchvision import transforms
import torch.utils.data as data
import sys, os
import json
from torch.autograd import Variable
import time
import os.path as osp
from measures.f_boundary import db_eval_boundary as eval_F
from measures.jaccard import db_eval_iou as jaccard_simple
import pickle


class Evaluate():

    def __init__(self,args):

        self.split = args.eval_split
        self.dataset = args.dataset
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        image_transforms = transforms.Compose([to_tensor,normalize])
        
        if args.dataset == 'davis2017':
            dataset = get_dataset(args,
                                split=self.split,
                                image_transforms=image_transforms,
                                target_transforms=None,
                                augment=args.augment and self.split == 'train',
                                inputRes = (240,427),
                                video_mode = True,
                                use_prev_mask = True)
        else: #args.dataset == 'youtube'
            dataset = get_dataset(args,
                                split=self.split,
                                image_transforms=image_transforms,
                                target_transforms=None,
                                augment=args.augment and self.split == 'train',
                                inputRes = (256, 448),
                                video_mode = True,
                                use_prev_mask = True)

        self.loader = data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.num_workers,
                                         drop_last=False)
        

        self.args = args

        print(args.model_name)
        encoder_dict, decoder_dict, _, _, load_args = load_checkpoint(args.model_name,args.use_gpu)
        load_args.use_gpu = args.use_gpu
        self.encoder = FeatureExtractor(load_args)
        self.decoder = RSISMask(load_args)

        print(load_args)

        if args.ngpus > 1 and args.use_gpu:
            self.decoder = torch.nn.DataParallel(self.decoder,device_ids=range(args.ngpus))
            self.encoder = torch.nn.DataParallel(self.encoder,device_ids=range(args.ngpus))

        encoder_dict, decoder_dict = check_parallel(encoder_dict,decoder_dict)
        self.encoder.load_state_dict(encoder_dict)
        
        to_be_deleted_dec = []
        for k in decoder_dict.keys():
            if 'fc_stop' in k:
                to_be_deleted_dec.append(k)
        for k in to_be_deleted_dec:
            del decoder_dict[k]
        self.decoder.load_state_dict(decoder_dict)

        if args.use_gpu:
            self.encoder.cuda()
            self.decoder.cuda()

        self.encoder.eval()
        self.decoder.eval()
        if load_args.length_clip == 1:
            self.video_mode = False
            print('video mode not activated')
        else:
            self.video_mode = True
            print('video mode activated')

    def run_eval(self):
        print ("Dataset is %s"%(self.dataset))
        print ("Split is %s"%(self.split))

        if args.overlay_masks:

            colors = []
            palette = sequence_palette()
            inv_palette = {}
            for k, v in palette.items():
                inv_palette[v] = k
            num_colors = len(inv_palette.keys())
            for id_color in range(num_colors):
                if id_color == 0 or id_color == 21:
                    continue
                c = inv_palette[id_color]
                colors.append(c)

        jaccard_vec, F_vec= [],[]
        
        if args.dataset == 'youtube':

            masks_sep_dir = os.path.join('../models', args.model_name, 'masks_sep_2assess')
            make_dir(masks_sep_dir)
            if args.overlay_masks:
                results_dir = os.path.join('../models', args.model_name, 'results')
                make_dir(results_dir)
        
            json_data = open('../../databases/YouTubeVOS/train/train-val-meta.json')
            data = json.load(json_data)

        else: #args.dataset == 'davis2017'

            import lmdb
            from misc.config import cfg

            masks_sep_dir = os.path.join('../models', args.model_name, 'masks_sep_2assess-davis')
            make_dir(masks_sep_dir)

            if args.overlay_masks:
                results_dir = os.path.join('../models', args.model_name, 'results-davis')
                make_dir(results_dir)

            lmdb_env_seq_dir = osp.join(cfg.PATH.DATA, 'lmdb_seq')

            if osp.isdir(lmdb_env_seq_dir):
                lmdb_env_seq = lmdb.open(lmdb_env_seq_dir)
            else:
                lmdb_env_seq = None
            
        for batch_idx, (inputs, inputs_flip, targets, targets_flip,seq_name,starting_frame) in enumerate(self.loader):
            prev_hidden_temporal_list = None
            hideen_temporal_first = None
            max_ii = min(len(inputs),args.length_clip)

            if args.overlay_masks:
                base_dir = results_dir + '/' + seq_name[0] + '/'
                make_dir(base_dir)

            if args.dataset == 'davis2017':
                key_db = osp.basename(seq_name[0])

                if not lmdb_env_seq == None:
                    with lmdb_env_seq.begin() as txn:
                        _files_vec = txn.get(key_db.encode()).decode().split('|')
                        _files = [osp.splitext(f)[0] for f in _files_vec]
                else:
                    seq_dir = osp.join(cfg['PATH']['SEQUENCES'], key_db)
                    _files_vec = os.listdir(seq_dir)
                    _files = [osp.splitext(f)[0] for f in _files_vec]

                frame_names = sorted(_files)
                
            for ii in range(max_ii):
                x, y_mask, sw_mask = batch_to_var(args, inputs[ii], targets[ii])

                if ii == 0:
                    prev_mask = y_mask
                    if args.use_GS_hidden:
                        mask_first = y_mask
                    else:
                        mask_first = None
                
                #from one frame to the following frame the prev_hidden_temporal_list is updated.
                outs, hidden_temporal_list = test_prev_mask(args, self.encoder, self.decoder, x,hideen_temporal_first, prev_hidden_temporal_list, prev_mask, mask_first)

                if ii == 0:
                    hideen_temporal_first = hidden_temporal_list
                            
                # Check measures
                jaccard_vec.append(jaccard_simple(y_mask.cpu().numpy(),outs.cpu().numpy()))
                #F_vec.append(eval_F(y_mask.cpu().numpy(),outs.cpu().numpy()))
                
                if args.dataset == 'youtube':
                    num_instances = len(data['videos'][seq_name[0]]['objects'])
                else:
                    num_instances = int(torch.sum(sw_mask.data).data.cpu().numpy())

                base_dir_masks_sep = masks_sep_dir + '/' + seq_name[0] + '/'
                make_dir(base_dir_masks_sep)

                x_tmp = x.data.cpu().numpy()
                height = x_tmp.shape[-2]
                width = x_tmp.shape[-1]
                for t in range(num_instances):
                    mask_pred = (torch.squeeze(outs[0,t,:])).cpu().numpy()
                    mask_pred = np.reshape(mask_pred, (height, width))
                    
                    GS = (torch.squeeze(y_mask[0,t,:])).cpu().numpy()
                    GS = np.reshape(GS, (height, width))
                                                
                    indxs_instance = np.where(mask_pred > 0.5)
                    mask2assess = np.zeros((height,width))
                    maskboll = np.zeros((height,width))
                    mask2assess[indxs_instance] = 255
                    maskboll[indxs_instance] = 1
                    
                    jaccard_vec.append(jaccard_simple(GS,maskboll))
                    F_vec.append(eval_F(GS,maskboll))
                    
                    if args.dataset == 'youtube':
                        toimage(mask2assess, cmin=0, cmax=255).save(base_dir_masks_sep + '%05d_instance_%02d.png' %(starting_frame[0]+ii,t))
                    else:
                        toimage(mask2assess, cmin=0, cmax=255).save(base_dir_masks_sep + frame_names[ii] + '_instance_%02d.png' % (t))

                #end_saving_masks_time = time.time()
                #print("inference + saving masks time: %.3f" %(end_saving_masks_time - start_time))
                if args.dataset == 'youtube':
                    print(seq_name[0] + '/' + '%05d' % (starting_frame[0] + ii))
                else:
                    print(seq_name[0] + '/' + frame_names[ii])

                if args.overlay_masks:

                    frame_img = x.data.cpu().numpy()[0,:,:,:].squeeze()
                    frame_img = np.transpose(frame_img, (1,2,0))
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    frame_img = std * frame_img + mean
                    frame_img = np.clip(frame_img, 0, 1)
                    plt.figure();plt.axis('off')
                    plt.figure();plt.axis('off')
                    plt.imshow(frame_img)

                    for t in range(num_instances):

                        mask_pred = (torch.squeeze(outs[0,t,:])).cpu().numpy()
                        mask_pred = np.reshape(mask_pred, (height, width))

                        ax = plt.gca()
                        tmp_img = np.ones((mask_pred.shape[0], mask_pred.shape[1], 3))
                        color_mask = np.array(colors[t])/255.0
                        for i in range(3):
                            tmp_img[:,:,i] = color_mask[i]
                        ax.imshow(np.dstack( (tmp_img, mask_pred*0.7) ))

                    if args.dataset == 'youtube':
                        figname = base_dir + 'frame_%02d.png' %(starting_frame[0]+ii)
                    else:
                        figname = base_dir + frame_names[ii] + '.png'

                    plt.savefig(figname,bbox_inches='tight')
                    plt.close()


                if self.video_mode:
                    if args.only_spatial == False:
                        prev_hidden_temporal_list = hidden_temporal_list
                    if ii > 0:
                        prev_mask = outs
                    else:
                        prev_mask = y_mask

                del outs, hidden_temporal_list, x, y_mask, sw_mask

            #print('Mean Jaccard index(IOU) = %2f, mean F score = %2f' %(sumjaccard_vec),mean(F_vec)))
            print('Mean Jaccard index(IOU) = %2f' % (sum(jaccard_vec)/len(jaccard_vec)))
        
        Mean_jaccared = sum(jaccard_vec)/len(jaccard_vec)
        Mean_F = sum(F_vec)/len(F_vec)
        return Mean_jaccared, Mean_F                   

                        
def annot_from_mask(annot, instance_ids):        

    h = annot.shape[0]
    w = annot.shape[1]

    total_num_instances = len(instance_ids)
    max_instance_id = 0
    if total_num_instances > 0:
        max_instance_id = int(np.max(instance_ids))
    num_instances = max(args.maxseqlen,max_instance_id)

    gt_seg = np.zeros((num_instances, h*w))

    for i in range(total_num_instances):

        id_instance = int(instance_ids[i])
        aux_mask = np.zeros((h, w))
        aux_mask[annot==id_instance] = 1
        gt_seg[id_instance-1,:] = np.reshape(aux_mask,h*w)

    gt_seg = gt_seg[:][:args.maxseqlen]

    return gt_seg                        
                    

if __name__ == "__main__":
    
    model_name = '15_03_22-00_youtube_prev_mask'
    args = pickle.load(open(os.path.join('../models',model_name,'args.pkl'),'rb'))
   
    args.model_name = model_name
    args.dataset = 'davis2017' 
    args.eval_split = 'val' 
    args.batch_size = 1 
    args.length_clip = 130 
    args.num_workers = 0

    gpu_id = args.gpu_id
    if args.use_gpu:
        torch.cuda.set_device(device=gpu_id)
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    if not args.log_term:
        print ("Eval logs will be saved to:", os.path.join('../models',args.model_name, 'eval.log'))
        sys.stdout = open(os.path.join('../models',args.model_name, 'eval.log'), 'w')

    E = Evaluate(args)
    Mean_jaccared, Mean_F  = E.run_eval()

    print('Mean jaccared index(IOU)= %.2f, Mean F= %.2f' % (Mean_jaccared,Mean_F))