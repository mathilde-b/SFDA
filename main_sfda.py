#/usr/bin/env python3.6
import math
import re
import argparse
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from operator import itemgetter
from shutil import copytree, rmtree
import typing
from typing import Any, Callable, List, Tuple
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from dice3d import dice3d, dice3dn,dice3dn2, hd3dn
from networks import weights_init
from dataloader import get_loaders
from utils import map_, save_dict_to_file
from utils import dice_coef, dice_batch, save_images,save_images_p,save_be_images, tqdm_, save_images_ent
from utils import probs2one_hot, probs2class, mask_resize, resize, haussdorf
from utils import exp_lr_scheduler
import datetime
from itertools import cycle
import os
from time import sleep
from bounds import CheckBounds
import matplotlib.pyplot as plt
from itertools import chain
import platform



def setup(args, n_class, dtype) -> Tuple[Any, Any, Any, List[Callable], List[float],List[Callable], List[float], Callable]:
    print(">>> Setting up")
    cpu: bool = args.cpu or not torch.cuda.is_available()
    if cpu:
        print("WARNING CUDA NOT AVAILABLE")
    device = torch.device("cpu") if cpu else torch.device("cuda")
    n_epoch = args.n_epoch
    if args.model_weights:
        if cpu:
            net = torch.load(args.model_weights, map_location='cpu')
        else:
            net = torch.load(args.model_weights)
    else:
        net_class = getattr(__import__('networks'), args.network)
        net = net_class(1, n_class).type(dtype).to(device)
        net.apply(weights_init)
    net.to(device)
    if args.saveim:
        print("WARNING SAVING MASKS at each epc")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.l_rate, betas=(0.9, 0.999),weight_decay=args.weight_decay)
    if args.adamw:
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.l_rate, betas=(0.9, 0.999))

    print(args.target_losses)
    losses = eval(args.target_losses)
    loss_fns: List[Callable] = []
    for loss_name, loss_params, _, bounds_params, fn, _ in losses:
        loss_class = getattr(__import__('losses'), loss_name)
        loss_fns.append(loss_class(**loss_params, dtype=dtype, fn=fn))
        print("bounds_params", bounds_params)
        if bounds_params!=None:
            bool_predexist = CheckBounds(**bounds_params)
            print(bool_predexist,"size predictor")
            if not bool_predexist:
                n_epoch = 0

    loss_weights = map_(itemgetter(5), losses)

    if args.scheduler:
        scheduler = getattr(__import__('scheduler'), args.scheduler)(**eval(args.scheduler_params))
    else:
        scheduler = ''

    return net, optimizer, device, loss_fns, loss_weights, scheduler, n_epoch


def do_epoch(args, mode: str, net: Any, device: Any, epc: int,
             loss_fns: List[Callable], loss_weights: List[float],
              new_w:int, C: int, metric_axis:List[int], savedir: str = "",
             optimizer: Any = None, target_loader: Any = None, best_dice3d_val:Any=None):

    assert mode in ["train", "val"]
    L: int = len(loss_fns)
    indices = torch.tensor(metric_axis,device=device)
    if mode == "train":
        net.train()
        desc = f">> Training   ({epc})"
    elif mode == "val":
        net.eval()
        # net.train()
        desc = f">> Validation ({epc})"

    total_it_t, total_images_t = len(target_loader), len(target_loader.dataset)
    total_iteration = total_it_t
    #total_iteration = 2
    total_images = total_images_t

    if args.debug:
        total_iteration = 10
    pho=1
    dtype = eval(args.dtype)

    all_dices: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_sizes: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_sizes: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_gt_sizes: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_sizes2: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_inter_card: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_card_gt: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_card_pred: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_gt = []
    all_pred = []
    if args.do_hd: 
        #all_gt: Tensor = torch.zeros((total_images, 256, 256), dtype=dtype)
        all_gt: Tensor = torch.zeros((total_images, 384, 384), dtype=dtype)
        #all_pred: Tensor = torch.zeros((total_images, 256, 256), dtype=dtype)
        all_pred: Tensor = torch.zeros((total_images, 384, 384), dtype=dtype)
    loss_log: Tensor = torch.zeros((total_images), dtype=dtype, device=device)
    loss_cons: Tensor = torch.zeros((total_images), dtype=dtype, device=device)
    loss_se: Tensor = torch.zeros((total_images), dtype=dtype, device=device)
    loss_tot: Tensor = torch.zeros((total_images), dtype=dtype, device=device)
    posim_log: Tensor = torch.zeros((total_images), dtype=dtype, device=device)
    haussdorf_log: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    all_grp: Tensor = torch.zeros((total_images, C), dtype=dtype, device=device)
    #all_pnames = np.empty([total_images]).astype('U256') 
    all_pnames = np.zeros([total_images]).astype('U256') 
    dice_3d_log: Tensor = torch.zeros((1, C), dtype=dtype, device=device)
    dice_3d_sd_log: Tensor = torch.zeros((1, C), dtype=dtype, device=device)
    hd95_3d_log: Tensor = torch.zeros((1, C), dtype=dtype, device=device)
    hd95_3d_sd_log: Tensor = torch.zeros((1, C), dtype=dtype, device=device)
    tq_iter = tqdm_(enumerate(target_loader), total=total_iteration, desc=desc)
    done: int = 0
    #ratio_losses = 0
    n_warmup = args.n_warmup
    mult_lw = [pho ** (epc - n_warmup + 1)] * len(loss_weights)
    #if epc > 100:
    #    mult_lw = [pho ** 100] * len(loss_weights)
    mult_lw[0] = 1
    loss_weights = [a * b for a, b in zip(loss_weights, mult_lw)]
    losses_vec, source_vec, target_vec, baseline_target_vec = [], [], [], []
    pen_count = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        count_losses = 0
        for j, target_data in tq_iter:
            target_data[1:] = [e.to(device) for e in target_data[1:]]  # Move all tensors to device
            filenames_target, target_image, target_gt = target_data[:3]
            #print("target", filenames_target)
            labels = target_data[3:3+L]
            bounds = target_data[3+L:]
            filenames_target = [f.split('.nii')[0] for f in filenames_target]
            assert len(labels) == len(bounds), len(bounds)
            B = len(target_image)
            # Reset gradients
            if optimizer:
                #adjust_learning_rate(optimizer, 1, args.l_rate, args.power)
                optimizer.zero_grad()

            # Forward
            with torch.set_grad_enabled(mode == "train"):
                pred_logits: Tensor = net(target_image)
                pred_probs: Tensor = F.softmax(pred_logits, dim=1)
                if new_w > 0:
                    pred_probs = resize(pred_probs, new_w)
                    labels = [resize(label, new_w) for label in labels]
                    target = resize(target, new_w)
                predicted_mask: Tensor = probs2one_hot(pred_probs)  # Used only for dice computation
            assert len(bounds) == len(loss_fns) == len(loss_weights)
            if epc < n_warmup:
                loss_weights = [0]*len(loss_weights)
            loss: Tensor = torch.zeros(1, requires_grad=True).to(device)
            loss_vec = []
            loss_kw = []
            for loss_fn,label, w, bound in zip(loss_fns,labels, loss_weights, bounds):
                if w > 0:
                    if args.lin_aug_w:
                        if epc <13:
                            lamb_cons_pred = 0
                        else:
                            lamb_cons_pred = min((epc)/20,1)
                    elif args.both:
                        lamb_cons_pred = 1
                    else:
                        #lamb = 1 # only the pred
                        lamb_cons_pred = 0 # only the size prior
                    if eval(args.target_losses)[0][0]=="EntKLProp": 
                        loss_1, loss_cons_prior,est_prop =  loss_fn(pred_probs, label, bound)
                        loss = loss_1 + loss_cons_prior 
                    else:
                        loss =  loss_fn(pred_probs, label, bound)
                        loss = w*loss
                        loss_1 = loss
                        loss_2 = loss
                    #pen_count += count_b.detach()
                    #print(count_b.detach())
                    loss_kw.append(loss_1.detach())
                    loss_kw.append(loss_2.detach())
           # Backward
            if optimizer:
                loss.backward()
                optimizer.step()
            # Compute and log metrics
            #dices: Tensor = dice_coef(predicted_mask.detach(), target.detach())
            # baseline_dices: Tensor = dice_coef(labels[0].detach(), target.detach())
            #batch_dice: Tensor = dice_batch(predicted_mask.detach(), target.detach())
            # assert batch_dice.shape == (C,) and dices.shape == (B, C), (batch_dice.shape, dices.shape, B, C)
            #print(predicted_mask.shape,target_gt.shape)
            dices, inter_card, card_gt, card_pred = dice_coef(predicted_mask.detach(), target_gt.detach())
            assert dices.shape == (B, C), (dices.shape, B, C)
            #n_digits = 2
            #print(torch.round(dices * 10**n_digits) / (10**n_digits))
            sm_slice = slice(done, done + B)  # Values only for current batch
            all_dices[sm_slice, ...] = dices
            if eval(args.target_losses)[0][0] in ["EntKLProp","WeightedEntKLProp","EntKLProp2","CEKLProp2"]:
                all_sizes[sm_slice, ...] = torch.round(est_prop.detach()*target_image.shape[2]*target_image.shape[3])
            all_sizes2[sm_slice, ...] = torch.sum(predicted_mask,dim=(2,3)) 
            all_gt_sizes[sm_slice, ...] = torch.sum(target_gt,dim=(2,3)) 
            #all_sizes2[sm_slice, ...] = torch.sum(target_gt,dim=(2,3)) 
            # # for 3D dice
            #print(filenames_target)
            #print(torch.FloatTensor([int(re.split('_', x)[1]) for x in filenames_target]).unsqueeze(1).repeat(1,C))
            #print(filenames_target, "filenames target")
            if 'slice' in args.grp_regex:
                all_grp[sm_slice, ...] = torch.FloatTensor([int(re.split('_',re.split('slice',x)[1])[0]) for x in filenames_target]).unsqueeze(1).repeat(1,C)
                #all_grp[sm_slice, ...] = torch.FloatTensor([int(re.split('_',re.split('Subj',x)[1])[0]) for x in filenames_target]).unsqueeze(1).repeat(1,C)
            elif 'Case' in args.grp_regex:
                all_grp[sm_slice, ...] = torch.FloatTensor([int(re.split('_',re.split('Case',x)[1])[0]) for x in filenames_target]).unsqueeze(1).repeat(1,C)
            else:
                all_grp[sm_slice, ...] = int(re.split('_', filenames_target[0])[1]) * torch.ones([1, C])
            all_pnames[sm_slice] = filenames_target
            #print(all_pnames)
            all_inter_card[sm_slice, ...] = inter_card
            all_card_gt[sm_slice, ...] = card_gt
            all_card_pred[sm_slice, ...] = card_pred
            if args.do_hd:
                all_pred[sm_slice, ...] = probs2class(predicted_mask[:,:,:,:]).cpu().detach()
                all_gt[sm_slice, ...] = probs2class(target_gt).detach()
            #loss_log[sm_slice] = loss.detach()
            loss_se[sm_slice] = loss_kw[0]
            if len(loss_kw)>1:
            	loss_cons[sm_slice] = loss_kw[1]
            	loss_tot[sm_slice] = loss_kw[1]+loss_kw[0]
            else:
            	loss_cons[sm_slice] = 0
            	loss_tot[sm_slice] = loss_kw[0]
            #posim_log[sm_slice] = torch.einsum("bcwh->b", [target_gt[:, 1:, :, :]]).detach() > 0
            
            #haussdorf_res: Tensor = haussdorf(predicted_mask.detach(), target_gt.detach(), dtype)
            #assert haussdorf_res.shape == (B, C)
            #haussdorf_log[sm_slice] = haussdorf_res
            #print(filenames_source,loss_cons[sm_slice].detach().cpu().numpy(),loss_s[sm_slice].detach().cpu().numpy()) 
            # # Save images
            if savedir and args.saveim and mode =="val":
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    warnings.simplefilter("ignore") 
                    predicted_class: Tensor = probs2class(pred_probs)
                    save_images(predicted_class, filenames_target, savedir, mode, epc, False)
                    if args.entmap:
                        ent_map = torch.einsum("bcwh,bcwh->bwh", [-pred_probs, (pred_probs+1e-10).log()])
                        save_images_ent(ent_map, filenames_target, savedir,'ent_map', epc)

          
            # Logging
            big_slice = slice(0, done + B)  # Value for current and previous batches
            stat_dict = {**{f"DSC{n}": all_dices[big_slice, n].mean() for n in metric_axis},
                         **{f"SZ{n}": all_sizes[big_slice, n].mean() for n in metric_axis},
                         **({f"DSC_source{n}": all_dices_s[big_slice, n].mean() for n in metric_axis}
                           if args.source_metrics else {})}

            size_dict = {**{f"SZ{n}": all_sizes[big_slice, n].mean() for n in metric_axis}}
            #print(stat_dict)
            #stat_dict = {"dice": torch.index_select(all_dices, 1, indices).mean(),
            #             "loss": loss_log[big_slice].mean()}
            nice_dict = {k: f"{v:.4f}" for (k, v) in stat_dict.items()}
            #nice_dict2 = {k: f"{v:.0f}" for (k, v) in size_dict.items()}
            #nice_dict = dict(nice_dict.items() + nice_dict2.items())
            #nice_dict = dict(chain.from_iterable(d.iteritems() for d in (nice_dict,nice_dict2)))
            done += B
            tq_iter.set_postfix(nice_dict)
            #print(f"{desc} " + ', '.join(f"{k}={v}" for (k, v) in nice_dict.items()))
    #dice_posim = torch.masked_select(all_dices[:, -1], posim_log.type(dtype=torch.uint8)).mean()
    # dice3D gives back the 3d dice mai on images
    # if not args.debug:
    #    dice_3d_log_o, dice_3d_sd_log_o = dice3d(args.workdir, f"iter{epc:03d}", mode, "Subj_\\d+_",args.dataset + mode + '/CT_GT', C)
    
    if args.dice_3d and (mode == 'val'):
    #if args.dice_3d :
        #dice_3d_log, dice_3d_sd_log,hd95_3d_log, hd95_3d_sd_log=dice3dn(all_grp, all_inter_card, all_card_gt, all_card_pred,all_pred,all_gt,all_pnames,metric_axis,args.pprint,95)
        dice_3d_log, dice_3d_sd_log,hd95_3d_log, hd95_3d_sd_log = dice3dn2(all_grp, all_inter_card, all_card_gt, all_card_pred,all_pred,all_gt,all_pnames,metric_axis,args.pprint,args.do_hd, best_dice3d_val)
          
    dice_2d = torch.index_select(all_dices, 1, indices).mean().cpu().numpy()
    target_vec = [ dice_3d_log, dice_3d_sd_log,hd95_3d_log,hd95_3d_sd_log,dice_2d]
    #size_mean = torch.index_select(all_sizes, 1, indices).mean(dim=0).cpu().numpy()
    size_mean = torch.index_select(all_sizes2, 1, indices).mean(dim=0).cpu().numpy()
    size_gt_mean = torch.index_select(all_gt_sizes, 1, indices).mean(dim=0).cpu().numpy()
    #print(size_mean.mean())
    mask_pos = torch.index_select(all_sizes2, 1, indices)!=0
    gt_pos = torch.index_select(all_gt_sizes, 1, indices)!=0
    #mask_pos = torch.index_select(all_sizes, 1, indices)!=0
    size_mean_pos = torch.index_select(all_sizes2, 1, indices).sum(dim=0).cpu().numpy()/mask_pos.sum(dim=0).cpu().numpy()
    gt_size_mean_pos = torch.index_select(all_gt_sizes, 1, indices).sum(dim=0).cpu().numpy()/gt_pos.sum(dim=0).cpu().numpy()
    size_mean2 = torch.index_select(all_sizes2, 1, indices).mean(dim=0).cpu().numpy()
    #size_mean_pos = torch.index_select(all_sizes, 1, indices).sum(dim=0).cpu().numpy()/mask_pos.sum(dim=0).cpu().numpy()
    #print(size_mean_pos.mean())
    #print("epc:",epc,mode,"sz probs:",[np.int(s) for s in size_mean],np.int(size_mean.mean()),"sz mask:",[np.int(s) for s in size_mean2],np.int(size_mean2.mean()),"sz probs pos",[np.int(s) for s in size_mean_pos],np.int(size_mean_pos.mean()))
    losses_vec = [loss_se.mean().item(),loss_cons.mean().item(),loss_tot.mean().item(),np.int(size_mean.mean()),np.int(size_mean_pos.mean()),np.int(size_gt_mean.mean()),np.int(gt_size_mean_pos.mean())]
    if not epc%10:
        df_t = pd.DataFrame({
           "val_ids":all_pnames,
           "proposal_size":all_sizes2.cpu()})
        df_t.to_csv(Path(savedir,mode+str(epc)+"sizes.csv"), float_format="%.4f", index_label="epoch")
    return losses_vec, target_vec,source_vec



def run(args: argparse.Namespace) -> None:
    # save args to dict
    d = vars(args)
    d['time'] = str(datetime.datetime.now())
    d['server']=platform.node()
    save_dict_to_file(d,args.workdir)

    temperature: float = 0.1
    n_class: int = args.n_class
    metric_axis: List = args.metric_axis
    lr: float = args.l_rate
    dtype = eval(args.dtype)

    # Proper params
    savedir: str = args.workdir
    n_epoch: int = args.n_epoch

    net, optimizer, device, loss_fns, loss_weights, scheduler, n_epoch = setup(args, n_class, dtype)
    #print(f'> Loss weights cons: {loss_weights}, Loss weights source:{loss_weights_source}')
    #print("args.source_dataset: ",args.dataset,"args.target_dataset: ",args.target_dataset)
    shuffle = True
    #if args.mix:
    #    shuffle = True
    #print("args.dataset",args.dataset)
    '''
    loader, loader_val = get_loaders(args, args.dataset,args.source_folders,
                                           args.batch_size, n_class,
                                           args.debug, args.in_memory, dtype, shuffle, "source")
    '''
    #print("source loader loaded")
    print(args.target_folders)
    target_loader, target_loader_val = get_loaders(args, args.target_dataset,args.target_folders,
                                           args.batch_size, n_class,
                                           args.debug, args.in_memory, dtype, shuffle, "target", args.val_target_folders)

    #print("target loader loaded")
    print("metric axis",metric_axis)
    best_dice_pos: Tensor = np.zeros(1)
    best_dice: Tensor = np.zeros(1)
    best_hd3d_dice: Tensor = np.zeros(1)
    #best_3d_dice: Tensor = np.zeros(1)
    #best_3d_dice_source: Tensor = np.zeros(1)
    best_3d_dice: Tensor = 0 
    best_2d_dice: Tensor = 0 
    print("Results saved in ", savedir)
    print(">>> Starting the training")
    for i in range(n_epoch):

       if args.mode =="makeim":
            with torch.no_grad():
                 
                val_losses_vec, val_target_vec,val_source_vec                                        = do_epoch(args, "val", net, device,
                                                                                                i, loss_fns,
                                                                                               loss_weights,
                                                                                               args.resize,
                                                                                                n_class,metric_axis,
                                                                                               savedir=savedir,
                                                                                               target_loader=target_loader_val, best_dice3d_val=best_3d_dice)
                
                '''
                tra_losses_vec, tra_target_vec,tra_source_vec                                        = do_epoch(args, "val", net, device,
                                                                                               i, loss_fns,
                                                                                               loss_weights,
                                                                                               args.resize,
                                                                                               n_class,metric_axis,
                                                                                               savedir=savedir,
                                                                                               target_loader=target_loader)
                
                '''
                tra_losses_vec = val_losses_vec
                tra_target_vec = val_target_vec
                tra_source_vec = val_source_vec
       else:
            tra_losses_vec, tra_target_vec,tra_source_vec                                    = do_epoch(args, "train", net, device,
                                                                                           i, loss_fns,
                                                                                           loss_weights,
                                                                                           args.resize,
                                                                                           n_class, metric_axis,
                                                                                           savedir=savedir,
                                                                                           optimizer=optimizer,
                                                                                           target_loader=target_loader, best_dice3d_val=best_3d_dice)
       
            with torch.no_grad():
                val_losses_vec, val_target_vec,val_source_vec                                        = do_epoch(args, "val", net, device,
                                                                                               i, loss_fns,
                                                                                               loss_weights,
                                                                                               args.resize,
                                                                                               n_class,metric_axis,
                                                                                               savedir=savedir,
                                                                                               target_loader=target_loader_val, best_dice3d_val=best_3d_dice)
       '''
       print(val_target_vec[0])

       print("val_dice_3d",val_target_vec[0][metric_axis])
        #print("val_dice_3d", val_target_vec[0][metric_axis].mean()),
       
       
       print('val_target_vec[4]', val_target_vec[4])
       print('val_target_vec[2]', val_target_vec[2].shape)
       print(metric_axis)
       print('val_target_vec[2][metric_axis]',val_target_vec[2][metric_axis])
       print('val_target_vec[2][metric_axis].mean()',val_target_vec[2][metric_axis].mean())
        # Save model if better
       '''
       current_val_target_2d_dice = val_target_vec[4]
       current_val_target_3d_dice = val_target_vec[0]
       #print(current_val_target_3d_dice, "current_val_target_3d_dice") 
       '''
       if current_val_target_hd3d > best_hd3d_dice:
            best_epoch = i
            best_hd3d_dice = current_val_target_hd3d
            with open(Path(savedir, "best_epoch_3dhd.txt"), 'w') as f:
                f.write(str(i))
            best_folder_hd3d = Path(savedir, "best_epoch_3dhd")
            if best_folder_hd3d.exists():
                rmtree(best_folder_hd3d)
            copytree(Path(savedir, f"iter{i:03d}"), Path(best_folder_hd3d))
            torch.save(net, Path(savedir, "best_3dhd.pkl"))
        
       current_val_target_3d_dice = val_target_vec[0]
       
       if current_val_target_2d_dice > best_2d_dice:
            best_epoch = i
            best_2d_dice = current_val_target_2d_dice
            with open(Path(savedir, "best_epoch_2d.txt"), 'w') as f:
                f.write(str(i))
            best_folder_2d = Path(savedir, "best_epoch_2d")
            if best_folder_2d.exists():
                rmtree(best_folder_2d)
            if args.saveim:
                copytree(Path(savedir, f"iter{i:03d}"), Path(best_folder_2d))
            torch.save(net, Path(savedir, "best_2d.pkl"))
       ''' 
       #print(current_val_target_3d_dice)
       #print(best_3d_dice)
       if args.dice_3d:
           if current_val_target_3d_dice > best_3d_dice:
               best_epoch = i
               best_3d_dice = current_val_target_3d_dice
               with open(Path(savedir, "3dbestepoch.txt"), 'w') as f:
                   f.write(str(i)+','+str(best_3d_dice))
               best_folder_3d = Path(savedir, "best_epoch_3d")
               if best_folder_3d.exists():
                    rmtree(best_folder_3d)
               if args.saveim:
                    copytree(Path(savedir, f"iter{i:03d}"), Path(best_folder_3d))
           torch.save(net, Path(savedir, "best_3d.pkl"))

       
        #Save source model if better
       #current_val_source_3d_dice = val_source_vec[0]
       if not(i % 10) :
            print("epoch",str(i),savedir,'best 3d dice',best_3d_dice)
            torch.save(net, Path(savedir, "epoch_"+str(i)+".pkl"))
       #print("warning saving pkl every epoch")
       
       if i == n_epoch - 1:
            with open(Path(savedir, "last_epoch.txt"), 'w') as f:
                f.write(str(i))
            last_folder = Path(savedir, "last_epoch")
            if last_folder.exists():
                rmtree(last_folder)
            if args.saveim:
                copytree(Path(savedir, f"iter{i:03d}"), Path(last_folder))
            torch.save(net, Path(savedir, "last.pkl"))

        # remove images from iteration
       if args.saveim:
           rmtree(Path(savedir, f"iter{i:03d}"))

        #if i == 0:
         #   keep_tra_baseline_target_vec = tra_baseline_target_vec
          #  keep_val_baseline_target_vec = val_baseline_target_vec
        # print(keep_val_baseline_target_vec)

        # print(val_target_vec)
        # df_t_tmp = pd.DataFrame({
        #     "val_dice_3d": [val_target_vec[0]],
        #     "val_dice_3d_sd": [val_target_vec[1]]})
       if args.source_metrics:
            df_s_tmp = pd.DataFrame({
            #"tra_dice_3d": [tra_source_vec[0]],
            #"tra_dice_3d_sd": [tra_source_vec[1]],
            #"tra_dice_2d": [tra_source_vec[2]],
            "val_dice_3d": [val_source_vec[0]],
            "val_dice_3d_sd": [val_source_vec[1]],
            "val_dice_2d": [val_source_vec[2]]})
            if i == 0:
               df_s = df_s_tmp
            else:
                df_s = df_s.append(df_s_tmp)
            df_s.to_csv(Path(savedir, "_".join((args.source_folders.split("'")[1],"source", args.csv))), float_format="%.4f", index_label="epoch")

       #print(val_target_vec)
       df_t_tmp = pd.DataFrame({
            "epoch":i,
            "tra_loss_s":[tra_losses_vec[0]],
            "tra_loss_cons":[tra_losses_vec[1]],
            "tra_loss_tot":[tra_losses_vec[2]],
            "tra_size_mean":[tra_losses_vec[3]],
            "tra_size_mean_pos":[tra_losses_vec[4]],
            "val_loss_s":[val_losses_vec[0]],
            "val_loss_cons":[val_losses_vec[1]],
            "val_loss_tot":[val_losses_vec[2]],
            "val_size_mean":[val_losses_vec[3]],
            "val_size_mean_pos":[val_losses_vec[4]],
            "val_gt_size_mean":[val_losses_vec[5]],
            "val_gt_size_mean_pos":[val_losses_vec[6]],
            #"tra_dice_3d": [tra_target_vec[0]],
            #"tra_dice_3d_sd": [tra_target_vec[1][metric_axis]],
            #"tra_dice": [tra_target_vec[2][metric_axis].mean()],
            'tra_dice': [tra_target_vec[4]],
            'val_dice': [val_target_vec[4]],
            "val_dice_3d_sd": [val_target_vec[1]],
            "val_dice_3d": [val_target_vec[0]]})
            #"val_hd95_3d": [val_target_vec[2][metric_axis].mean()],

       if i == 0:
            df_t = df_t_tmp
       else:
            df_t = df_t.append(df_t_tmp)

       df_t.to_csv(Path(savedir, "_".join((args.target_folders.split("'")[1],"target", args.csv))), float_format="%.4f", index=False)

       if args.flr==False:
            #adjust_learning_rate(optimizer, i, args.l_rate, n_epoch, 0.9)
            exp_lr_scheduler(optimizer, i, args.lr_decay,args.lr_decay_epoch)
    print("Results saved in ", savedir, "best 3d dice",best_3d_dice)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--target_dataset', type=str, required=True)
    # parser.add_argument('--weak_subfolder', type=str, required=True)
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--target_losses", type=str, required=True,
                        help="List of (loss_name, loss_params, bounds_name, bounds_params, fn, weight)")
    parser.add_argument("--target_folders", type=str, required=True,
                        help="List of (subfolder, transform, is_hot)")
    parser.add_argument("--val_target_folders", type=str, required=True,
                        help="List of (subfolder, transform, is_hot)")
    parser.add_argument("--network", type=str, required=True, help="The network to use")
    parser.add_argument("--grp_regex", type=str, required=True)
    parser.add_argument("--n_class", type=int, required=True)
    parser.add_argument("--mode", type=str, default="learn")
    parser.add_argument("--lin_aug_w", action="store_true")
    parser.add_argument("--both", action="store_true")
    parser.add_argument("--trainval", action="store_true")
    parser.add_argument("--valonly", action="store_true")
    parser.add_argument("--flr", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--mix", type=bool, default=True)
    parser.add_argument("--do_hd", type=bool, default=False)
    parser.add_argument("--saveim", type=bool, default=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--csv", type=str, default='metrics.csv')
    parser.add_argument("--source_metrics", action="store_true")
    parser.add_argument("--adamw", action="store_true")
    parser.add_argument("--dice_3d", action="store_true")
    parser.add_argument("--ontest", action="store_true")
    parser.add_argument("--ontrain", action="store_true")
    parser.add_argument("--best_losses", action="store_true")
    parser.add_argument("--pprint", action="store_true")
    parser.add_argument("--ontrain1", action="store_true")
    parser.add_argument("--ontrain9_1", action="store_true")
    parser.add_argument("--ontrain19_1", action="store_true")
    parser.add_argument("--ontrain019_1", action="store_true")
    parser.add_argument("--ontrain1023_1", action="store_true")
    parser.add_argument("--ontrain1023", action="store_true")
    parser.add_argument("--ontrain1176_1353", action="store_true")
    parser.add_argument("--entmap", action="store_true")
    parser.add_argument("--model_weights", type=str, default='')
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--in_memory", action='store_true')
    parser.add_argument("--resize", type=int, default=0)
    # parser.add_argument("--weak", action="store_true")
    parser.add_argument("--pho", nargs='?', type=float, default=1,
                        help='augment')
    parser.add_argument("--n_warmup", type=int, default=0)
    parser.add_argument('--n_epoch', nargs='?', type=int, default=200,
                        help='# of the epochs')
    parser.add_argument('--l_rate', nargs='?', type=float, default=5e-4,
                        help='Learning Rate')
    parser.add_argument('--lr_decay', nargs='?', type=float, default=0.7),
    parser.add_argument('--lr_decay_epoch', nargs='?', type=float, default=20),
    parser.add_argument('--weight_decay', nargs='?', type=float, default=1e-5,
                        help='L2 regularisation of network weights')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--dtype", type=str, default="torch.float32")
    parser.add_argument("--scheduler", type=str, default="DummyScheduler")
    parser.add_argument("--scheduler_params", type=str, default="{}")
    parser.add_argument("--power",type=float, default=0.9)
    parser.add_argument("--metric_axis",type=int, nargs='*', required=True, help="Classes to display metrics. \
        Display only the average of everything if empty")
    args = parser.parse_args()
    print(args)

    return args


if __name__ == '__main__':

#    for i in range(0,15):

 #       args=argparse.Namespace(augment=False, batch_size=12, bounds_on_fgt=False, bounds_on_train_stats='', cpu=False, csv='metrics.csv', dataset='data/all_transverse', debug=False, dtype='torch.float32', flr=False, grp_regex='Subj_\\d+_\\d+', in_memory=False, l_rate=0.0005, lin_aug_w=False, metric_axis=[1], mix=True, model_weights='results/Inn/JCESource/best_3d.pkl', n_class=2, n_epoch=10, network='ENet', pho=1, power=0.9, resize=0, scheduler='DummyScheduler', scheduler_params='{}', source_folders="[('Inn', png_transform, False), ('GT', gt_transform, True),('WatonInn_pjce', gt_transform, True)]", source_losses="[('CrossEntropy', {'idc': [0,1], 'weights':[1,1]}, None, None, None, 1)]", source_metrics=False, target_dataset='data/all_transverse', target_folders="[('Inn', png_transform, False), ('GT', gt_transform, True),('WatonInn_pjce', gt_transform, True)]", target_losses="[('NaivePenalty', {'idc': [1],'power': 1},'PreciseBoundsOnWeakWTags', {'margin':0.1,'idc':[1], 'power': 1, 'mode':'percentage'},'soft_size',80)]", weight_decay=0.0001, workdir='results/Inn/PWSizeLossNs_jce/')
    # args = argparse.Namespace(batch_size=4,cpu=False, csv='metrics.csv', dataset='data/all_transverse',
	# 	    target_dataset='data/all_transverse', mix=True, metric_axis=[1], augment=False,
    #                           debug=False, dtype='torch.float32', power=0.9,lin_aug_w=False,
    #                           bounds_on_fgt=False, bounds_on_train_stats='',
    #                           folders="[('Wat', png_transform, False), ('GT', gt_transform, False),"
    #                                   "('GT', gt_transform, False)]",flr=False,
    #                           target_folders="[('Inn', png_transform, False), ('GT', gt_transform, False)]+"
    #                                          "[('GT', gt_transform, False),('GT', gt_transform, False),('GT', gt_transform, False)]",
    #                           grp_regex='Subj_\\d+_\\d+', in_memory=False, l_rate=0.0005, weight_decay=1e-4,
    #                           losses="[('NaivePenalty', {'idc': [1]},'PredictionBoundswTags', "
    #                           " {'margin':0.1,'idc':[1], 'mode':'percentage','net': 'results/ls_winr2/pred_size40.pkl'} , 'soft_size',1),('SelfEntropy', {'idc': [0,1], 'weights':[1,1]}, None, None, None,0 ),"
    #                             " ('CEProp', {'fgt':True, 'power': 3}, 'PredictionValues',{'margin':0.1,'mode':'percentage','idc':[1],'sizefile':'results/trainval_size_Inn/ls_winr2_40/trainvalreg_metrics_C2.csv'}, 'norm_soft_size', 1)]",
    #                           losses_source="[('CrossEntropy', {'idc': [0,1], 'weights':[1,1]}, None, None, None, 1)]",
    #                           model_weights='results/all_transverse/fse/best_3d.pkl', n_class=2, n_epoch=150, network='ENet', pho=1.0, resize=0,
	# 		      scheduler='DummyScheduler', scheduler_params='{}', workdir='results/Inn/foo')
    #
    run(get_args())
    #    run(args)
