import numpy as np
import random
import os
import torch
# import pickle
import time
from collections import defaultdict
from dataset_seq import *
#from dataset import *
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import argparse
import torch.nn.functional as F
#from model import *
from model_seq import *
from sklearn.metrics import roc_auc_score
from pathlib import Path
# from pypai.model import upload_model
from tqdm import tqdm
from functools import partial
import logging
from utils import *
# from thop import profile
from sklearn.model_selection import train_test_split  # 划分数据集

logger = logging.getLogger()

def test(model,args,valLoader):
    model.eval()
    stats = AverageMeter('loss','loss_cls')
    # stats = AverageMeter('loss','ndcg_1_d1','ndcg_5_d1','ndcg_10_d1','ndcg_1_d2','ndcg_5_d2','ndcg_10_d2','hit_1_d1','hit_5_d1','hit_10_d1','hit_1_d2','hit_5_d2','hit_10_d2','MRR_d1','MRR_d2')
    pred_d1_list = None
    pred_d2_list = None
    pred_d1_list_ov = None
    pred_d1_list_no = None
    pred_d2_list_ov = None
    pred_d2_list_no = None
    criterion_cls = nn.BCELoss(reduce=False)
    fix_value = 1e-7 # fix the same value 
    for k,sample in enumerate(tqdm(valLoader)):
        u_node = torch.LongTensor(sample['user_node'].long()).cuda()
        i_node = torch.LongTensor(sample['i_node'].long()).cuda()
        neg_samples = torch.LongTensor(sample['neg_samples'].long()).cuda()
        seq_d1 = torch.LongTensor(sample['seq_d1'].long()).cuda()
        seq_d2 = torch.LongTensor(sample['seq_d2'].long()).cuda()
        long_tail_mask_d1 = torch.LongTensor(sample['long_tail_mask_d1'].long()).cuda()
        long_tail_mask_d2 = torch.LongTensor(sample['long_tail_mask_d2'].long()).cuda()
        domain_id = torch.LongTensor(sample['domain_id'].long()).cuda()
        overlap_label = torch.LongTensor(sample['overlap_label'].long()).cuda()
        labels = torch.LongTensor(sample['label'].long()).cuda()
        labels = labels.float()
        with torch.no_grad():
            predict_d1, predict_d2, predict_ips_d1, predict_ips_d2, predict_gfunc_d1, predict_gfunc_d2 = model(u_node,i_node,neg_samples,seq_d1,seq_d2,long_tail_mask_d1,long_tail_mask_d2)
            # predict_d1, predict_d2,u_feat_enhance_m1_d1, u_feat_enhance_m1_d2, u_feat_enhance_m2_d1,u_feat_enhance_m2_d2, u_feat_enhance_m3_d1,u_feat_enhance_m3_d2, u_feat_enhance_m4_d1, u_feat_enhance_m4_d2 = model(u_node,i_node,neg_samples,seq_d1,seq_d2,long_tail_mask_d1,long_tail_mask_d2)
        predict_d1 = predict_d1.squeeze()
        predict_d2 = predict_d2.squeeze()
        one_value = torch.LongTensor(torch.ones(domain_id.shape[0]).long()).cuda()
        mask_d1 = torch.LongTensor((one_value.cpu() - domain_id.cpu()).long()).cuda()
        mask_d2 = torch.LongTensor((domain_id.cpu()).long()).cuda()
        loss_cls = criterion_cls(predict_d1,labels) * mask_d1.unsqueeze(1) + criterion_cls(predict_d2,labels) * mask_d2.unsqueeze(1)
        loss_cls = torch.mean(loss_cls)
        # label_domain = torch.LongTensor([0,1,0,1]).cuda().float()
        # loss_cl = nn.BCELoss()(predict_domain.squeeze(),label_domain)
        # loss_cl =  cal_loss_cl_refine(u_feat_enhance_m1_d1,u_feat_enhance_m4_d1)+cal_loss_cl_refine(u_feat_enhance_m1_d2,u_feat_enhance_m4_d2)
        loss = loss_cls #+ loss_cl * 0.05
        stats.update(loss=loss.item(),loss_cls=loss_cls.item())#,loss_cl=loss_cl.item())
        domain_id = domain_id.unsqueeze(1).expand_as(predict_d1)
        overlap_label = overlap_label.unsqueeze(1).expand_as(predict_d1)
        predict_d1 = predict_d1.view(-1,args.neg_nums+1).cpu().detach().numpy().copy()
        predict_d2 = predict_d2.view(-1,args.neg_nums+1).cpu().detach().numpy().copy()
        domain_id = domain_id.view(-1,args.neg_nums+1).cpu().detach().numpy().copy()
        if not args.overlap:
            predict_d1_cse, predict_d2_cse = choose_predict(predict_d1,predict_d2,domain_id)
            if pred_d1_list is None and not isinstance(predict_d1_cse,list):
                pred_d1_list = predict_d1_cse
            elif pred_d1_list is not None and not isinstance(predict_d1_cse,list):
                pred_d1_list = np.append(pred_d1_list, predict_d1_cse, axis=0)
            if pred_d2_list is None and not isinstance(predict_d2_cse,list):
                pred_d2_list = predict_d2_cse
            elif pred_d2_list is not None and not isinstance(predict_d2_cse,list):
                pred_d2_list = np.append(pred_d2_list, predict_d2_cse, axis=0)
        else:
            overlap_label = overlap_label.view(-1,args.neg_nums+1).cpu().detach().numpy().copy()
            predict_d1_cse_over, predict_d1_cse_nono, predict_d2_cse_over, predict_d2_cse_nono = choose_predict_overlap(predict_d1,predict_d2,domain_id,overlap_label)
            if pred_d1_list_ov is None and not isinstance(predict_d1_cse_over,list):
                pred_d1_list_ov = predict_d1_cse_over
            elif pred_d1_list_ov is not None and not isinstance(predict_d1_cse_over,list):
                pred_d1_list_ov = np.append(pred_d1_list_ov, predict_d1_cse_over, axis=0)
            if pred_d1_list_no is None and not isinstance(predict_d1_cse_nono,list):
                pred_d1_list_no = predict_d1_cse_nono
            elif pred_d1_list_no is not None and not isinstance(predict_d1_cse_nono,list):
                pred_d1_list_no = np.append(pred_d1_list_no, predict_d1_cse_nono, axis=0)
            if pred_d2_list_ov is None and not isinstance(predict_d2_cse_over,list):
                pred_d2_list_ov = predict_d2_cse_over
            elif pred_d2_list_ov is not None and not isinstance(predict_d2_cse_over,list):
                pred_d2_list_ov = np.append(pred_d2_list_ov, predict_d2_cse_over, axis=0)
            if pred_d2_list_no is None and not isinstance(predict_d2_cse_nono,list):
                pred_d2_list_no = predict_d2_cse_nono
            elif pred_d2_list_no is not None and not isinstance(predict_d2_cse_nono,list):
                pred_d2_list_no = np.append(pred_d2_list_no, predict_d2_cse_nono, axis=0)
            predict_d1_cse, predict_d2_cse = choose_predict(predict_d1,predict_d2,domain_id)
            if pred_d1_list is None and not isinstance(predict_d1_cse,list):
                pred_d1_list = predict_d1_cse
            elif pred_d1_list is not None and not isinstance(predict_d1_cse,list):
                pred_d1_list = np.append(pred_d1_list, predict_d1_cse, axis=0)
            if pred_d2_list is None and not isinstance(predict_d2_cse,list):
                pred_d2_list = predict_d2_cse
            elif pred_d2_list is not None and not isinstance(predict_d2_cse,list):
                pred_d2_list = np.append(pred_d2_list, predict_d2_cse, axis=0)
    if not args.overlap:        
        pred_d1_list[:,0] = pred_d1_list[:,0]-fix_value
        pred_d2_list[:,0] = pred_d2_list[:,0]-fix_value
        HIT_1_d1, NDCG_1_d1, HIT_5_d1, NDCG_5_d1, HIT_10_d1, NDCG_10_d1, MRR_d1 = get_sample_scores(pred_d1_list)
        HIT_1_d2, NDCG_1_d2, HIT_5_d2, NDCG_5_d2, HIT_10_d2, NDCG_10_d2, MRR_d2 = get_sample_scores(pred_d2_list)
        return stats.loss, stats.loss_cls, HIT_1_d1, NDCG_1_d1, HIT_5_d1, NDCG_5_d1, HIT_10_d1, NDCG_10_d1, MRR_d1, HIT_1_d2, NDCG_1_d2, HIT_5_d2, NDCG_5_d2, HIT_10_d2, NDCG_10_d2, MRR_d2
    else:
        HIT_1_d1_ov, NDCG_1_d1_ov, HIT_5_d1_ov, NDCG_5_d1_ov, HIT_10_d1_ov, NDCG_10_d1_ov, MRR_d1_ov = get_sample_scores(pred_d1_list_ov)
        HIT_1_d1_no, NDCG_1_d1_no, HIT_5_d1_no, NDCG_5_d1_no, HIT_10_d1_no, NDCG_10_d1_no, MRR_d1_no = get_sample_scores(pred_d1_list_no)
        HIT_1_d2_ov, NDCG_1_d2_ov, HIT_5_d2_ov, NDCG_5_d2_ov, HIT_10_d2_ov, NDCG_10_d2_ov, MRR_d2_ov = get_sample_scores(pred_d2_list_ov)
        HIT_1_d2_no, NDCG_1_d2_no, HIT_5_d2_no, NDCG_5_d2_no, HIT_10_d2_no, NDCG_10_d2_no, MRR_d2_no = get_sample_scores(pred_d2_list_no)
        pred_d1_list[:,0] = pred_d1_list[:,0]-fix_value
        pred_d2_list[:,0] = pred_d2_list[:,0]-fix_value
        HIT_1_d1, NDCG_1_d1, HIT_5_d1, NDCG_5_d1, HIT_10_d1, NDCG_10_d1, MRR_d1 = get_sample_scores(pred_d1_list)
        HIT_1_d2, NDCG_1_d2, HIT_5_d2, NDCG_5_d2, HIT_10_d2, NDCG_10_d2, MRR_d2 = get_sample_scores(pred_d2_list)
        return stats.loss, stats.loss_cls, HIT_1_d1_ov, NDCG_1_d1_ov, HIT_5_d1_ov, NDCG_5_d1_ov, HIT_10_d1_ov, NDCG_10_d1_ov, MRR_d1_ov, HIT_1_d1_no, NDCG_1_d1_no, HIT_5_d1_no, NDCG_5_d1_no, HIT_10_d1_no, NDCG_10_d1_no, MRR_d1_no, HIT_1_d2_ov, NDCG_1_d2_ov, HIT_5_d2_ov, NDCG_5_d2_ov, HIT_10_d2_ov, NDCG_10_d2_ov, MRR_d2_ov, HIT_1_d2_no, NDCG_1_d2_no, HIT_5_d2_no, NDCG_5_d2_no, HIT_10_d2_no, NDCG_10_d2_no, MRR_d2_no, HIT_1_d1, NDCG_1_d1, HIT_5_d1, NDCG_5_d1, HIT_10_d1, NDCG_10_d1, MRR_d1, HIT_1_d2, NDCG_1_d2, HIT_5_d2, NDCG_5_d2, HIT_10_d2, NDCG_10_d2, MRR_d2

def train(model,trainLoader,trainLoaderDR,args,valLoader):
    best_hit_1_d1 = 0
    best_hit_5_d1 = 0
    best_hit_10_d1 = 0
    best_hit_1_d2 = 0
    best_hit_5_d2 = 0
    best_hit_10_d2 = 0

    best_ndcg_1_d1 = 0
    best_ndcg_5_d1 = 0
    best_ndcg_10_d1 = 0
    best_ndcg_1_d2 = 0
    best_ndcg_5_d2 = 0
    best_ndcg_10_d2 = 0

    best_mrr_d1 = 0
    best_mrr_d2 = 0

    best_hit_1_d1_ov = 0
    best_hit_5_d1_ov = 0
    best_hit_10_d1_ov = 0
    best_hit_1_d2_ov = 0
    best_hit_5_d2_ov = 0
    best_hit_10_d2_ov = 0

    best_ndcg_1_d1_ov = 0
    best_ndcg_5_d1_ov = 0
    best_ndcg_10_d1_ov = 0
    best_ndcg_1_d2_ov = 0
    best_ndcg_5_d2_ov = 0
    best_ndcg_10_d2_ov = 0

    best_mrr_d1_ov = 0
    best_mrr_d2_ov = 0

    best_hit_1_d1_no = 0
    best_hit_5_d1_no = 0
    best_hit_10_d1_no = 0
    best_hit_1_d2_no = 0
    best_hit_5_d2_no = 0
    best_hit_10_d2_no = 0

    best_ndcg_1_d1_no = 0
    best_ndcg_5_d1_no = 0
    best_ndcg_10_d1_no = 0
    best_ndcg_1_d2_no = 0
    best_ndcg_5_d2_no = 0
    best_ndcg_10_d2_no = 0

    best_mrr_d1_no = 0
    best_mrr_d2_no = 0
    save_path1 = Path(args.model_dir) / 'checkpoint' / 'best_d1.pt'
    save_path2 = Path(args.model_dir) / 'checkpoint' / 'best_d2.pt'
    criterion_recon = partial(sce_loss, alpha=args.alpha_l)
    criterion_cls = nn.BCELoss(reduce=False)
    if not os.path.exists(os.path.join(Path(args.model_dir),'checkpoint')):
        os.mkdir(os.path.join(Path(args.model_dir),'checkpoint'))
    for epoch in range(args.epoch):
        stats = AverageMeter('loss_cls','loss_dr_e','loss_dr_r')
        model.train()

        for i,sample in enumerate(tqdm(trainLoader)):
            u_node = torch.LongTensor(sample['user_node'].long()).cuda()
            i_node = torch.LongTensor(sample['i_node'].long()).cuda()
            neg_samples = torch.LongTensor(sample['neg_samples'].long()).cuda()
            seq_d1 = torch.LongTensor(sample['seq_d1'].long()).cuda()
            seq_d2 = torch.LongTensor(sample['seq_d2'].long()).cuda()
            long_tail_mask_d1 = torch.LongTensor(sample['long_tail_mask_d1'].long()).cuda()
            long_tail_mask_d2 = torch.LongTensor(sample['long_tail_mask_d2'].long()).cuda()
            domain_id = torch.LongTensor(sample['domain_id'].long()).cuda()
            labels = torch.LongTensor(sample['label'].long()).cuda()
            labels = labels.float()
            # flops, params = profile(model, (u_node,i_node,neg_samples,seq_d1,seq_d2,long_tail_mask_d1,long_tail_mask_d2,))
            # print('flops: ', flops, 'params: ', params)
            # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
            predict_d1, predict_d2, predict_ips_d1, predict_ips_d2, predict_gfunc_d1, predict_gfunc_d2 = model(u_node,i_node,neg_samples,seq_d1,seq_d2,long_tail_mask_d1,long_tail_mask_d2)
            predict_d1 = predict_d1.squeeze()
            predict_d2 = predict_d2.squeeze()
            predict_gfunc_d1 = predict_gfunc_d1.squeeze()
            predict_gfunc_d2 = predict_gfunc_d2.squeeze()
            predict_ips_d1 = predict_ips_d1.squeeze()
            predict_ips_d2 = predict_ips_d2.squeeze()
            one_value = torch.LongTensor(torch.ones(domain_id.shape[0]).long()).cuda()
            mask_d1 = torch.LongTensor((one_value.cpu() - domain_id.cpu()).long()).cuda()
            mask_d2 = torch.LongTensor((domain_id.cpu()).long()).cuda()
            # print(predict_d1.shape)
            # print(labels.shape)
            loss_cls = criterion_cls(predict_d1,labels) * mask_d1.unsqueeze(1)  + criterion_cls(predict_d2,labels) * mask_d2.unsqueeze(1) #* 2
            loss_cls = torch.mean(loss_cls)
            loss_dr_e = (criterion_cls(predict_d1,labels)-predict_gfunc_d1)**2/predict_ips_d1 * mask_d1.unsqueeze(1) + (criterion_cls(predict_d2,labels)-predict_gfunc_d2)**2/predict_ips_d2 * mask_d2.unsqueeze(1)
            loss_dr_e = torch.mean(loss_dr_e)
            loss = loss_cls + loss_dr_e * args.dr_e_w # test
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # auc_avg_domain = roc_auc_score(labels.detach().cpu().numpy(),predict.detach().cpu().numpy())
            stats.update(loss_cls=loss_cls.item(),loss_dr_e=loss_dr_e.item())
            if i % 20 == 0:
                logger.info(f'train cls loss:{stats.loss_cls}, dr_e loss:{stats.loss_dr_e} \t')

        if args.overlap:
            val_loss, val_cls_loss, HIT_1_d1_ov, NDCG_1_d1_ov, HIT_5_d1_ov, NDCG_5_d1_ov, HIT_10_d1_ov, NDCG_10_d1_ov, MRR_d1_ov, HIT_1_d1_no, NDCG_1_d1_no, HIT_5_d1_no, NDCG_5_d1_no, HIT_10_d1_no, NDCG_10_d1_no, MRR_d1_no, HIT_1_d2_ov, NDCG_1_d2_ov, HIT_5_d2_ov, NDCG_5_d2_ov, HIT_10_d2_ov, NDCG_10_d2_ov, MRR_d2_ov, HIT_1_d2_no, NDCG_1_d2_no, HIT_5_d2_no, NDCG_5_d2_no, HIT_10_d2_no, NDCG_10_d2_no, MRR_d2_no, HIT_1_d1, NDCG_1_d1, HIT_5_d1, NDCG_5_d1, HIT_10_d1, NDCG_10_d1, MRR_d1, HIT_1_d2, NDCG_1_d2, HIT_5_d2, NDCG_5_d2, HIT_10_d2, NDCG_10_d2, MRR_d2  = test(model,args,valLoader)
            best_hit_1_d1_ov = max(HIT_1_d1_ov,best_hit_1_d1_ov)
            best_hit_5_d1_ov = max(HIT_5_d1_ov,best_hit_5_d1_ov)
            best_hit_10_d1_ov = max(HIT_10_d1_ov,best_hit_10_d1_ov)
            best_hit_1_d2_ov = max(HIT_1_d2_ov,best_hit_1_d2_ov)
            best_hit_5_d2_ov = max(HIT_5_d2_ov,best_hit_5_d2_ov)
            best_hit_10_d2_ov = max(HIT_10_d2_ov,best_hit_10_d2_ov)

            best_ndcg_1_d1_ov = max(best_ndcg_1_d1_ov,NDCG_1_d1_ov)
            best_ndcg_5_d1_ov = max(best_ndcg_5_d1_ov,NDCG_5_d1_ov)
            best_ndcg_10_d1_ov = max(best_ndcg_10_d1_ov,NDCG_10_d1_ov)
            best_ndcg_1_d2_ov = max(best_ndcg_1_d2_ov,NDCG_1_d2_ov)
            best_ndcg_5_d2_ov = max(best_ndcg_5_d2_ov,NDCG_5_d2_ov)
            best_ndcg_10_d2_ov = max(best_ndcg_10_d2_ov,NDCG_10_d2_ov)
            best_mrr_d1_ov = max(best_mrr_d1_ov,MRR_d1_ov)
            best_mrr_d2_ov = max(best_mrr_d2_ov,MRR_d2_ov)   

            best_hit_1_d1_no = max(HIT_1_d1_no,best_hit_1_d1_no)
            best_hit_5_d1_no = max(HIT_5_d1_no,best_hit_5_d1_no)
            best_hit_10_d1_no = max(HIT_10_d1_no,best_hit_10_d1_no)
            best_hit_1_d2_no = max(HIT_1_d2_no,best_hit_1_d2_no)
            best_hit_5_d2_no = max(HIT_5_d2_no,best_hit_5_d2_no)
            best_hit_10_d2_no = max(HIT_10_d2_no,best_hit_10_d2_no)

            best_ndcg_1_d1_no = max(best_ndcg_1_d1_no,NDCG_1_d1_no)
            best_ndcg_5_d1_no = max(best_ndcg_5_d1_no,NDCG_5_d1_no)
            best_ndcg_10_d1_no = max(best_ndcg_10_d1_no,NDCG_10_d1_no)
            best_ndcg_1_d2_no = max(best_ndcg_1_d2_no,NDCG_1_d2_no)
            best_ndcg_5_d2_no = max(best_ndcg_5_d2_no,NDCG_5_d2_no)
            best_ndcg_10_d2_no = max(best_ndcg_10_d2_no,NDCG_10_d2_no)
            best_mrr_d1_no = max(best_mrr_d1_no,MRR_d1_no)
            best_mrr_d2_no = max(best_mrr_d2_no,MRR_d2_no)       

            best_hit_1_d1 = max(HIT_1_d1,best_hit_1_d1)
            best_hit_5_d1 = max(HIT_5_d1,best_hit_5_d1)
            best_hit_10_d1 = max(HIT_10_d1,best_hit_10_d1)
            best_hit_1_d2 = max(HIT_1_d2,best_hit_1_d2)
            best_hit_5_d2 = max(HIT_5_d2,best_hit_5_d2)
            best_hit_10_d2 = max(HIT_10_d2,best_hit_10_d2)

            best_ndcg_1_d1 = max(best_ndcg_1_d1,NDCG_1_d1)
            best_ndcg_5_d1 = max(best_ndcg_5_d1,NDCG_5_d1)
            best_ndcg_10_d1 = max(best_ndcg_10_d1,NDCG_10_d1)
            best_ndcg_1_d2 = max(best_ndcg_1_d2,NDCG_1_d2)
            best_ndcg_5_d2 = max(best_ndcg_5_d2,NDCG_5_d2)
            best_ndcg_10_d2 = max(best_ndcg_10_d2,NDCG_10_d2)
            best_mrr_d1 = max(best_mrr_d1,MRR_d1)
            best_mrr_d2 = max(best_mrr_d2,MRR_d2)
            logger.info(f'Epoch: {epoch}/{args.epoch} \t'
                        f'Train Loss: {stats.loss_cls:.4f} dr_e Loss: {stats.loss_dr_e:.4f} \t'
                        f'Val loss: {val_loss:.4f}, cls loss: {val_cls_loss:.4f}\n'
                        f'val domain1 cur/max overlap HR@1: {HIT_1_d1_ov:.4f}/{best_hit_1_d1_ov:.4f} \n,' 
                        f'overlap HR@5: {HIT_5_d1_ov:.4f}/{best_hit_5_d1_ov:.4f} \n, '
                        f'overlap HR@10: {HIT_10_d1_ov:.4f}/{best_hit_10_d1_ov:.4f} \n'
                        f'overlap NDCG@5: {NDCG_5_d1_ov:.4f}/{best_ndcg_5_d1_ov:.4f} \n, '
                        f'overlap NDCG@10: {NDCG_10_d1_ov:.4f}/{best_ndcg_10_d1_ov:.4f}, \n'
                        f'overlap MRR: {MRR_d1_ov:.4f}/{best_mrr_d1_ov:.4f} \n'
                        f'val domain1 cur/max non-overlap HR@1: {HIT_1_d1_no:.4f}/{best_hit_1_d1_no:.4f} \n,' 
                        f'non-overlap HR@5: {HIT_5_d1_no:.4f}/{best_hit_5_d1_no:.4f} \n, '
                        f'non-overlap HR@10: {HIT_10_d1_no:.4f}/{best_hit_10_d1_no:.4f} \n'
                        f'non-overlap NDCG@5: {NDCG_5_d1_no:.4f}/{best_ndcg_5_d1_no:.4f} \n, '
                        f'non-overlap NDCG@10: {NDCG_10_d1_no:.4f}/{best_ndcg_10_d1_no:.4f}, \n'
                        f'non-overlap MRR: {MRR_d1_no:.4f}/{best_mrr_d1_no:.4f} \n'
                        f'val domain2 cur/max overlap HR@1: {HIT_1_d2_ov:.4f}/{best_hit_1_d2_ov:.4f} \n, '
                        f'overlap HR@5: {HIT_5_d2_ov:.4f}/{best_hit_5_d2_ov:.4f} \n, '
                        f'overlap HR@10: {HIT_10_d2_ov:.4f}/{best_hit_10_d2_ov:.4f} \n'
                        f'overlap NDCG@5: {NDCG_5_d2_ov:.4f}/{best_ndcg_5_d2_ov:.4f} \n, '
                        f'overlap NDCG@10: {NDCG_10_d2_ov:.4f}/{best_ndcg_10_d2_ov:.4f}, \n'
                        f'overlap MRR: {MRR_d2_ov:.4f}/{best_mrr_d2_ov:.4f} \n'
                        f'val domain2 cur/max non-overlap HR@1: {HIT_1_d2_no:.4f}/{best_hit_1_d2_no:.4f} \n, '
                        f'non-overlap HR@5: {HIT_5_d2_no:.4f}/{best_hit_5_d2_no:.4f} \n, '
                        f'non-overlap HR@10: {HIT_10_d2_no:.4f}/{best_hit_10_d2_no:.4f} \n'
                        f'non-overlap NDCG@5: {NDCG_5_d2_no:.4f}/{best_ndcg_5_d2_no:.4f} \n, '
                        f'non-overlap NDCG@10: {NDCG_10_d2_no:.4f}/{best_ndcg_10_d2_no:.4f}, \n'
                        f'non-overlap MRR: {MRR_d2_no:.4f}/{best_mrr_d2_no:.4f} \n'
                        f'val domain1 cur/max HR@1: {HIT_1_d1:.4f}/{best_hit_1_d1:.4f} \n,' 
                        f'HR@5: {HIT_5_d1:.4f}/{best_hit_5_d1:.4f} \n, '
                        f'HR@10: {HIT_10_d1:.4f}/{best_hit_10_d1:.4f} \n'
                        # f'val domain1 cur/max NDCG@1: {NDCG_1_d1:.4f}/{best_ndcg_1_d1:.4f} \n, '
                        f'NDCG@5: {NDCG_5_d1:.4f}/{best_ndcg_5_d1:.4f} \n, '
                        f'NDCG@10: {NDCG_10_d1:.4f}/{best_ndcg_10_d1:.4f}, \n'
                        f'val domain1 cur/max MRR: {MRR_d1:.4f}/{best_mrr_d1:.4f} \n'
                        f'val domain2 cur/max HR@1: {HIT_1_d2:.4f}/{best_hit_1_d2:.4f} \n, '
                        f'HR@5: {HIT_5_d2:.4f}/{best_hit_5_d2:.4f} \n, '
                        f'HR@10: {HIT_10_d2:.4f}/{best_hit_10_d2:.4f} \n'
                        # f'val domain2 cur/max NDCG@1: {NDCG_1_d2:.4f}/{best_ndcg_1_d2:.4f} \n, '
                        f'NDCG@5: {NDCG_5_d2:.4f}/{best_ndcg_5_d2:.4f} \n, '
                        f'NDCG@10: {NDCG_10_d2:.4f}/{best_ndcg_10_d2:.4f}, \n'
                        f'val domain2 cur/max MRR: {MRR_d2:.4f}/{best_mrr_d2:.4f} \n')
        else:
            val_loss, val_cls_loss, HIT_1_d1, NDCG_1_d1, HIT_5_d1, NDCG_5_d1, HIT_10_d1, NDCG_10_d1, MRR_d1, HIT_1_d2, NDCG_1_d2, HIT_5_d2, NDCG_5_d2, HIT_10_d2, NDCG_10_d2, MRR_d2 = test(model,args,valLoader)
            best_hit_1_d1 = max(HIT_1_d1,best_hit_1_d1)
            best_hit_5_d1 = max(HIT_5_d1,best_hit_5_d1)
            best_hit_10_d1 = max(HIT_10_d1,best_hit_10_d1)
            best_hit_1_d2 = max(HIT_1_d2,best_hit_1_d2)
            best_hit_5_d2 = max(HIT_5_d2,best_hit_5_d2)
            best_hit_10_d2 = max(HIT_10_d2,best_hit_10_d2)

            best_ndcg_1_d1 = max(best_ndcg_1_d1,NDCG_1_d1)
            best_ndcg_5_d1 = max(best_ndcg_5_d1,NDCG_5_d1)
            best_ndcg_10_d1 = max(best_ndcg_10_d1,NDCG_10_d1)
            best_ndcg_1_d2 = max(best_ndcg_1_d2,NDCG_1_d2)
            best_ndcg_5_d2 = max(best_ndcg_5_d2,NDCG_5_d2)
            best_ndcg_10_d2 = max(best_ndcg_10_d2,NDCG_10_d2)
            # if MRR_d1 >= best_mrr_d1:
            #     best_auc1 = auc_testd1
                # torch.save(model.state_dict(), str(save_path1))
            # if MRR_d2 >= best_mrr_d2:
            #     best_auc2 = auc_testd2
                # torch.save(model.state_dict(), str(save_path2))
            best_mrr_d1 = max(best_mrr_d1,MRR_d1)
            best_mrr_d2 = max(best_mrr_d2,MRR_d2)       
            logger.info(f'Epoch: {epoch}/{args.epoch} \t'
                        f'Train Loss: {stats.loss_cls:.4f} dr_e Loss: {stats.loss_dr_e:.4f} \t'
                        f'Val loss: {val_loss:.4f}, cls loss: {val_cls_loss:.4f}\n'
                        f'val domain1 cur/max HR@1: {HIT_1_d1:.4f}/{best_hit_1_d1:.4f} \n,' 
                        f'HR@5: {HIT_5_d1:.4f}/{best_hit_5_d1:.4f} \n, '
                        f'HR@10: {HIT_10_d1:.4f}/{best_hit_10_d1:.4f} \n'
                        # f'val domain1 cur/max NDCG@1: {NDCG_1_d1:.4f}/{best_ndcg_1_d1:.4f} \n, '
                        f'NDCG@5: {NDCG_5_d1:.4f}/{best_ndcg_5_d1:.4f} \n, '
                        f'NDCG@10: {NDCG_10_d1:.4f}/{best_ndcg_10_d1:.4f}, \n'
                        f'val domain1 cur/max MRR: {MRR_d1:.4f}/{best_mrr_d1:.4f} \n'
                        f'val domain2 cur/max HR@1: {HIT_1_d2:.4f}/{best_hit_1_d2:.4f} \n, '
                        f'HR@5: {HIT_5_d2:.4f}/{best_hit_5_d2:.4f} \n, '
                        f'HR@10: {HIT_10_d2:.4f}/{best_hit_10_d2:.4f} \n'
                        # f'val domain2 cur/max NDCG@1: {NDCG_1_d2:.4f}/{best_ndcg_1_d2:.4f} \n, '
                        f'NDCG@5: {NDCG_5_d2:.4f}/{best_ndcg_5_d2:.4f} \n, '
                        f'NDCG@10: {NDCG_10_d2:.4f}/{best_ndcg_10_d2:.4f}, \n'
                        f'val domain2 cur/max MRR: {MRR_d2:.4f}/{best_mrr_d2:.4f} \n')
                        
        model.train()        
        for i,sample in enumerate(tqdm(trainLoaderDR)):
            u_node = torch.LongTensor(sample['user_node'].long()).cuda()
            i_node = torch.LongTensor(sample['i_node'].long()).cuda()
            neg_samples = torch.LongTensor(sample['neg_samples'].long()).cuda()
            seq_d1 = torch.LongTensor(sample['seq_d1'].long()).cuda()
            seq_d2 = torch.LongTensor(sample['seq_d2'].long()).cuda()
            long_tail_mask_d1 = torch.LongTensor(sample['long_tail_mask_d1'].long()).cuda()
            long_tail_mask_d2 = torch.LongTensor(sample['long_tail_mask_d2'].long()).cuda()
            domain_id = torch.LongTensor(sample['domain_id'].long()).cuda()
            ob_label = torch.LongTensor(sample['ob_label'].long()).cuda()
            labels = torch.LongTensor(sample['label'].long()).cuda()
            labels = labels.float()
            if not args.isDR:
                predict_d1, predict_d2 = model(u_node,i_node,neg_samples,seq_d1,seq_d2,long_tail_mask_d1,long_tail_mask_d2)
            else:
                predict_d1, predict_d2, predict_ips_d1, predict_ips_d2, predict_gfunc_d1, predict_gfunc_d2 = model(u_node,i_node,neg_samples,seq_d1,seq_d2,long_tail_mask_d1,long_tail_mask_d2)
            predict_d1 = predict_d1.squeeze()
            predict_d2 = predict_d2.squeeze()
            predict_gfunc_d1 = predict_gfunc_d1.squeeze()
            predict_gfunc_d2 = predict_gfunc_d2.squeeze()
            predict_ips_d1 = predict_ips_d1.squeeze()
            predict_ips_d2 = predict_ips_d2.squeeze()
            one_value = torch.LongTensor(torch.ones(domain_id.shape[0]).long()).cuda()
            mask_d1 = torch.LongTensor((one_value.cpu() - domain_id.cpu()).long()).cuda()
            mask_d2 = torch.LongTensor((domain_id.cpu()).long()).cuda()
            # print(predict_d1.shape)
            # print(labels.shape)
            # loss_cls = criterion_cls(predict_d1,labels) * mask_d1.unsqueeze(1)  + criterion_cls(predict_d2,labels) * mask_d2.unsqueeze(1) #* 2
            # loss_cls = torch.mean(loss_cls)
            ob_label = ob_label.unsqueeze(1).repeat(1,2)
            loss_dr_r = (predict_gfunc_d1**2+ob_label*((criterion_cls(predict_d1,labels)**2-predict_gfunc_d1**2)**2)/predict_ips_d1)* mask_d1.unsqueeze(1) + (predict_gfunc_d2**2+ob_label*((criterion_cls(predict_d2,labels)**2-predict_gfunc_d2**2)**2)/predict_ips_d2) * mask_d2.unsqueeze(1)
            loss2 = torch.mean(loss_dr_r)
            
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            # auc_avg_domain = roc_auc_score(labels.detach().cpu().numpy(),predict.detach().cpu().numpy())
            stats.update(loss_dr_r=loss2.item())
            if i % 20 == 0:
                logger.info(f'train loss_dr_r:{stats.loss_dr_r} \t')
            #print("epoch :{} train loss:{}, auc:{}".format(epoch,stats.loss,stats.auc)) 
        #val(epoch)
        if args.overlap:
            val_loss, val_cls_loss, HIT_1_d1_ov, NDCG_1_d1_ov, HIT_5_d1_ov, NDCG_5_d1_ov, HIT_10_d1_ov, NDCG_10_d1_ov, MRR_d1_ov, HIT_1_d1_no, NDCG_1_d1_no, HIT_5_d1_no, NDCG_5_d1_no, HIT_10_d1_no, NDCG_10_d1_no, MRR_d1_no, HIT_1_d2_ov, NDCG_1_d2_ov, HIT_5_d2_ov, NDCG_5_d2_ov, HIT_10_d2_ov, NDCG_10_d2_ov, MRR_d2_ov, HIT_1_d2_no, NDCG_1_d2_no, HIT_5_d2_no, NDCG_5_d2_no, HIT_10_d2_no, NDCG_10_d2_no, MRR_d2_no, HIT_1_d1, NDCG_1_d1, HIT_5_d1, NDCG_5_d1, HIT_10_d1, NDCG_10_d1, MRR_d1, HIT_1_d2, NDCG_1_d2, HIT_5_d2, NDCG_5_d2, HIT_10_d2, NDCG_10_d2, MRR_d2  = test(model,args,valLoader)
            best_hit_1_d1_ov = max(HIT_1_d1_ov,best_hit_1_d1_ov)
            best_hit_5_d1_ov = max(HIT_5_d1_ov,best_hit_5_d1_ov)
            best_hit_10_d1_ov = max(HIT_10_d1_ov,best_hit_10_d1_ov)
            best_hit_1_d2_ov = max(HIT_1_d2_ov,best_hit_1_d2_ov)
            best_hit_5_d2_ov = max(HIT_5_d2_ov,best_hit_5_d2_ov)
            best_hit_10_d2_ov = max(HIT_10_d2_ov,best_hit_10_d2_ov)

            best_ndcg_1_d1_ov = max(best_ndcg_1_d1_ov,NDCG_1_d1_ov)
            best_ndcg_5_d1_ov = max(best_ndcg_5_d1_ov,NDCG_5_d1_ov)
            best_ndcg_10_d1_ov = max(best_ndcg_10_d1_ov,NDCG_10_d1_ov)
            best_ndcg_1_d2_ov = max(best_ndcg_1_d2_ov,NDCG_1_d2_ov)
            best_ndcg_5_d2_ov = max(best_ndcg_5_d2_ov,NDCG_5_d2_ov)
            best_ndcg_10_d2_ov = max(best_ndcg_10_d2_ov,NDCG_10_d2_ov)
            best_mrr_d1_ov = max(best_mrr_d1_ov,MRR_d1_ov)
            best_mrr_d2_ov = max(best_mrr_d2_ov,MRR_d2_ov)   

            best_hit_1_d1_no = max(HIT_1_d1_no,best_hit_1_d1_no)
            best_hit_5_d1_no = max(HIT_5_d1_no,best_hit_5_d1_no)
            best_hit_10_d1_no = max(HIT_10_d1_no,best_hit_10_d1_no)
            best_hit_1_d2_no = max(HIT_1_d2_no,best_hit_1_d2_no)
            best_hit_5_d2_no = max(HIT_5_d2_no,best_hit_5_d2_no)
            best_hit_10_d2_no = max(HIT_10_d2_no,best_hit_10_d2_no)

            best_ndcg_1_d1_no = max(best_ndcg_1_d1_no,NDCG_1_d1_no)
            best_ndcg_5_d1_no = max(best_ndcg_5_d1_no,NDCG_5_d1_no)
            best_ndcg_10_d1_no = max(best_ndcg_10_d1_no,NDCG_10_d1_no)
            best_ndcg_1_d2_no = max(best_ndcg_1_d2_no,NDCG_1_d2_no)
            best_ndcg_5_d2_no = max(best_ndcg_5_d2_no,NDCG_5_d2_no)
            best_ndcg_10_d2_no = max(best_ndcg_10_d2_no,NDCG_10_d2_no)
            best_mrr_d1_no = max(best_mrr_d1_no,MRR_d1_no)
            best_mrr_d2_no = max(best_mrr_d2_no,MRR_d2_no)       

            best_hit_1_d1 = max(HIT_1_d1,best_hit_1_d1)
            best_hit_5_d1 = max(HIT_5_d1,best_hit_5_d1)
            best_hit_10_d1 = max(HIT_10_d1,best_hit_10_d1)
            best_hit_1_d2 = max(HIT_1_d2,best_hit_1_d2)
            best_hit_5_d2 = max(HIT_5_d2,best_hit_5_d2)
            best_hit_10_d2 = max(HIT_10_d2,best_hit_10_d2)

            best_ndcg_1_d1 = max(best_ndcg_1_d1,NDCG_1_d1)
            best_ndcg_5_d1 = max(best_ndcg_5_d1,NDCG_5_d1)
            best_ndcg_10_d1 = max(best_ndcg_10_d1,NDCG_10_d1)
            best_ndcg_1_d2 = max(best_ndcg_1_d2,NDCG_1_d2)
            best_ndcg_5_d2 = max(best_ndcg_5_d2,NDCG_5_d2)
            best_ndcg_10_d2 = max(best_ndcg_10_d2,NDCG_10_d2)
            best_mrr_d1 = max(best_mrr_d1,MRR_d1)
            best_mrr_d2 = max(best_mrr_d2,MRR_d2)
            logger.info(f'Epoch: {epoch}/{args.epoch} \t'
                        f'Train dr_r Loss : {stats.loss_dr_r:.4f} \t'
                        f'Val loss: {val_loss:.4f}, cls loss: {val_cls_loss:.4f}\n'
                        f'val domain1 cur/max overlap HR@1: {HIT_1_d1_ov:.4f}/{best_hit_1_d1_ov:.4f} \n,' 
                        f'overlap HR@5: {HIT_5_d1_ov:.4f}/{best_hit_5_d1_ov:.4f} \n, '
                        f'overlap HR@10: {HIT_10_d1_ov:.4f}/{best_hit_10_d1_ov:.4f} \n'
                        f'overlap NDCG@5: {NDCG_5_d1_ov:.4f}/{best_ndcg_5_d1_ov:.4f} \n, '
                        f'overlap NDCG@10: {NDCG_10_d1_ov:.4f}/{best_ndcg_10_d1_ov:.4f}, \n'
                        f'overlap MRR: {MRR_d1_ov:.4f}/{best_mrr_d1_ov:.4f} \n'
                        f'val domain1 cur/max non-overlap HR@1: {HIT_1_d1_no:.4f}/{best_hit_1_d1_no:.4f} \n,' 
                        f'non-overlap HR@5: {HIT_5_d1_no:.4f}/{best_hit_5_d1_no:.4f} \n, '
                        f'non-overlap HR@10: {HIT_10_d1_no:.4f}/{best_hit_10_d1_no:.4f} \n'
                        f'non-overlap NDCG@5: {NDCG_5_d1_no:.4f}/{best_ndcg_5_d1_no:.4f} \n, '
                        f'non-overlap NDCG@10: {NDCG_10_d1_no:.4f}/{best_ndcg_10_d1_no:.4f}, \n'
                        f'non-overlap MRR: {MRR_d1_no:.4f}/{best_mrr_d1_no:.4f} \n'
                        f'val domain2 cur/max overlap HR@1: {HIT_1_d2_ov:.4f}/{best_hit_1_d2_ov:.4f} \n, '
                        f'overlap HR@5: {HIT_5_d2_ov:.4f}/{best_hit_5_d2_ov:.4f} \n, '
                        f'overlap HR@10: {HIT_10_d2_ov:.4f}/{best_hit_10_d2_ov:.4f} \n'
                        f'overlap NDCG@5: {NDCG_5_d2_ov:.4f}/{best_ndcg_5_d2_ov:.4f} \n, '
                        f'overlap NDCG@10: {NDCG_10_d2_ov:.4f}/{best_ndcg_10_d2_ov:.4f}, \n'
                        f'overlap MRR: {MRR_d2_ov:.4f}/{best_mrr_d2_ov:.4f} \n'
                        f'val domain2 cur/max non-overlap HR@1: {HIT_1_d2_no:.4f}/{best_hit_1_d2_no:.4f} \n, '
                        f'non-overlap HR@5: {HIT_5_d2_no:.4f}/{best_hit_5_d2_no:.4f} \n, '
                        f'non-overlap HR@10: {HIT_10_d2_no:.4f}/{best_hit_10_d2_no:.4f} \n'
                        f'non-overlap NDCG@5: {NDCG_5_d2_no:.4f}/{best_ndcg_5_d2_no:.4f} \n, '
                        f'non-overlap NDCG@10: {NDCG_10_d2_no:.4f}/{best_ndcg_10_d2_no:.4f}, \n'
                        f'non-overlap MRR: {MRR_d2_no:.4f}/{best_mrr_d2_no:.4f} \n'
                        f'val domain1 cur/max HR@1: {HIT_1_d1:.4f}/{best_hit_1_d1:.4f} \n,' 
                        f'HR@5: {HIT_5_d1:.4f}/{best_hit_5_d1:.4f} \n, '
                        f'HR@10: {HIT_10_d1:.4f}/{best_hit_10_d1:.4f} \n'
                        # f'val domain1 cur/max NDCG@1: {NDCG_1_d1:.4f}/{best_ndcg_1_d1:.4f} \n, '
                        f'NDCG@5: {NDCG_5_d1:.4f}/{best_ndcg_5_d1:.4f} \n, '
                        f'NDCG@10: {NDCG_10_d1:.4f}/{best_ndcg_10_d1:.4f}, \n'
                        f'val domain1 cur/max MRR: {MRR_d1:.4f}/{best_mrr_d1:.4f} \n'
                        f'val domain2 cur/max HR@1: {HIT_1_d2:.4f}/{best_hit_1_d2:.4f} \n, '
                        f'HR@5: {HIT_5_d2:.4f}/{best_hit_5_d2:.4f} \n, '
                        f'HR@10: {HIT_10_d2:.4f}/{best_hit_10_d2:.4f} \n'
                        # f'val domain2 cur/max NDCG@1: {NDCG_1_d2:.4f}/{best_ndcg_1_d2:.4f} \n, '
                        f'NDCG@5: {NDCG_5_d2:.4f}/{best_ndcg_5_d2:.4f} \n, '
                        f'NDCG@10: {NDCG_10_d2:.4f}/{best_ndcg_10_d2:.4f}, \n'
                        f'val domain2 cur/max MRR: {MRR_d2:.4f}/{best_mrr_d2:.4f} \n')
        else:
            val_loss, val_cls_loss, HIT_1_d1, NDCG_1_d1, HIT_5_d1, NDCG_5_d1, HIT_10_d1, NDCG_10_d1, MRR_d1, HIT_1_d2, NDCG_1_d2, HIT_5_d2, NDCG_5_d2, HIT_10_d2, NDCG_10_d2, MRR_d2 = test(model,args,valLoader)
            best_hit_1_d1 = max(HIT_1_d1,best_hit_1_d1)
            best_hit_5_d1 = max(HIT_5_d1,best_hit_5_d1)
            best_hit_10_d1 = max(HIT_10_d1,best_hit_10_d1)
            best_hit_1_d2 = max(HIT_1_d2,best_hit_1_d2)
            best_hit_5_d2 = max(HIT_5_d2,best_hit_5_d2)
            best_hit_10_d2 = max(HIT_10_d2,best_hit_10_d2)

            best_ndcg_1_d1 = max(best_ndcg_1_d1,NDCG_1_d1)
            best_ndcg_5_d1 = max(best_ndcg_5_d1,NDCG_5_d1)
            best_ndcg_10_d1 = max(best_ndcg_10_d1,NDCG_10_d1)
            best_ndcg_1_d2 = max(best_ndcg_1_d2,NDCG_1_d2)
            best_ndcg_5_d2 = max(best_ndcg_5_d2,NDCG_5_d2)
            best_ndcg_10_d2 = max(best_ndcg_10_d2,NDCG_10_d2)
            # if MRR_d1 >= best_mrr_d1:
            #     best_auc1 = auc_testd1
                # torch.save(model.state_dict(), str(save_path1))
            # if MRR_d2 >= best_mrr_d2:
            #     best_auc2 = auc_testd2
                # torch.save(model.state_dict(), str(save_path2))
            best_mrr_d1 = max(best_mrr_d1,MRR_d1)
            best_mrr_d2 = max(best_mrr_d2,MRR_d2)       
            logger.info(f'Epoch: {epoch}/{args.epoch} \t'
                        f'Train dr_r Loss : {stats.loss_dr_r:.4f} \t'
                        f'Val loss: {val_loss:.4f}, cls loss: {val_cls_loss:.4f}\n'
                        f'val domain1 cur/max HR@1: {HIT_1_d1:.4f}/{best_hit_1_d1:.4f} \n,' 
                        f'HR@5: {HIT_5_d1:.4f}/{best_hit_5_d1:.4f} \n, '
                        f'HR@10: {HIT_10_d1:.4f}/{best_hit_10_d1:.4f} \n'
                        # f'val domain1 cur/max NDCG@1: {NDCG_1_d1:.4f}/{best_ndcg_1_d1:.4f} \n, '
                        f'NDCG@5: {NDCG_5_d1:.4f}/{best_ndcg_5_d1:.4f} \n, '
                        f'NDCG@10: {NDCG_10_d1:.4f}/{best_ndcg_10_d1:.4f}, \n'
                        f'val domain1 cur/max MRR: {MRR_d1:.4f}/{best_mrr_d1:.4f} \n'
                        f'val domain2 cur/max HR@1: {HIT_1_d2:.4f}/{best_hit_1_d2:.4f} \n, '
                        f'HR@5: {HIT_5_d2:.4f}/{best_hit_5_d2:.4f} \n, '
                        f'HR@10: {HIT_10_d2:.4f}/{best_hit_10_d2:.4f} \n'
                        # f'val domain2 cur/max NDCG@1: {NDCG_1_d2:.4f}/{best_ndcg_1_d2:.4f} \n, '
                        f'NDCG@5: {NDCG_5_d2:.4f}/{best_ndcg_5_d2:.4f} \n, '
                        f'NDCG@10: {NDCG_10_d2:.4f}/{best_ndcg_10_d2:.4f}, \n'
                        f'val domain2 cur/max MRR: {MRR_d2:.4f}/{best_mrr_d2:.4f} \n')
    if not args.overlap:     
        return best_hit_1_d1, best_hit_5_d1, best_hit_10_d1, best_ndcg_5_d1, best_ndcg_10_d1, best_mrr_d1, best_hit_1_d2, best_hit_5_d2, best_hit_10_d2, best_ndcg_5_d2, best_ndcg_10_d2, best_mrr_d2 
    else:
        return best_hit_1_d1_ov, best_hit_5_d1_ov, best_hit_10_d1_ov, best_ndcg_5_d1_ov, best_ndcg_10_d1_ov, best_mrr_d1_ov, best_hit_1_d1_no, best_hit_5_d1_no, best_hit_10_d1_no, best_ndcg_5_d1_no, best_ndcg_10_d1_no, best_mrr_d1_no, best_hit_1_d2_ov, best_hit_5_d2_ov, best_hit_10_d2_ov, best_ndcg_5_d2_ov, best_ndcg_10_d2_ov, best_mrr_d2_ov, best_hit_1_d2_no, best_hit_5_d2_no, best_hit_10_d2_no, best_ndcg_5_d2_no, best_ndcg_10_d2_no, best_mrr_d2_no, best_hit_1_d1, best_hit_5_d1, best_hit_10_d1, best_ndcg_5_d1, best_ndcg_10_d1, best_mrr_d1, best_hit_1_d2, best_hit_5_d2, best_hit_10_d2, best_ndcg_5_d2, best_ndcg_10_d2, best_mrr_d2  

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Multi-edge multi-domain training')
    parser.add_argument('--epoch', type=int, default=50, help='# of epoch')
    parser.add_argument('--bs', type=int, default=256, help='# images in batch')
    parser.add_argument('--use_gpu', type=bool, default=True, help='gpu flag, true for GPU and false for CPU')
    parser.add_argument('--lr', type=float, default=5e-4, help='initial learning rate for adam') #1e-3 for cdr23 3e-4 for cdr12
    parser.add_argument('--lr2', type=float, default=0.01, help='initial learning rate for adam') #1e-3 for cdr23 3e-4 for cdr12
    parser.add_argument('--emb_dim', type=int, default=128, help='embedding size')
    parser.add_argument('--hid_dim', type=int, default=32, help='hidden layer dim')
    parser.add_argument('--seq_len', type=int, default=20, help='the length of the sequence') # 20 for mybank 150 for amazon
    parser.add_argument('--graph_nums', type=int, default=2, help='numbers of graph layers')
    parser.add_argument('--head_nums', type=int, default=32, help='head nums for u-u graph')
    parser.add_argument('--long_length', type=int, default=7, help='the length for setting long-tail node')
    parser.add_argument('--m1_layers', type=int, default=3, help='m1 layer nums')
    parser.add_argument('--m2_layers', type=int, default=3, help='m2 layer nums')
    parser.add_argument('--m3_layers', type=int, default=4, help='m3 layer nums')
    parser.add_argument('--m4_layers', type=int, default=2, help='m4 layer nums')
    parser.add_argument('--alpha_l', type=int, default=3, help='sce loss')
    parser.add_argument('--neg_nums', type=int, default=199, help='sample negative numbers')
    parser.add_argument('--mask_rate_enc', type=float, default=0.9, help='mask rate for encoder')
    parser.add_argument('--mask_rate_dec', type=float, default=0.9, help='mask rate for decoder')
    parser.add_argument('--overlap_ratio', type=float, default=0.5, help='overlap ratio for choose dataset ')
    parser.add_argument('--bs_ratio', type=float, default=0.5, help='user-user connect ratio in the mini-batch graph')
    parser.add_argument('-md','--model-dir', type=str, default='model/')
    parser.add_argument('--log-file', type=str, default='log')
    parser.add_argument('--model', type=str, default='model select')
    parser.add_argument('-ds','--dataset_type', type=str, default='amazon')
    parser.add_argument('-dm','--domain_type', type=str, default='movie_book')
    parser.add_argument('--isInC', type=bool, default=False, help='add inc ')
    parser.add_argument('--isItC', type=bool, default=False, help='add itc')    
    parser.add_argument('--ts1', type=float, default=0.5, help='mask rate for encoder')
    parser.add_argument('--ts2', type=float, default=0.5, help='mask rate for decoder')
    parser.add_argument('--overlap', type=bool, default=False, help='divided the performance by the overlapped users and non-overlapped users')    
    parser.add_argument('--isDR', type=bool, default=True, help='add itc')    
    parser.add_argument('--dr_e_w', type=float, default=0.1, help='mask rate for decoder')

    args = parser.parse_args()

    hit_1_d1 = []
    hit_5_d1 = []
    hit_10_d1 = []
    hit_1_d2 = []
    hit_5_d2 = []
    hit_10_d2 = []

    ndcg_5_d1 = []
    ndcg_10_d1 = []
    ndcg_5_d2 = []
    ndcg_10_d2 = []

    mrr_d1 = []
    mrr_d2 = []

    hit_1_d1_ov = []
    hit_5_d1_ov = []
    hit_10_d1_ov = []
    hit_1_d2_ov = []
    hit_5_d2_ov = []
    hit_10_d2_ov = []

    ndcg_5_d1_ov = []
    ndcg_10_d1_ov = []
    ndcg_5_d2_ov = []
    ndcg_10_d2_ov = []

    mrr_d1_ov = []
    mrr_d2_ov = []

    hit_1_d1_no = []
    hit_5_d1_no = []
    hit_10_d1_no = []
    hit_1_d2_no = []
    hit_5_d2_no = []
    hit_10_d2_no = []

    ndcg_5_d1_no = []
    ndcg_10_d1_no = []
    ndcg_5_d2_no = []
    ndcg_10_d2_no = []

    mrr_d1_no = []
    mrr_d2_no = []

    for i in range(5):#for i in range(5):
        SEED = i
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        args.log_file = "log" + str(i) + ".txt"
        
        user_length = 895510#63275#6814 cdr23 #63275 cdr12
        item_length_d1 = 8240
        item_length_d2 = 26272
        item_length = 447410#item_length_d1 + item_length_d2 + 1 + 20000#1739+2 #13713 cdr23 #1739 + 2#item_length_d1 + item_length_d2 + 1 + 20000#1739 + 1 +200 # 1 = pad item #item_length_d1 + item_length_d2 + 1 + 20000
        datasetTrain = DualDomainSeqDataset(seq_len=args.seq_len,isTrain=True,neg_nums=args.neg_nums,long_length=args.long_length,pad_id=item_length+1,csv_path="{}_dataset/{}_train".format(args.dataset_type,args.domain_type)+str(int(args.overlap_ratio*100))+".csv")
        trainLoader = data.DataLoader(datasetTrain, batch_size=args.bs, shuffle=True, num_workers=8,drop_last=True,collate_fn=collate_fn_enhance)

        datasetTrainDR = DualDomainSeqDatasetDR(seq_len=args.seq_len,isTrain=True,neg_nums=args.neg_nums,long_length=args.long_length,pad_id=item_length+1,csv_path="{}_dataset/{}_train".format(args.dataset_type,args.domain_type)+str(int(args.overlap_ratio*100))+"_DR.csv")
        trainLoaderDR = data.DataLoader(datasetTrainDR, batch_size=args.bs, shuffle=True, num_workers=8,drop_last=True,collate_fn=collate_fn_enhanceDR)

        datasetVal = DualDomainSeqDataset(seq_len=args.seq_len,isTrain=False,neg_nums=args.neg_nums,long_length=args.long_length,pad_id=item_length+1,csv_path="{}_dataset/{}_test.csv".format(args.dataset_type,args.domain_type))
        valLoader = data.DataLoader(datasetVal, batch_size=args.bs, shuffle=False, num_workers=8,drop_last=True,collate_fn=collate_fn_enhance)
        item_length = item_length * 2  #for pad id
        user_length = user_length * 2
        if args.model.lower() == "gru4rec":
            model = GRU4Rec(user_length=user_length, user_emb_dim=args.emb_dim, item_length=item_length, item_emb_dim=args.emb_dim, seq_len=args.seq_len, hid_dim=args.hid_dim, bs=args.bs, isInC=args.isInC, isItC=args.isItC, threshold1=args.ts1, threshold2=args.ts2, isDR=args.isDR).cuda()
        elif args.model.lower() == "sasrec":
            model = SASRec(user_length=user_length, user_emb_dim=args.emb_dim, item_length=item_length, item_emb_dim=args.emb_dim, seq_len=args.seq_len, hid_dim=args.hid_dim, bs=args.bs, isInC=args.isInC, isItC=args.isItC, threshold1=args.ts1, threshold2=args.ts2, isDR=args.isDR).cuda()
        elif args.model.lower() == "bert4rec":
            model = BERT4Rec(user_length=user_length, user_emb_dim=args.emb_dim, item_length=item_length, item_emb_dim=args.emb_dim, seq_len=args.seq_len, hid_dim=args.hid_dim, bs=args.bs, isInC=args.isInC, isItC=args.isItC, threshold1=args.ts1, threshold2=args.ts2, isDR=args.isDR).cuda()
        cuda = True if torch.cuda.is_available() else False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("find cuda right !!\n")
        # if cuda:
        #     torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # else:
        #     torch.set_default_tensor_type('torch.FloatTensor')

        if cuda:
            #model = torch.nn.DataParallel(model)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            #cudnn.benchmark = True
            model = model.cuda()
            # model.to(device)
            print("use cuda!")
        optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)
        optimizer2 = torch.optim.Adam(model.parameters(),lr = args.lr*args.lr2)
        init_logger(args.model_dir, args.log_file)
        logger.info(vars(args))
        # if os.path.exists(args.model_dir + "best_d1.pt"):
        #     print("load_pretrained")
        #     state_dict = torch.load(args.model_dir + "best_d1.pt")
        #     model.load_state_dict(state_dict,strict=False)
        if not args.overlap:
            best_hit_1_d1, best_hit_5_d1, best_hit_10_d1, best_ndcg_5_d1, best_ndcg_10_d1, best_mrr_d1, best_hit_1_d2, best_hit_5_d2, best_hit_10_d2, best_ndcg_5_d2, best_ndcg_10_d2, best_mrr_d2  = train(model,trainLoader,trainLoaderDR,args,valLoader)
            # test(model,args,valLoader)
            hit_1_d1.append(best_hit_1_d1)
            hit_5_d1.append(best_hit_5_d1)
            hit_10_d1.append(best_hit_10_d1)
            ndcg_5_d1.append(best_ndcg_5_d1)
            ndcg_10_d1.append(best_ndcg_10_d1)
            mrr_d1.append(best_mrr_d1)

            hit_1_d2.append(best_hit_1_d2)
            hit_5_d2.append(best_hit_5_d2)
            hit_10_d2.append(best_hit_10_d2)
            ndcg_5_d2.append(best_ndcg_5_d2)
            ndcg_10_d2.append(best_ndcg_10_d2)
            mrr_d2.append(best_mrr_d2)
            # break
        else:
            best_hit_1_d1_ov, best_hit_5_d1_ov, best_hit_10_d1_ov, best_ndcg_5_d1_ov, best_ndcg_10_d1_ov, best_mrr_d1_ov, best_hit_1_d1_no, best_hit_5_d1_no, best_hit_10_d1_no, best_ndcg_5_d1_no, best_ndcg_10_d1_no, best_mrr_d1_no, best_hit_1_d2_ov, best_hit_5_d2_ov, best_hit_10_d2_ov, best_ndcg_5_d2_ov, best_ndcg_10_d2_ov, best_mrr_d2_ov, best_hit_1_d2_no, best_hit_5_d2_no, best_hit_10_d2_no, best_ndcg_5_d2_no, best_ndcg_10_d2_no, best_mrr_d2_no, best_hit_1_d1, best_hit_5_d1, best_hit_10_d1, best_ndcg_5_d1, best_ndcg_10_d1, best_mrr_d1, best_hit_1_d2, best_hit_5_d2, best_hit_10_d2, best_ndcg_5_d2, best_ndcg_10_d2, best_mrr_d2    = train(model,trainLoader,trainLoaderDR,args,valLoader)
            # test(model,args,valLoader)
            hit_1_d1_ov.append(best_hit_1_d1_ov)
            hit_5_d1_ov.append(best_hit_5_d1_ov)
            hit_10_d1_ov.append(best_hit_10_d1_ov)
            ndcg_5_d1_ov.append(best_ndcg_5_d1_ov)
            ndcg_10_d1_ov.append(best_ndcg_10_d1_ov)
            mrr_d1_ov.append(best_mrr_d1_ov)

            hit_1_d2_ov.append(best_hit_1_d2_ov)
            hit_5_d2_ov.append(best_hit_5_d2_ov)
            hit_10_d2_ov.append(best_hit_10_d2_ov)
            ndcg_5_d2_ov.append(best_ndcg_5_d2_ov)
            ndcg_10_d2_ov.append(best_ndcg_10_d2_ov)
            mrr_d2_ov.append(best_mrr_d2_ov)

            hit_1_d1_no.append(best_hit_1_d1_no)
            hit_5_d1_no.append(best_hit_5_d1_no)
            hit_10_d1_no.append(best_hit_10_d1_no)
            ndcg_5_d1_no.append(best_ndcg_5_d1_no)
            ndcg_10_d1_no.append(best_ndcg_10_d1_no)
            mrr_d1_no.append(best_mrr_d1_no)

            hit_1_d2_no.append(best_hit_1_d2_no)
            hit_5_d2_no.append(best_hit_5_d2_no)
            hit_10_d2_no.append(best_hit_10_d2_no)
            ndcg_5_d2_no.append(best_ndcg_5_d2_no)
            ndcg_10_d2_no.append(best_ndcg_10_d2_no)
            mrr_d2_no.append(best_mrr_d2_no)

            hit_1_d1.append(best_hit_1_d1)
            hit_5_d1.append(best_hit_5_d1)
            hit_10_d1.append(best_hit_10_d1)
            ndcg_5_d1.append(best_ndcg_5_d1)
            ndcg_10_d1.append(best_ndcg_10_d1)
            mrr_d1.append(best_mrr_d1)

            hit_1_d2.append(best_hit_1_d2)
            hit_5_d2.append(best_hit_5_d2)
            hit_10_d2.append(best_hit_10_d2)
            ndcg_5_d2.append(best_ndcg_5_d2)
            ndcg_10_d2.append(best_ndcg_10_d2)
            mrr_d2.append(best_mrr_d2)

    if not args.overlap:
        log_all_txt = "log_all.txt"
        init_logger(args.model_dir, log_all_txt)
        logger.info(f'domain1 HR@1: {np.mean(hit_1_d1):.4f}/{np.std(hit_1_d1):.4f} \n,' 
                    f'HR@5: {np.mean(hit_5_d1):.4f}/{np.std(hit_5_d1):.4f} \n, '
                    f'HR@10: {np.mean(hit_10_d1):.4f}/{np.std(hit_10_d1):.4f} \n'
                    f'NDCG@5: {np.mean(ndcg_5_d1):.4f}/{np.std(ndcg_5_d1):.4f} \n, '
                    f'NDCG@10: {np.mean(ndcg_10_d1):.4f}/{np.std(ndcg_10_d1):.4f}, \n'
                    f'MRR: {np.mean(mrr_d1):.4f}/{np.std(mrr_d1):.4f} \n'
                    f'domain2 HR@1: {np.mean(hit_1_d2):.4f}/{np.std(hit_1_d2):.4f} \n,' 
                    f'HR@5: {np.mean(hit_5_d2):.4f}/{np.std(hit_5_d2):.4f} \n, '
                    f'HR@10: {np.mean(hit_10_d2):.4f}/{np.std(hit_10_d2):.4f} \n'
                    f'NDCG@5: {np.mean(ndcg_5_d2):.4f}/{np.std(ndcg_5_d2):.4f} \n, '
                    f'NDCG@10: {np.mean(ndcg_10_d2):.4f}/{np.std(ndcg_10_d2):.4f}, \n'
                    f'MRR: {np.mean(mrr_d2):.4f}/{np.std(mrr_d2):.4f} \n'
                    f'Avg HR@1: {(np.mean(hit_1_d2)+np.mean(hit_1_d1))/2:.4f}/{(np.std(hit_1_d2)**2+np.std(hit_1_d1)**2)**0.5:.4f} \n,' 
                    f'HR@5: {(np.mean(hit_5_d2)+np.mean(hit_5_d1))/2:.4f}/{(np.std(hit_5_d2)**2+np.std(hit_5_d1)**2)**0.5:.4f} \n, '
                    f'HR@10: {(np.mean(hit_10_d2)+np.mean(hit_10_d1))/2:.4f}/{(np.std(hit_10_d2)**2+np.std(hit_10_d1)**2)**0.5:.4f} \n'
                    f'NDCG@5: {(np.mean(ndcg_5_d2)+np.mean(ndcg_5_d1))/2:.4f}/{(np.std(ndcg_5_d2)**2+np.std(ndcg_5_d1)**2)**0.5:.4f} \n, '
                    f'NDCG@10: {(np.mean(ndcg_10_d2)+np.mean(ndcg_10_d1))/2:.4f}/{(np.std(ndcg_10_d2)**2+np.std(ndcg_10_d1)**2)**0.5:.4f}, \n'
                    f'MRR: {(np.mean(mrr_d2)+np.mean(mrr_d1))/2:.4f}/{(np.std(mrr_d2)**2+np.std(mrr_d1)**2)**0.5:.4f} \n')
    else:
        log_all_txt = "log_all.txt"
        init_logger(args.model_dir, log_all_txt)
        logger.info(f'domain1 overlap HR@1: {np.mean(hit_1_d1_ov):.4f}/{np.std(hit_1_d1_ov):.4f} \n,' 
                    f'overlap HR@5: {np.mean(hit_5_d1_ov):.4f}/{np.std(hit_5_d1_ov):.4f} \n, '
                    f'overlap HR@10: {np.mean(hit_10_d1_ov):.4f}/{np.std(hit_10_d1_ov):.4f} \n'
                    f'overlap NDCG@5: {np.mean(ndcg_5_d1_ov):.4f}/{np.std(ndcg_5_d1_ov):.4f} \n, '
                    f'overlap NDCG@10: {np.mean(ndcg_10_d1_ov):.4f}/{np.std(ndcg_10_d1_ov):.4f}, \n'
                    f'overlap MRR: {np.mean(mrr_d1_ov):.4f}/{np.std(mrr_d1_ov):.4f} \n'
                    f'domain1 non-overlap HR@1: {np.mean(hit_1_d1_no):.4f}/{np.std(hit_1_d1_no):.4f} \n,' 
                    f'non-overlap HR@5: {np.mean(hit_5_d1_no):.4f}/{np.std(hit_5_d1_no):.4f} \n, '
                    f'non-overlap HR@10: {np.mean(hit_10_d1_no):.4f}/{np.std(hit_10_d1_no):.4f} \n'
                    f'non-overlap NDCG@5: {np.mean(ndcg_5_d1_no):.4f}/{np.std(ndcg_5_d1_no):.4f} \n, '
                    f'non-overlap NDCG@10: {np.mean(ndcg_10_d1_no):.4f}/{np.std(ndcg_10_d1_no):.4f}, \n'
                    f'non-overlap MRR: {np.mean(mrr_d1_no):.4f}/{np.std(mrr_d1_no):.4f} \n'
                    f'overlap domain2 HR@1: {np.mean(hit_1_d2_ov):.4f}/{np.std(hit_1_d2_ov):.4f} \n,' 
                    f'overlap HR@5: {np.mean(hit_5_d2_ov):.4f}/{np.std(hit_5_d2_ov):.4f} \n, '
                    f'overlap HR@10: {np.mean(hit_10_d2_ov):.4f}/{np.std(hit_10_d2_ov):.4f} \n'
                    f'overlap NDCG@5: {np.mean(ndcg_5_d2_ov):.4f}/{np.std(ndcg_5_d2_ov):.4f} \n, '
                    f'overlap NDCG@10: {np.mean(ndcg_10_d2_ov):.4f}/{np.std(ndcg_10_d2_ov):.4f}, \n'
                    f'overlap MRR: {np.mean(mrr_d2_ov):.4f}/{np.std(mrr_d2_ov):.4f} \n'
                    f'non-overlap domain2 HR@1: {np.mean(hit_1_d2_no):.4f}/{np.std(hit_1_d2_no):.4f} \n,' 
                    f'non-overlap HR@5: {np.mean(hit_5_d2_no):.4f}/{np.std(hit_5_d2_no):.4f} \n, '
                    f'non-overlap HR@10: {np.mean(hit_10_d2_no):.4f}/{np.std(hit_10_d2_no):.4f} \n'
                    f'non-overlap NDCG@5: {np.mean(ndcg_5_d2_no):.4f}/{np.std(ndcg_5_d2_no):.4f} \n, '
                    f'non-overlap NDCG@10: {np.mean(ndcg_10_d2_no):.4f}/{np.std(ndcg_10_d2_no):.4f}, \n'
                    f'non-overlap MRR: {np.mean(mrr_d2_no):.4f}/{np.std(mrr_d2_no):.4f} \n'
                    f'overlap Avg HR@1: {(np.mean(hit_1_d2_ov)+np.mean(hit_1_d1_ov))/2:.4f}/{(np.std(hit_1_d2_ov)**2+np.std(hit_1_d1_ov)**2)**0.5:.4f} \n,' 
                    f'overlap HR@5: {(np.mean(hit_5_d2_ov)+np.mean(hit_5_d1_ov))/2:.4f}/{(np.std(hit_5_d2_ov)**2+np.std(hit_5_d1_ov)**2)**0.5:.4f} \n, '
                    f'overlap HR@10: {(np.mean(hit_10_d2_ov)+np.mean(hit_10_d1_ov))/2:.4f}/{(np.std(hit_10_d2_ov)**2+np.std(hit_10_d1_ov)**2)**0.5:.4f} \n'
                    f'overlap NDCG@5: {(np.mean(ndcg_5_d2_ov)+np.mean(ndcg_5_d1_ov))/2:.4f}/{(np.std(ndcg_5_d2_ov)**2+np.std(ndcg_5_d1_ov)**2)**0.5:.4f} \n, '
                    f'overlap NDCG@10: {(np.mean(ndcg_10_d2_ov)+np.mean(ndcg_10_d1_ov))/2:.4f}/{(np.std(ndcg_10_d2_ov)**2+np.std(ndcg_10_d1_ov)**2)**0.5:.4f}, \n'
                    f'overlap MRR: {(np.mean(mrr_d2_ov)+np.mean(mrr_d1_ov))/2:.4f}/{(np.std(mrr_d2_ov)**2+np.std(mrr_d1_ov)**2)**0.5:.4f} \n'
                    f'non-overlap Avg HR@1: {(np.mean(hit_1_d2_no)+np.mean(hit_1_d1_no))/2:.4f}/{(np.std(hit_1_d2_no)**2+np.std(hit_1_d1_no)**2)**0.5:.4f} \n,' 
                    f'non-overlap HR@5: {(np.mean(hit_5_d2_no)+np.mean(hit_5_d1_no))/2:.4f}/{(np.std(hit_5_d2_no)**2+np.std(hit_5_d1_no)**2)**0.5:.4f} \n, '
                    f'non-overlap HR@10: {(np.mean(hit_10_d2_no)+np.mean(hit_10_d1_no))/2:.4f}/{(np.std(hit_10_d2_no)**2+np.std(hit_10_d1_no)**2)**0.5:.4f} \n'
                    f'non-overlap NDCG@5: {(np.mean(ndcg_5_d2_no)+np.mean(ndcg_5_d1_no))/2:.4f}/{(np.std(ndcg_5_d2_no)**2+np.std(ndcg_5_d1_no)**2)**0.5:.4f} \n, '
                    f'non-overlap NDCG@10: {(np.mean(ndcg_10_d2_no)+np.mean(ndcg_10_d1_no))/2:.4f}/{(np.std(ndcg_10_d2_no)**2+np.std(ndcg_10_d1_no)**2)**0.5:.4f}, \n'
                    f'non-overlap MRR: {(np.mean(mrr_d2_no)+np.mean(mrr_d1_no))/2:.4f}/{(np.std(mrr_d2_no)**2+np.std(mrr_d1_no)**2)**0.5:.4f} \n'
                    f'domain1 HR@1: {np.mean(hit_1_d1):.4f}/{np.std(hit_1_d1):.4f} \n,' 
                    f'HR@5: {np.mean(hit_5_d1):.4f}/{np.std(hit_5_d1):.4f} \n, '
                    f'HR@10: {np.mean(hit_10_d1):.4f}/{np.std(hit_10_d1):.4f} \n'
                    f'NDCG@5: {np.mean(ndcg_5_d1):.4f}/{np.std(ndcg_5_d1):.4f} \n, '
                    f'NDCG@10: {np.mean(ndcg_10_d1):.4f}/{np.std(ndcg_10_d1):.4f}, \n'
                    f'MRR: {np.mean(mrr_d1):.4f}/{np.std(mrr_d1):.4f} \n'
                    f'domain2 HR@1: {np.mean(hit_1_d2):.4f}/{np.std(hit_1_d2):.4f} \n,' 
                    f'HR@5: {np.mean(hit_5_d2):.4f}/{np.std(hit_5_d2):.4f} \n, '
                    f'HR@10: {np.mean(hit_10_d2):.4f}/{np.std(hit_10_d2):.4f} \n'
                    f'NDCG@5: {np.mean(ndcg_5_d2):.4f}/{np.std(ndcg_5_d2):.4f} \n, '
                    f'NDCG@10: {np.mean(ndcg_10_d2):.4f}/{np.std(ndcg_10_d2):.4f}, \n'
                    f'MRR: {np.mean(mrr_d2):.4f}/{np.std(mrr_d2):.4f} \n'
                    f'Avg HR@1: {(np.mean(hit_1_d2)+np.mean(hit_1_d1))/2:.4f}/{(np.std(hit_1_d2)**2+np.std(hit_1_d1)**2)**0.5:.4f} \n,' 
                    f'HR@5: {(np.mean(hit_5_d2)+np.mean(hit_5_d1))/2:.4f}/{(np.std(hit_5_d2)**2+np.std(hit_5_d1)**2)**0.5:.4f} \n, '
                    f'HR@10: {(np.mean(hit_10_d2)+np.mean(hit_10_d1))/2:.4f}/{(np.std(hit_10_d2)**2+np.std(hit_10_d1)**2)**0.5:.4f} \n'
                    f'NDCG@5: {(np.mean(ndcg_5_d2)+np.mean(ndcg_5_d1))/2:.4f}/{(np.std(ndcg_5_d2)**2+np.std(ndcg_5_d1)**2)**0.5:.4f} \n, '
                    f'NDCG@10: {(np.mean(ndcg_10_d2)+np.mean(ndcg_10_d1))/2:.4f}/{(np.std(ndcg_10_d2)**2+np.std(ndcg_10_d1)**2)**0.5:.4f}, \n'
                    f'MRR: {(np.mean(mrr_d2)+np.mean(mrr_d1))/2:.4f}/{(np.std(mrr_d2)**2+np.std(mrr_d1)**2)**0.5:.4f} \n')