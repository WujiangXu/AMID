import os
import random
from typing import DefaultDict
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
import json
from random import sample

def select_overlap_user(train_name,save_train_name,overlap_ratio):
    data = pd.read_csv(train_name)
    user_node = data['user_id'].tolist()
    seq_d1 = data['seq_d1'].tolist()
    seq_d2 = data['seq_d2'].tolist()
    domain_id = data['domain_id'].tolist()
    user_node_overlap,seq_d1_overlap, seq_d2_overlap, domain_id_overlap  = [], [], [], []
    user_node_nolap,seq_d1_nolap, seq_d2_nolap, domain_id_nolap, ob_nolap  = [], [], [], [], []
    for i in range(len(user_node)):
        seq1_tmp = json.loads(seq_d1[i])
        seq2_tmp = json.loads(seq_d2[i])
        if len(seq1_tmp)!=0 and len(seq2_tmp)!=0:
            user_node_overlap.append(user_node[i])
            seq_d1_overlap.append(seq1_tmp)
            seq_d2_overlap.append(seq2_tmp)
            domain_id_overlap.append(domain_id[i])
            ob_nolap.append(1)
        else :
            user_node_nolap.append(user_node[i])
            seq_d1_nolap.append(seq1_tmp)
            seq_d2_nolap.append(seq2_tmp)
            domain_id_nolap.append(domain_id[i])
            # ob_nolap.append(1) # observed data
    print(len(user_node_overlap),len(user_node_nolap)) # 3384 69945
    #nolap_num = int(len(user_node_overlap)/overlap_ratio-len(user_node_overlap)) # 3384 + 
    sample_nolap_num = int(len(user_node_nolap)*overlap_ratio)
    idx_lst = [i for i in range(len(user_node_nolap))]
    select_idx = sample(idx_lst, sample_nolap_num)
    not_select_idx = list(set(idx_lst).difference(set(select_idx)))
    print(sample_nolap_num,len(not_select_idx))
    # print(select_idx)
    for idx_tmp in select_idx:
        user_node_overlap.append(user_node_nolap[idx_tmp])
        seq_d1_overlap.append(seq_d1_nolap[idx_tmp])
        seq_d2_overlap.append(seq_d2_nolap[idx_tmp])
        domain_id_overlap.append(domain_id_nolap[idx_tmp])
        # user_node_nolap.append(user_node_overlap[idx_tmp])
        # seq_d1_nolap.append(seq_d1_overlap[idx_tmp])
        # seq_d2_nolap.append(seq_d2_overlap[idx_tmp])
        # domain_id_nolap.append(domain_id_overlap[idx_tmp])
        ob_nolap.append(1) # observed data
    # print(len(user_node_nolap))
    for idx_tmp in not_select_idx:
        user_node_overlap.append(user_node_nolap[idx_tmp])
        seq_d1_overlap.append(seq_d1_nolap[idx_tmp])
        seq_d2_overlap.append(seq_d2_nolap[idx_tmp])
        domain_id_overlap.append(domain_id_nolap[idx_tmp])
        # user_node_nolap.append(user_node_overlap[idx_tmp])
        # seq_d1_nolap.append(seq_d1_overlap[idx_tmp])
        # seq_d2_nolap.append(seq_d2_overlap[idx_tmp])
        # domain_id_nolap.append(domain_id_overlap[idx_tmp])
        ob_nolap.append(0) # observed data
    # append not observed data 
    dataframe = pd.DataFrame({'user_id':user_node_overlap,'seq_d1':seq_d1_overlap,'seq_d2':seq_d2_overlap,'domain_id':domain_id_overlap,'ob_label':ob_nolap})
    dataframe.to_csv(save_train_name,index=False,sep=',')

train_name = "/ossfs/workspace/CDSR/amazon_dataset_oldnodr/phone_elec_train100.csv"

ratios = [0.25,0.75]
for overlap_ratio in ratios:
    save_train_name = "/ossfs/workspace/CDSR/amazon_dataset/phone_elec_train"+str(int(overlap_ratio*100))+"_DR.csv"
    select_overlap_user(train_name,save_train_name,overlap_ratio)