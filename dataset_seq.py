import os
import random
from typing import DefaultDict
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from collections import defaultdict
import json

def seq_padding(seq, length_enc, long_length, pad_id):
    if len(seq)>= long_length:
        long_mask = 1
    else:
        long_mask = 0
    if len(seq) >= length_enc:
        enc_in = seq[-length_enc + 1:]
    else:
        enc_in = [pad_id] * (length_enc - len(seq) - 1) + seq

    return enc_in, long_mask

class SingleDomainSeqDataset(data.Dataset):
    def __init__(self,seq_len,isTrain,neg_nums,long_length,pad_id,subdomain,csv_path=''):
        super(SingleDomainSeqDataset, self).__init__()
        self.user_item_data = pd.read_csv(csv_path)
        self.user_item_data = self.user_item_data.loc[self.user_item_data['domain_id']==subdomain]

        print(self.user_item_data['user_id'].max())
        self.user_nodes = self.user_item_data['user_id'].tolist()
        #self.user_nodes = self.__encode_uid__(self.user_nodes_old)
        if subdomain==0:
            self.seq = self.user_item_data['seq_d1'].tolist()
        elif subdomain==1:
            self.seq = self.user_item_data['seq_d2'].tolist()
        self.seq_d1 = self.user_item_data['seq_d1'].tolist()
        self.seq_d2 = self.user_item_data['seq_d2'].tolist()
        self.domain_id = self.user_item_data['domain_id'].tolist()
        self.item_pool = self.__build_i_set__(self.seq)
        print("domain 1 len:{}".format(len(self.item_pool)))
        self.seq_len = seq_len
        self.isTrain = isTrain
        self.neg_nums = neg_nums
        self.long_length = long_length
        self.pad_id = pad_id
        self.subdomain = subdomain
    
    def __build_i_set__(self,seq1):
        item_d1 = list()
        for item_seq in seq1:
            item_seq_list = json.loads(item_seq)
            for i_tmp in item_seq_list:
                item_d1.append(i_tmp)
        item_pool_d1 = set(item_d1)
        return item_pool_d1

    def __encode_uid__(self,user_nodes):
        u_node_dict = defaultdict(list)
        i = 0
        u_node_new = list()
        for u_node_tmp in user_nodes:
            if len(u_node_dict[u_node_tmp])==0:
                u_node_dict[u_node_tmp].append(i)
                i += 1
        for u_node_tmp in user_nodes:
            u_node_new.append(u_node_dict[u_node_tmp][0])
        print("u_id len:{}".format(len(u_node_dict)))
        return u_node_new

    def __len__(self):
        print("dataset len:{}\n".format(len(self.user_nodes)))
        return len(self.user_nodes)

    def __getitem__(self, idx):
        user_node = self.user_nodes[idx]
        seq_tmp = json.loads(self.seq[idx])
        seq_d1_tmp = json.loads(self.seq_d1[idx])
        seq_d2_tmp = json.loads(self.seq_d2[idx])
        if len(seq_d1_tmp)!=0 and len(seq_d2_tmp)!=0:
            overlap_label = 1
        else:
            overlap_label = 0
        label = list()
        neg_items_set = self.item_pool - set(seq_tmp)
        item = seq_tmp[-1]
        seq_tmp = seq_tmp[:-1]
        label.append(1)
        # print("item :{}".format(item))
        # print("seq before:{}".format(seq_d1_tmp))
        while(item in seq_tmp):
            seq_tmp.remove(item)
        # print("seq after:{}".format(seq_d1_tmp))
        if self.isTrain:
            neg_samples = random.sample(neg_items_set, 1)
            label.append(0)
        else:
            neg_samples = random.sample(neg_items_set, self.neg_nums)
            for _ in range(self.neg_nums):
                label.append(0)
        seq_tmp,long_tail_mask = seq_padding(seq_tmp,self.seq_len+1,self.long_length,self.pad_id)
        sample = dict()
        sample['user_node'] = np.array([user_node])
        sample['i_node'] = np.array([item])
        sample['seq'] = np.array([seq_tmp])
        sample['long_tail_mask'] = np.array([long_tail_mask])
        sample['overlap_label'] = np.array([overlap_label])
        sample['label'] = np.array(label) # no need copy
        sample['neg_samples'] = np.array(neg_samples)
        sample['label'] = sample['label']
        return sample

def collate_fn_enhance_SD(batch):
    user_node = torch.cat([ torch.Tensor(sample['user_node']) for sample in batch],dim=0)
    i_node = torch.cat([ torch.Tensor(sample['i_node']) for sample in batch],dim=0)
    seq = torch.cat([ torch.Tensor(sample['seq']) for sample in batch],dim=0)
    long_tail_mask = torch.cat([ torch.Tensor(sample['long_tail_mask']) for sample in batch],dim=0)
    label = torch.stack([ torch.Tensor(sample['label']) for sample in batch],dim=0)
    overlap_label = torch.cat([ torch.Tensor(sample['overlap_label']) for sample in batch],dim=0)
    neg_samples = torch.stack([ torch.Tensor(sample['neg_samples']) for sample in batch],dim=0)
    data = {'user_node' : user_node,
            'i_node': i_node,
            'seq' : seq,
            'long_tail_mask' : long_tail_mask,
            'label':label,
            'overlap_label' : overlap_label,
            'neg_samples':neg_samples
            }
    return data

class DualDomainSeqDataset(data.Dataset):
    def __init__(self,seq_len,isTrain,neg_nums,long_length,pad_id,csv_path=''):
        super(DualDomainSeqDataset, self).__init__()
        self.user_item_data = pd.read_csv(csv_path)
        print(self.user_item_data['user_id'].max())
        self.user_nodes = self.user_item_data['user_id'].tolist()
        #self.user_nodes = self.__encode_uid__(self.user_nodes_old)
        self.seq_d1 = self.user_item_data['seq_d1'].tolist()
        self.seq_d2 = self.user_item_data['seq_d2'].tolist()
        self.domain_id = self.user_item_data['domain_id'].tolist()
        self.item_pool_d1 = self.__build_i_set__(self.seq_d1)
        self.item_pool_d2 = self.__build_i_set__(self.seq_d2)
        print("domain 1 len:{}".format(len(self.item_pool_d1)))
        print("domain 2 len:{}".format(len(self.item_pool_d2)))        
        self.seq_len = seq_len
        self.isTrain = isTrain
        self.neg_nums = neg_nums
        self.long_length = long_length
        self.pad_id = pad_id
    
    def __build_i_set__(self,seq1):
        item_d1 = list()
        for item_seq in seq1:
            item_seq_list = json.loads(item_seq)
            for i_tmp in item_seq_list:
                item_d1.append(i_tmp)
        item_pool_d1 = set(item_d1)
        return item_pool_d1

    def __encode_uid__(self,user_nodes):
        u_node_dict = defaultdict(list)
        i = 0
        u_node_new = list()
        for u_node_tmp in user_nodes:
            if len(u_node_dict[u_node_tmp])==0:
                u_node_dict[u_node_tmp].append(i)
                i += 1
        for u_node_tmp in user_nodes:
            u_node_new.append(u_node_dict[u_node_tmp][0])
        print("u_id len:{}".format(len(u_node_dict)))
        return u_node_new

    def __len__(self):
        print("dataset len:{}\n".format(len(self.user_nodes)))
        return len(self.user_nodes)

    def __getitem__(self, idx):
        user_node = self.user_nodes[idx]
        seq_d1_tmp = json.loads(self.seq_d1[idx])
        seq_d2_tmp = json.loads(self.seq_d2[idx])
        if len(seq_d1_tmp)!=0 and len(seq_d2_tmp)!=0:
            overlap_label = 1
        else:
            overlap_label = 0
        domain_id_old = self.domain_id[idx]
        label = list()
        if domain_id_old == 0:
            neg_items_set = self.item_pool_d1 - set(seq_d1_tmp)
            item = seq_d1_tmp[-1]
            seq_d1_tmp = seq_d1_tmp[:-1]
            label.append(1)
            # print("item :{}".format(item))
            # print("seq before:{}".format(seq_d1_tmp))
            while(item in seq_d1_tmp):
                seq_d1_tmp.remove(item)
            # print("seq after:{}".format(seq_d1_tmp))
            if self.isTrain:
                neg_samples = random.sample(neg_items_set, 1)
                label.append(0)
            else:
                neg_samples = random.sample(neg_items_set, self.neg_nums)
                for _ in range(self.neg_nums):
                    label.append(0)
            domain_id = 0
        else:
            neg_items_set = self.item_pool_d2 - set(seq_d2_tmp)
            item = seq_d2_tmp[-1]
            seq_d2_tmp = seq_d2_tmp[:-1] 
            label.append(1)
            # print("item :{}".format(item))
            # print("seq before:{}".format(seq_d2_tmp))
            while(item in seq_d2_tmp):
                seq_d2_tmp.remove(item)
            # print("seq after:{}".format(seq_d2_tmp))
            if self.isTrain:
                neg_samples = random.sample(neg_items_set, 1)
                label.append(0)
            else:
                neg_samples = random.sample(neg_items_set, self.neg_nums)
                for _ in range(self.neg_nums):
                    label.append(0)
            domain_id = 1
        seq_d1_tmp,long_tail_mask_d1 = seq_padding(seq_d1_tmp,self.seq_len+1,self.long_length,self.pad_id)
        seq_d2_tmp,long_tail_mask_d2 = seq_padding(seq_d2_tmp,self.seq_len+1,self.long_length,self.pad_id)
        sample = dict()
        sample['user_node'] = np.array([user_node])
        sample['i_node'] = np.array([item])
        sample['seq_d1'] = np.array([seq_d1_tmp])
        sample['seq_d2'] = np.array([seq_d2_tmp])
        sample['long_tail_mask_d1'] = np.array([long_tail_mask_d1])
        sample['long_tail_mask_d2'] = np.array([long_tail_mask_d2])
        sample['domain_id'] = np.array([domain_id])
        sample['overlap_label'] = np.array([overlap_label])
        sample['label'] = np.array(label) # no need copy
        sample['neg_samples'] = np.array(neg_samples)
        # copy neg item
        # sample['user_node'] = np.repeat(sample['user_node'], sample['neg_samples'].shape[0]+1, axis=0)
        # sample['seq_d1'] = np.repeat(sample['seq_d1'], sample['neg_samples'].shape[0]+1, axis=0)
        # sample['seq_d2'] = np.repeat(sample['seq_d2'], sample['neg_samples'].shape[0]+1, axis=0)
        # sample['domain_id'] = np.repeat(sample['domain_id'], sample['neg_samples'].shape[0]+1, axis=0)
        # sample['i_node'] = np.concatenate((sample['i_node'],sample['neg_samples']),axis=0)
        sample['label'] = sample['label']
        # print(sample['label'].shape)
        # print("user_node:{}".format(sample['user_node']))
        # print("i_node:{}".format(sample['i_node']))
        # print("seq_d1:{}".format(sample['seq_d1']))
        # print("seq_d2:{}".format(sample['seq_d2']))
        # print("domain_id:{}".format(sample['domain_id']))
        # print("neg_samples:{}".format(sample['neg_samples']))
        return sample

def collate_fn_enhance(batch):
    user_node = torch.cat([ torch.Tensor(sample['user_node']) for sample in batch],dim=0)
    i_node = torch.cat([ torch.Tensor(sample['i_node']) for sample in batch],dim=0)
    seq_d1 = torch.cat([ torch.Tensor(sample['seq_d1']) for sample in batch],dim=0)
    seq_d2 = torch.cat([ torch.Tensor(sample['seq_d2']) for sample in batch],dim=0)
    long_tail_mask_d1 = torch.cat([ torch.Tensor(sample['long_tail_mask_d1']) for sample in batch],dim=0)
    long_tail_mask_d2 = torch.cat([ torch.Tensor(sample['long_tail_mask_d2']) for sample in batch],dim=0)
    label = torch.stack([ torch.Tensor(sample['label']) for sample in batch],dim=0)
    domain_id = torch.cat([ torch.Tensor(sample['domain_id']) for sample in batch],dim=0)
    overlap_label = torch.cat([ torch.Tensor(sample['overlap_label']) for sample in batch],dim=0)
    neg_samples = torch.stack([ torch.Tensor(sample['neg_samples']) for sample in batch],dim=0)
    data = {'user_node' : user_node,
            'i_node': i_node,
            'seq_d1' : seq_d1,
            'seq_d2': seq_d2,
            'long_tail_mask_d1' : long_tail_mask_d1,
            'long_tail_mask_d2': long_tail_mask_d2,
            'label':label,
            'domain_id' : domain_id,
            'overlap_label' : overlap_label,
            'neg_samples':neg_samples
            }
    return data

def generate_corr_seq(real_seq,fake_seq):
    seq = list()
    for i in range(len(real_seq)):
        seq.append(real_seq[i])
        seq.append(fake_seq[i])
    return seq

class DualDomainSeqDatasetC2DSR(data.Dataset):
    def __init__(self,seq_len,isTrain,neg_nums,long_length,pad_id,csv_path=''):
        super(DualDomainSeqDatasetC2DSR, self).__init__()
        self.user_item_data = pd.read_csv(csv_path)
        print(self.user_item_data['user_id'].max())
        self.user_nodes = self.user_item_data['user_id'].tolist()
        #self.user_nodes = self.__encode_uid__(self.user_nodes_old)
        self.seq_d1 = self.user_item_data['seq_d1'].tolist()
        self.seq_d2 = self.user_item_data['seq_d2'].tolist()
        self.domain_id = self.user_item_data['domain_id'].tolist()
        self.item_pool_d1 = self.__build_i_set__(self.seq_d1)
        self.item_pool_d2 = self.__build_i_set__(self.seq_d2)
        print("domain 1 len:{}".format(len(self.item_pool_d1)))
        print("domain 2 len:{}".format(len(self.item_pool_d2)))        
        self.seq_len = seq_len
        self.isTrain = isTrain
        self.neg_nums = neg_nums
        self.long_length = long_length
        self.pad_id = pad_id
    
    def __build_i_set__(self,seq1):
        item_d1 = list()
        for item_seq in seq1:
            item_seq_list = json.loads(item_seq)
            for i_tmp in item_seq_list:
                item_d1.append(i_tmp)
        item_pool_d1 = set(item_d1)
        return item_pool_d1

    def __encode_uid__(self,user_nodes):
        u_node_dict = defaultdict(list)
        i = 0
        u_node_new = list()
        for u_node_tmp in user_nodes:
            if len(u_node_dict[u_node_tmp])==0:
                u_node_dict[u_node_tmp].append(i)
                i += 1
        for u_node_tmp in user_nodes:
            u_node_new.append(u_node_dict[u_node_tmp][0])
        print("u_id len:{}".format(len(u_node_dict)))
        return u_node_new

    def __len__(self):
        print("dataset len:{}\n".format(len(self.user_nodes)))
        return len(self.user_nodes)

    def __getitem__(self, idx):
        user_node = self.user_nodes[idx]
        seq_d1_tmp = json.loads(self.seq_d1[idx])
        seq_d2_tmp = json.loads(self.seq_d2[idx])
        domain_id_old = self.domain_id[idx]
        if len(seq_d1_tmp)!=0 and len(seq_d2_tmp)!=0:
            overlap_label = 1
        else:
            overlap_label = 0
        label = list()
        # seq_corr = list()
        if domain_id_old == 0:
            neg_items_set = self.item_pool_d1 - set(seq_d1_tmp)
            item = seq_d1_tmp[-1]
            seq_d1_tmp = seq_d1_tmp[:-1]
            label.append(1)
            # print("item :{}".format(item))
            # print("seq before:{}".format(seq_d1_tmp))
            while(item in seq_d1_tmp):
                seq_d1_tmp.remove(item)
            # print("seq after:{}".format(seq_d1_tmp))
            if self.isTrain:
                neg_samples = random.sample(neg_items_set, 1)
                label.append(0)
            else:
                neg_samples = random.sample(neg_items_set, self.neg_nums)
                for _ in range(self.neg_nums):
                    label.append(0)
            domain_id = 0
            corr_seq = random.sample(neg_items_set, self.seq_len)
        else:
            neg_items_set = self.item_pool_d2 - set(seq_d2_tmp)
            item = seq_d2_tmp[-1]
            seq_d2_tmp = seq_d2_tmp[:-1] 
            label.append(1)
            # print("item :{}".format(item))
            # print("seq before:{}".format(seq_d2_tmp))
            while(item in seq_d2_tmp):
                seq_d2_tmp.remove(item)
            # print("seq after:{}".format(seq_d2_tmp))
            if self.isTrain:
                neg_samples = random.sample(neg_items_set, 1)
                label.append(0)
            else:
                neg_samples = random.sample(neg_items_set, self.neg_nums)
                for _ in range(self.neg_nums):
                    label.append(0)
            domain_id = 1
            corr_seq = random.sample(neg_items_set, self.seq_len)
        seq_d1_tmp,long_tail_mask_d1 = seq_padding(seq_d1_tmp,self.seq_len+1,self.long_length,self.pad_id)
        seq_d2_tmp,long_tail_mask_d2 = seq_padding(seq_d2_tmp,self.seq_len+1,self.long_length,self.pad_id)
        corr_seq_d1 = generate_corr_seq(seq_d1_tmp,corr_seq)
        corr_seq_d2 = generate_corr_seq(seq_d2_tmp,corr_seq)
        all_seq = generate_corr_seq(seq_d1_tmp,seq_d2_tmp)
        sample = dict()
        sample['user_node'] = np.array([user_node])
        sample['i_node'] = np.array([item])
        sample['seq_d1'] = np.array([seq_d1_tmp])
        sample['seq_d2'] = np.array([seq_d2_tmp])
        sample['corr_seq_d1'] = np.array([corr_seq_d1])
        sample['corr_seq_d2'] = np.array([corr_seq_d2])
        sample['all_seq'] = np.array([all_seq])
        sample['long_tail_mask_d1'] = np.array([long_tail_mask_d1])
        sample['long_tail_mask_d2'] = np.array([long_tail_mask_d2])
        sample['domain_id'] = np.array([domain_id])
        sample['overlap_label'] = np.array([overlap_label])
        sample['label'] = np.array(label) # no need copy
        sample['neg_samples'] = np.array(neg_samples)
        # copy neg item
        # sample['user_node'] = np.repeat(sample['user_node'], sample['neg_samples'].shape[0]+1, axis=0)
        # sample['seq_d1'] = np.repeat(sample['seq_d1'], sample['neg_samples'].shape[0]+1, axis=0)
        # sample['seq_d2'] = np.repeat(sample['seq_d2'], sample['neg_samples'].shape[0]+1, axis=0)
        # sample['domain_id'] = np.repeat(sample['domain_id'], sample['neg_samples'].shape[0]+1, axis=0)
        # sample['i_node'] = np.concatenate((sample['i_node'],sample['neg_samples']),axis=0)
        sample['label'] = sample['label']
        # print(sample['label'].shape)
        # print("user_node:{}".format(sample['user_node']))
        # print("i_node:{}".format(sample['i_node']))
        # print("seq_d1:{}".format(sample['seq_d1']))
        # print("seq_d2:{}".format(sample['seq_d2']))
        # print("domain_id:{}".format(sample['domain_id']))
        # print("neg_samples:{}".format(sample['neg_samples']))
        return sample

def collate_fn_enhanceC2DSR(batch):
    user_node = torch.cat([ torch.Tensor(sample['user_node']) for sample in batch],dim=0)
    i_node = torch.cat([ torch.Tensor(sample['i_node']) for sample in batch],dim=0)
    seq_d1 = torch.cat([ torch.Tensor(sample['seq_d1']) for sample in batch],dim=0)
    seq_d2 = torch.cat([ torch.Tensor(sample['seq_d2']) for sample in batch],dim=0)
    corr_seq_d1 = torch.cat([ torch.Tensor(sample['corr_seq_d1']) for sample in batch],dim=0)
    corr_seq_d2 = torch.cat([ torch.Tensor(sample['corr_seq_d2']) for sample in batch],dim=0)
    all_seq = torch.cat([ torch.Tensor(sample['all_seq']) for sample in batch],dim=0)
    long_tail_mask_d1 = torch.cat([ torch.Tensor(sample['long_tail_mask_d1']) for sample in batch],dim=0)
    long_tail_mask_d2 = torch.cat([ torch.Tensor(sample['long_tail_mask_d2']) for sample in batch],dim=0)
    label = torch.stack([ torch.Tensor(sample['label']) for sample in batch],dim=0)
    domain_id = torch.cat([ torch.Tensor(sample['domain_id']) for sample in batch],dim=0)
    overlap_label = torch.cat([ torch.Tensor(sample['overlap_label']) for sample in batch],dim=0)
    neg_samples = torch.stack([ torch.Tensor(sample['neg_samples']) for sample in batch],dim=0)
    data = {'user_node' : user_node,
            'i_node': i_node,
            'seq_d1' : seq_d1,
            'seq_d2': seq_d2,
            'corr_seq_d1' : corr_seq_d1,
            'corr_seq_d2': corr_seq_d2,
            'all_seq': all_seq,
            'long_tail_mask_d1' : long_tail_mask_d1,
            'long_tail_mask_d2': long_tail_mask_d2,
            'overlap_label' : overlap_label,
            'label':label,
            'domain_id' : domain_id,
            'neg_samples':neg_samples
            }
    return data

class DualDomainSeqDatasetDR(data.Dataset):
    def __init__(self,seq_len,isTrain,neg_nums,long_length,pad_id,csv_path=''):
        super(DualDomainSeqDatasetDR, self).__init__()
        self.user_item_data = pd.read_csv(csv_path)
        print(self.user_item_data['user_id'].max())
        self.user_nodes = self.user_item_data['user_id'].tolist()
        #self.user_nodes = self.__encode_uid__(self.user_nodes_old)
        self.seq_d1 = self.user_item_data['seq_d1'].tolist()
        self.seq_d2 = self.user_item_data['seq_d2'].tolist()
        self.domain_id = self.user_item_data['domain_id'].tolist()
        self.observe_label = self.user_item_data['ob_label'].tolist()
        self.item_pool_d1 = self.__build_i_set__(self.seq_d1)
        self.item_pool_d2 = self.__build_i_set__(self.seq_d2)
        print("domain 1 len:{}".format(len(self.item_pool_d1)))
        print("domain 2 len:{}".format(len(self.item_pool_d2)))        
        self.seq_len = seq_len
        self.isTrain = isTrain
        self.neg_nums = neg_nums
        self.long_length = long_length
        self.pad_id = pad_id
    
    def __build_i_set__(self,seq1):
        item_d1 = list()
        for item_seq in seq1:
            item_seq_list = json.loads(item_seq)
            for i_tmp in item_seq_list:
                item_d1.append(i_tmp)
        item_pool_d1 = set(item_d1)
        return item_pool_d1

    def __encode_uid__(self,user_nodes):
        u_node_dict = defaultdict(list)
        i = 0
        u_node_new = list()
        for u_node_tmp in user_nodes:
            if len(u_node_dict[u_node_tmp])==0:
                u_node_dict[u_node_tmp].append(i)
                i += 1
        for u_node_tmp in user_nodes:
            u_node_new.append(u_node_dict[u_node_tmp][0])
        print("u_id len:{}".format(len(u_node_dict)))
        return u_node_new

    def __len__(self):
        print("dataset len:{}\n".format(len(self.user_nodes)))
        return len(self.user_nodes)

    def __getitem__(self, idx):
        user_node = self.user_nodes[idx]
        seq_d1_tmp = json.loads(self.seq_d1[idx])
        seq_d2_tmp = json.loads(self.seq_d2[idx])
        if len(seq_d1_tmp)!=0 and len(seq_d2_tmp)!=0:
            overlap_label = 1
        else:
            overlap_label = 0
        domain_id_old = self.domain_id[idx]
        ob_label = self.observe_label[idx]
        label = list()
        if domain_id_old == 0:
            neg_items_set = self.item_pool_d1 - set(seq_d1_tmp)
            item = seq_d1_tmp[-1]
            seq_d1_tmp = seq_d1_tmp[:-1]
            label.append(1)
            # print("item :{}".format(item))
            # print("seq before:{}".format(seq_d1_tmp))
            while(item in seq_d1_tmp):
                seq_d1_tmp.remove(item)
            # print("seq after:{}".format(seq_d1_tmp))
            if self.isTrain:
                neg_samples = random.sample(neg_items_set, 1)
                label.append(0)
            else:
                neg_samples = random.sample(neg_items_set, self.neg_nums)
                for _ in range(self.neg_nums):
                    label.append(0)
            domain_id = 0
        else:
            neg_items_set = self.item_pool_d2 - set(seq_d2_tmp)
            item = seq_d2_tmp[-1]
            seq_d2_tmp = seq_d2_tmp[:-1] 
            label.append(1)
            # print("item :{}".format(item))
            # print("seq before:{}".format(seq_d2_tmp))
            while(item in seq_d2_tmp):
                seq_d2_tmp.remove(item)
            # print("seq after:{}".format(seq_d2_tmp))
            if self.isTrain:
                neg_samples = random.sample(neg_items_set, 1)
                label.append(0)
            else:
                neg_samples = random.sample(neg_items_set, self.neg_nums)
                for _ in range(self.neg_nums):
                    label.append(0)
            domain_id = 1
        seq_d1_tmp,long_tail_mask_d1 = seq_padding(seq_d1_tmp,self.seq_len+1,self.long_length,self.pad_id)
        seq_d2_tmp,long_tail_mask_d2 = seq_padding(seq_d2_tmp,self.seq_len+1,self.long_length,self.pad_id)
        sample = dict()
        sample['user_node'] = np.array([user_node])
        sample['i_node'] = np.array([item])
        sample['seq_d1'] = np.array([seq_d1_tmp])
        sample['seq_d2'] = np.array([seq_d2_tmp])
        sample['long_tail_mask_d1'] = np.array([long_tail_mask_d1])
        sample['long_tail_mask_d2'] = np.array([long_tail_mask_d2])
        sample['domain_id'] = np.array([domain_id])
        sample['ob_label'] = np.array([ob_label])
        sample['overlap_label'] = np.array([overlap_label])
        sample['label'] = np.array(label) # no need copy
        sample['neg_samples'] = np.array(neg_samples)
        # copy neg item
        # sample['user_node'] = np.repeat(sample['user_node'], sample['neg_samples'].shape[0]+1, axis=0)
        # sample['seq_d1'] = np.repeat(sample['seq_d1'], sample['neg_samples'].shape[0]+1, axis=0)
        # sample['seq_d2'] = np.repeat(sample['seq_d2'], sample['neg_samples'].shape[0]+1, axis=0)
        # sample['domain_id'] = np.repeat(sample['domain_id'], sample['neg_samples'].shape[0]+1, axis=0)
        # sample['i_node'] = np.concatenate((sample['i_node'],sample['neg_samples']),axis=0)
        sample['label'] = sample['label']
        # print(sample['label'].shape)
        # print("user_node:{}".format(sample['user_node']))
        # print("i_node:{}".format(sample['i_node']))
        # print("seq_d1:{}".format(sample['seq_d1']))
        # print("seq_d2:{}".format(sample['seq_d2']))
        # print("domain_id:{}".format(sample['domain_id']))
        # print("neg_samples:{}".format(sample['neg_samples']))
        return sample

def collate_fn_enhanceDR(batch):
    user_node = torch.cat([ torch.Tensor(sample['user_node']) for sample in batch],dim=0)
    i_node = torch.cat([ torch.Tensor(sample['i_node']) for sample in batch],dim=0)
    seq_d1 = torch.cat([ torch.Tensor(sample['seq_d1']) for sample in batch],dim=0)
    seq_d2 = torch.cat([ torch.Tensor(sample['seq_d2']) for sample in batch],dim=0)
    long_tail_mask_d1 = torch.cat([ torch.Tensor(sample['long_tail_mask_d1']) for sample in batch],dim=0)
    long_tail_mask_d2 = torch.cat([ torch.Tensor(sample['long_tail_mask_d2']) for sample in batch],dim=0)
    label = torch.stack([ torch.Tensor(sample['label']) for sample in batch],dim=0)
    domain_id = torch.cat([ torch.Tensor(sample['domain_id']) for sample in batch],dim=0)
    ob_label = torch.cat([ torch.Tensor(sample['ob_label']) for sample in batch],dim=0)
    overlap_label = torch.cat([ torch.Tensor(sample['overlap_label']) for sample in batch],dim=0)
    neg_samples = torch.stack([ torch.Tensor(sample['neg_samples']) for sample in batch],dim=0)
    data = {'user_node' : user_node,
            'i_node': i_node,
            'seq_d1' : seq_d1,
            'seq_d2': seq_d2,
            'ob_label': ob_label,
            'long_tail_mask_d1' : long_tail_mask_d1,
            'long_tail_mask_d2': long_tail_mask_d2,
            'label':label,
            'domain_id' : domain_id,
            'overlap_label' : overlap_label,
            'neg_samples':neg_samples
            }
    return data