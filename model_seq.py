import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from random import sample
import math

class embUserLayerEnhance(nn.Module):
    def __init__(self,user_length,emb_dim):
        super(embUserLayerEnhance, self).__init__()
        self.emb_user_share = nn.Embedding(user_length,emb_dim)
        self.transd1 = nn.Linear(emb_dim,emb_dim)
        self.transd2 = nn.Linear(emb_dim,emb_dim)

    def forward(self, user_id):
        user_nomarl = self.emb_user_share(user_id)
        user_spf1 = self.transd1(user_nomarl)
        user_spf2 = self.transd2(user_nomarl)
        return user_spf1, user_spf2#, user_nomarl

class embItemLayerEnhance(nn.Module):
    def __init__(self,item_length,emb_dim):
        super(embItemLayerEnhance, self).__init__()
        self.emb_item = nn.Embedding(item_length,emb_dim)

    def forward(self,item_id):
        item_f = self.emb_item(item_id)
        return item_f


class predictModule(nn.Module):
    def __init__(self, emb_dim, hid_dim):
        super(predictModule, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim*2,hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim,1))
    
    def forward(self, user_spf1, user_spf2, i_feat):
        '''
            user_spf : [bs,dim]
            i_feat : [bs,dim]
            neg_samples_feat: [bs,1/99,dim] 1 for train, 99 for test
        '''
        user_spf1 = user_spf1.unsqueeze(1).expand_as(i_feat)
        user_item_concat_feat_d1 = torch.cat((user_spf1,i_feat),-1)
        logits_d1 = torch.sigmoid(self.fc(user_item_concat_feat_d1))

        user_spf2 = user_spf2.unsqueeze(1).expand_as(i_feat)
        user_item_concat_feat_d2 = torch.cat((user_spf2,i_feat),-1)
        logits_d2 = torch.sigmoid(self.fc(user_item_concat_feat_d2))

        return logits_d1.squeeze(), logits_d2.squeeze()

class GRU4Rec(nn.Module):

    def __init__(self, user_length, user_emb_dim, item_length, item_emb_dim, seq_len, hid_dim, bs, isInC, isItC, threshold1, threshold2, isDR=False):
        super(GRU4Rec, self).__init__()
        self.user_emb_dim = user_emb_dim
        # self.user_emb_layer = embUserLayerEnhance(user_length, user_emb_dim)
        self.item_emb_layer = embItemLayerEnhance(item_length, item_emb_dim)
        self.isInC = isInC
        self.isItC = isItC
        if self.isInC:
            seq_len *= 2
            self.inc_d1 = InnerComp(user_emb_dim,bs,threshold1)
            self.inc_d2 = InnerComp(user_emb_dim,bs,threshold1)
        if self.isItC: 
            self.itc_d1 = InterComp(user_emb_dim,bs,threshold2)
            self.itc_d2 = InterComp(user_emb_dim,bs,threshold2)
        self.gru1 = nn.GRU(user_emb_dim, user_emb_dim, 1, batch_first=True, dropout=0.5)
        self.gru2 = nn.GRU(user_emb_dim, user_emb_dim, 1, batch_first=True, dropout=0.5)
        self.predictModule = predictModule(user_emb_dim,hid_dim)
        self.isDR = isDR
        if self.isDR:
            self.predict_ips = predictModule(user_emb_dim,hid_dim)
            self.predict_gfunc = predictModule(user_emb_dim,hid_dim)
        # self.down1 = nn.Linear(user_emb_dim*2,user_emb_dim)
        # self.down2 = nn.Linear(user_emb_dim*2,user_emb_dim)


    def forward(self,u_node,i_node,neg_samples,seq_d1,seq_d2,long_tail_mask_d1,long_tail_mask_d2,isTrain=True):
        # user_spf1, user_spf2 = self.user_emb_layer(u_node)
        i_feat = self.item_emb_layer(i_node).unsqueeze(1)
        neg_samples_feat = self.item_emb_layer(neg_samples)
        seq_d1_feat = self.item_emb_layer(seq_d1)
        seq_d2_feat = self.item_emb_layer(seq_d2)
        if self.isInC:
            seq_d1_feat = self.inc_d1(seq_d1_feat)
            seq_d2_feat = self.inc_d2(seq_d2_feat)
        hidden1 = torch.zeros(1, i_feat.shape[0], self.user_emb_dim).cuda()
        hidden2 = torch.zeros(1, i_feat.shape[0], self.user_emb_dim).cuda()
        seq_d1_feat, hidden1 = self.gru1(seq_d1_feat, hidden1)
        seq_d2_feat, hidden2 = self.gru2(seq_d2_feat, hidden2)
        # print(seq_d1_feat.shape,torch.mean(seq_d1_feat,1).shape,user_spf1.shape)
        if self.isItC:
            seq_d1_init = seq_d1_feat
            seq_d2_init = seq_d2_feat
            seq_d1_feat = self.itc_d1(seq_d1_init,seq_d2_init)
            seq_d2_feat = self.itc_d2(seq_d2_init,seq_d1_init)
        u_feat_enhance_d1  = torch.mean(seq_d1_feat,1)
        # print(u_feat_enhance_d1.shape)
        u_feat_enhance_d2  = torch.mean(seq_d2_feat,1)
        i_feat = torch.cat((i_feat,neg_samples_feat),1)   
        if self.isDR:
            logits_d1, logits_d2 = self.predictModule(u_feat_enhance_d1, u_feat_enhance_d2, i_feat)
            logits_ips_d1, logits_ips_d2 = self.predict_ips(u_feat_enhance_d1, u_feat_enhance_d2, i_feat)
            logits_gfunc_d1, logits_gfunc_d2 = self.predict_gfunc(u_feat_enhance_d1, u_feat_enhance_d2, i_feat)
            return logits_d1, logits_d2, logits_ips_d1, logits_ips_d2, logits_gfunc_d1, logits_gfunc_d2
        else:
            logits_d1, logits_d2 = self.predictModule(u_feat_enhance_d1, u_feat_enhance_d2, i_feat)
            return logits_d1, logits_d2#, u_feat_enhance_m1_d1, u_feat_enhance_m3_d1, u_feat_enhance_m4_d1

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None: 
            # print(mask.shape,scores.shape) #torch.Size([256, 1, 40, 20]) torch.Size([256, 4, 40, 40]) torch.Size([256, 1, 20, 20]) torch.Size([256, 4, 20, 20])
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))



class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class BERT4Rec(nn.Module):

    def __init__(self, user_length, user_emb_dim, item_length, item_emb_dim, seq_len, hid_dim, bs, isInC, isItC, threshold1, threshold2, isDR=False):
        super(BERT4Rec, self).__init__()
        self.user_emb_dim = user_emb_dim
        # self.user_emb_layer = embUserLayerEnhance(user_length, user_emb_dim)
        self.item_emb_layer = embItemLayerEnhance(item_length, item_emb_dim)
        self.isInC = isInC
        self.isItC = isItC
        if self.isInC:
            seq_len *= 2
            self.inc_d1 = InnerComp(user_emb_dim,bs,threshold1)
            self.inc_d2 = InnerComp(user_emb_dim,bs,threshold1)
        if self.isItC: 
            self.itc_d1 = InterComp(user_emb_dim,bs,threshold2)
            self.itc_d2 = InterComp(user_emb_dim,bs,threshold2)
        self.transform1 = nn.ModuleList(
            [TransformerBlock(128, 4, 128 * 4, 0.1) for _ in range(2)])
        self.transform2 = nn.ModuleList(
            [TransformerBlock(128, 4, 128 * 4, 0.1) for _ in range(2)])
        self.predictModule = predictModule(user_emb_dim,hid_dim)
        self.isDR = isDR
        if self.isDR:
            self.predict_ips = predictModule(user_emb_dim,hid_dim)
            self.predict_gfunc = predictModule(user_emb_dim,hid_dim)
        # self.down1 = nn.Linear(user_emb_dim*2,user_emb_dim)
        # self.down2 = nn.Linear(user_emb_dim*2,user_emb_dim)


    def forward(self,u_node,i_node,neg_samples,seq_d1,seq_d2,long_tail_mask_d1,long_tail_mask_d2,isTrain=True):
        # user_spf1, user_spf2 = self.user_emb_layer(u_node)
        i_feat = self.item_emb_layer(i_node).unsqueeze(1)
        neg_samples_feat = self.item_emb_layer(neg_samples)
        seq_d1_feat = self.item_emb_layer(seq_d1)
        seq_d2_feat = self.item_emb_layer(seq_d2)
        if self.isInC:
            seq_d1_feat = self.inc_d1(seq_d1_feat)
            seq_d2_feat = self.inc_d2(seq_d2_feat)
            mask = (seq_d2 > 0).unsqueeze(1).repeat(1, seq_d2_feat.size(1), 2).unsqueeze(1)
        else:
            mask = (seq_d2 > 0).unsqueeze(1).repeat(1, seq_d2_feat.size(1), 1).unsqueeze(1)
        if self.isItC:
            seq_d1_init = seq_d1_feat
            seq_d2_init = seq_d2_feat
            seq_d1_feat = self.itc_d1(seq_d1_init,seq_d2_init)
            seq_d2_feat = self.itc_d2(seq_d2_init,seq_d1_init)
            mask = (seq_d2 > 0).unsqueeze(1).repeat(1, seq_d2_feat.size(1), 2).unsqueeze(1)
        for transformer1 in self.transform1:
            seq_d1_feat = transformer1.forward(seq_d1_feat,mask)
        for transformer2 in self.transform2:
            seq_d2_feat = transformer2.forward(seq_d2_feat,mask)
        u_feat_enhance_d1  = torch.mean(seq_d1_feat,1)
        u_feat_enhance_d2  = torch.mean(seq_d2_feat,1)
        i_feat = torch.cat((i_feat,neg_samples_feat),1)   
        if self.isDR:
            logits_d1, logits_d2 = self.predictModule(u_feat_enhance_d1, u_feat_enhance_d2, i_feat)
            logits_ips_d1, logits_ips_d2 = self.predict_ips(u_feat_enhance_d1, u_feat_enhance_d2, i_feat)
            logits_gfunc_d1, logits_gfunc_d2 = self.predict_gfunc(u_feat_enhance_d1, u_feat_enhance_d2, i_feat)
            return logits_d1, logits_d2, logits_ips_d1, logits_ips_d2, logits_gfunc_d1, logits_gfunc_d2
        else:
            logits_d1, logits_d2 = self.predictModule(u_feat_enhance_d1, u_feat_enhance_d2, i_feat)
            return logits_d1, logits_d2#, u_feat_enhance_m1_d1, u_feat_enhance_m3_d1, u_feat_enhance_m4_d1

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py
class Log2feats(torch.nn.Module):
    def __init__(self, user_length, user_emb_dim, item_length, item_emb_dim, seq_len, hid_dim):
        super(Log2feats, self).__init__()
        self.pos_emb = torch.nn.Embedding(seq_len, item_emb_dim) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=0.5)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(user_emb_dim, eps=1e-8)

        for _ in range(2):
            new_attn_layernorm = torch.nn.LayerNorm(user_emb_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(user_emb_dim,
                                                            8,
                                                            0.5)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(user_emb_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(user_emb_dim, 0.5)
            self.forward_layers.append(new_fwd_layer)

    def forward(self, log_seqs):
        seqs = log_seqs
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).cuda())
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs.cpu() == 0).cuda()
        seqs *= ~timeline_mask # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device="cuda"))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats


class SASRec(torch.nn.Module):
    def __init__(self, user_length, user_emb_dim, item_length, item_emb_dim, seq_len, hid_dim, bs, isInC, isItC, threshold1, threshold2, isDR=False):
        super(SASRec, self).__init__()

        self.user_emb_dim = user_emb_dim
        # self.user_emb_layer = embUserLayerEnhance(user_length, user_emb_dim)
        self.item_emb_layer = embItemLayerEnhance(item_length, item_emb_dim)
        self.isInC = isInC
        self.isItC = isItC
        if self.isInC:
            seq_len *= 2
            self.inc_d1 = InnerComp(user_emb_dim,bs,threshold1)
            self.inc_d2 = InnerComp(user_emb_dim,bs,threshold1)
        if self.isItC: 
            self.itc_d1 = InterComp(user_emb_dim,bs,threshold2)
            self.itc_d2 = InterComp(user_emb_dim,bs,threshold2)
        self.sac1 = Log2feats(user_length, user_emb_dim, item_length, item_emb_dim, seq_len, hid_dim)
        self.sac2 = Log2feats(user_length, user_emb_dim, item_length, item_emb_dim, seq_len, hid_dim)
        # self.down1 = nn.Linear(user_emb_dim*2,user_emb_dim)
        # self.down2 = nn.Linear(user_emb_dim*2,user_emb_dim)
        self.predictModule = predictModule(user_emb_dim,hid_dim)
        self.isDR = isDR
        if self.isDR:
            self.predict_ips = predictModule(user_emb_dim,hid_dim)
            self.predict_gfunc = predictModule(user_emb_dim,hid_dim)

    def forward(self, u_node,i_node,neg_samples,seq_d1,seq_d2,long_tail_mask_d1,long_tail_mask_d2,isTrain=True): # for training        
        # user_spf1, user_spf2 = self.user_emb_layer(u_node)
        i_feat = self.item_emb_layer(i_node).unsqueeze(1)
        neg_samples_feat = self.item_emb_layer(neg_samples)
        seq_d1_feat = self.item_emb_layer(seq_d1)
        seq_d2_feat = self.item_emb_layer(seq_d2)
        if self.isInC:
            seq_d1_feat = self.inc_d1(seq_d1_feat)
            seq_d2_feat = self.inc_d2(seq_d2_feat)
        seq_d1_feat = self.sac1(seq_d1_feat) 
        seq_d2_feat = self.sac2(seq_d2_feat) 
        if self.isItC:
            seq_d1_init = seq_d1_feat
            seq_d2_init = seq_d2_feat
            seq_d1_feat = self.itc_d1(seq_d1_init,seq_d2_init)
            seq_d2_feat = self.itc_d2(seq_d2_init,seq_d1_init)
        u_feat_enhance_d1  = torch.mean(seq_d1_feat,1)
        # print(u_feat_enhance_d1.shape)
        u_feat_enhance_d2  = torch.mean(seq_d2_feat,1)
        i_feat = torch.cat((i_feat,neg_samples_feat),1)   
        if self.isDR:
            logits_d1, logits_d2 = self.predictModule(u_feat_enhance_d1, u_feat_enhance_d2, i_feat)
            logits_ips_d1, logits_ips_d2 = self.predict_ips(u_feat_enhance_d1, u_feat_enhance_d2, i_feat)
            logits_gfunc_d1, logits_gfunc_d2 = self.predict_gfunc(u_feat_enhance_d1, u_feat_enhance_d2, i_feat)
            return logits_d1, logits_d2, logits_ips_d1, logits_ips_d2, logits_gfunc_d1, logits_gfunc_d2
        else:
            logits_d1, logits_d2 = self.predictModule(u_feat_enhance_d1, u_feat_enhance_d2, i_feat)
            return logits_d1, logits_d2# pos_pred, neg_pred

def getBinaryTensor(imgTensor, boundary):
    one = torch.ones_like(imgTensor)
    zero = torch.zeros_like(imgTensor)
    return torch.where(imgTensor > boundary, one, zero)

class InnerComp(nn.Module):

    def __init__(self, user_emb_dim, bs, threshold):
        super(InnerComp, self).__init__()
        self.bs = bs 
        self.threshold = threshold
        self.trans_nn = nn.Linear(user_emb_dim,user_emb_dim)
        self.trans_bs = nn.Linear(bs,1)

    def forward(self,seq):
        # seq : [b,n,d]
        seq_self = seq
        seq1 = seq.unsqueeze(0).repeat(self.bs,1,1,1)
        seq1 = seq1.permute(0, 1, 3, 2)
        group_bool = torch.matmul(seq_self,seq1)
        group_bool = torch.max(torch.max(group_bool,dim=2)[0],dim=2)[0]
        group_bool = nn.Softmax(dim=1)(group_bool)
        group_bool = getBinaryTensor(group_bool,self.threshold)
        seq1 = seq1 * group_bool.unsqueeze(-1).unsqueeze(-1)
        seq1 = self.trans_nn(seq1.permute(0, 1, 3, 2)).permute(0, 3, 2, 1) #[b,b,n,d]-->[b,d,n,b]
        seq1 = self.trans_bs(seq1).squeeze().permute(0, 2, 1) #[b,d,n,b]-->[b,d,n] -->[b,n,d]
        seq_merge = torch.cat((seq_self,seq1),1)
        return seq_merge

class InterComp(nn.Module):

    def __init__(self, user_emb_dim, bs, threshold):
        super(InterComp, self).__init__()
        self.bs = bs 
        self.trans_nn = nn.Linear(user_emb_dim,user_emb_dim)
        self.trans_bs = nn.Linear(bs,1)
        self.threshold = threshold

    def forward(self,seq_d1,seq_d2):
        # seq : [b,n,d]
        # information : seq_d2 --> seq_d1
        seq_self = seq_d1
        seq1 = seq_d2.unsqueeze(0).repeat(self.bs,1,1,1)
        seq1 = seq1.permute(0, 1, 3, 2)
        group_bool = torch.matmul(seq_self,seq1)
        group_bool = torch.max(torch.max(group_bool,dim=2)[0],dim=2)[0]
        group_bool = nn.Softmax(dim=1)(group_bool)
        group_bool = getBinaryTensor(group_bool,self.threshold)
        seq1 = seq1 * group_bool.unsqueeze(-1).unsqueeze(-1)
        seq1 = self.trans_nn(seq1.permute(0, 1, 3, 2)).permute(0, 3, 2, 1) #[b,b,n,d]-->[b,d,n,b]
        seq1 = self.trans_bs(seq1).squeeze().permute(0, 2, 1) #[b,d,n,b]-->[b,d,n] -->[b,n,d]
        seq_merge = torch.cat((seq_self,seq1),1)
        return seq_merge
