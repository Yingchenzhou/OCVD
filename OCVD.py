# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
import copy
import math
import torch
import torch.nn as nn
from collections import Counter

from pandas import np

from src.tools.tokenizer import BERT_Tokenizer
from torch.nn.utils.rnn import pad_sequence
from .cat_attent import MultiHeadAttention, PositionwiseFeedForward
from .FeatEncoder import VisualFeatEncoder


class QGenModel(nn.Module):
    def __init__(
            self,
            num_wrds,
            wrd_pad_id,
            yesid,
            noid,
            naid,
            wrd_embed_size,
            obj_feat_size,
            lstm_hidden_size,
            cat_embed_size,
            num_bboxs,
            num_glimpses,
            num_cats,
            use_osda_glimpse=True,
            update_pi=True,
            see_one_region_per_q=False,
            pi_pre_tanh=True,
            keep_lstm_state=True,
            **kwargs
    ):
        super(QGenModel, self).__init__()

        self.lstm_hidden_size = lstm_hidden_size
        self.yesid = yesid
        self.noid = noid
        self.naid = naid
        self.vis_fc = VisualFeatEncoder(1024, 8, lstm_hidden_size)

        # Attention glimpse in the original paper
        self.num_glimpses = num_glimpses
        self.num_bboxs = num_bboxs

        self.see_one_region_per_q = see_one_region_per_q
        self.use_osda_glimpse = use_osda_glimpse
        self.update_pi = update_pi
        self.pi_pre_tanh = pi_pre_tanh
        self.keep_lstm_state = keep_lstm_state
        self.n_head = 4
        self.tokenizer = BERT_Tokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case=True)

        print(
            "[INFO] QGen: 1 reg. per q: {} | use glimpse: {} | update pi: {} | pi_pre_tanh: {} | keep stat.: {}".format(
                self.see_one_region_per_q, self.use_osda_glimpse, self.update_pi, self.pi_pre_tanh, self.keep_lstm_state
            ))

        self.img_mlp = nn.Sequential(
            nn.Linear(obj_feat_size, wrd_embed_size),
            nn.ReLU()  # TODO: swish activation
        )

        if see_one_region_per_q:
            self.onehot = nn.Embedding.from_pretrained(torch.eye(num_bboxs), freeze=True)
            self.post_mlp = nn.Sequential(
                nn.Linear(2 * wrd_embed_size, wrd_embed_size),
                nn.ReLU()
            )
            vis_repr_out_dim = wrd_embed_size
        elif use_osda_glimpse:
            self.attn_glimpse = nn.Sequential(
                nn.Linear(num_bboxs * wrd_embed_size, num_glimpses),
                nn.Softmax(dim=1)
            )
            vis_repr_out_dim = num_glimpses * wrd_embed_size
        else:
            vis_repr_out_dim = wrd_embed_size

        self.wrd_embed = nn.Embedding(
            num_wrds, wrd_embed_size, padding_idx=wrd_pad_id)
        self.cat_embed = nn.Embedding(
            num_cats, cat_embed_size, padding_idx=0)
        self.lstm = nn.LSTM(
            wrd_embed_size + vis_repr_out_dim + cat_embed_size,
            lstm_hidden_size, batch_first=True, bidirectional=False)
        self.attnlinear = nn.Linear(self.n_head, 1)
        d_inner = 512
        d_k = 256
        d_v = 256
        dropout = 0.2
        self.slf_attn = MultiHeadAttention(self.n_head, wrd_embed_size, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(wrd_embed_size, d_inner, dropout=dropout)
        self.vis_attn = MultiHeadAttention(self.n_head, wrd_embed_size, d_k, d_v, dropout=dropout)
        if wrd_embed_size != lstm_hidden_size:
            self.ans_proj = nn.Linear(wrd_embed_size, lstm_hidden_size)
        self.proj = nn.Linear(lstm_hidden_size, num_wrds)
        self.softmax = nn.Softmax(dim=-1)
        # TODO: Make no sense
        self.pi_obj_proj = nn.Linear(wrd_embed_size, wrd_embed_size)
        self.pi_wrd_proj = nn.Linear(2 * wrd_embed_size, wrd_embed_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    # def att_obj_cat(self, pi, a_emb, hidden_state, obj_repr, input_token=False, return_logits=False):

    def refresh_pi(self, pi, a_emb, hidden_state, obj_repr, noid, naid, find_cat, not_find_cat,cat_len,
                   input_token=True, return_logits=False):
        # a_emb[bs, 512] hidden_state[bs, 512] obj_repr[bs,num_box,512]
        for i in torch.nonzero(not_find_cat):
            if a_emb[i] == naid or a_emb[i] == noid:
                find_cat[i] = find_cat[i] + 1
                find_cat[i]=min(find_cat[i],cat_len[i]-1)
            else:
                not_find_cat[i] = 0

        if input_token:
            a_emb = self.wrd_embed(a_emb)
            # pi = pi.clone()
            # update pi
        if a_emb.size(-1) != hidden_state.size(-1):
            a_emb = self.ans_proj(a_emb)
            # (batch_size, 2 * lstm_hidden_size)
        h_a = torch.cat([hidden_state, a_emb], dim=-1)

        logits = torch.bmm(
            self.pi_obj_proj(obj_repr),
            self.pi_wrd_proj(h_a).unsqueeze(-1))
        if self.pi_pre_tanh:
            logits = self.tanh(logits)
        logits = logits.squeeze(-1)
        # logits = logits / math.sqrt(hidden_state.size(-1))

        _pi = self.softmax(logits / math.sqrt(hidden_state.size(-1)))
        # print(_pi[0])

        # ALPHA = 0.8
        # pi = (1 - ALPHA) * pi + ALPHA * _pi
        # TODO: Weird
        pi = pi * _pi
        # Sum norm
        pi = pi / pi.sum(dim=-1).unsqueeze(-1).float()

        if return_logits:
            return pi, logits, not_find_cat, find_cat
        else:
            return pi,not_find_cat, find_cat,

    def self_diff_attention(self, obj_repr, pi, bboxs_mask=None):
        num_bboxs = obj_repr.size(1)
        obj_feat_dim = obj_repr.size(-1)

        # Object-self Difference Attention (OsDA)
        # diff (bs, num_bboxs, num_bboxs, obj_feat_dim)
        diff = obj_repr.view(-1, num_bboxs, 1, obj_feat_dim) - \
               obj_repr.view(-1, 1, num_bboxs, obj_feat_dim)
        diff = obj_repr.unsqueeze(2) * diff
        diff = diff.view(-1, num_bboxs, num_bboxs * obj_feat_dim)
        logits = self.attn_glimpse[0](diff)
        if bboxs_mask is not None:
            bboxs_mask = bboxs_mask.unsqueeze(-1).repeat(1, 1, self.num_glimpses)
            logits[~bboxs_mask] = -1e10

        # (batch_size, num_glimpses, num_bboxs)
        weight = self.attn_glimpse[1](logits).transpose(1, 2)
        vis_repr = torch.bmm(weight, obj_repr)
        vis_repr = vis_repr.view(-1, self.num_glimpses * obj_feat_dim)

        return vis_repr

    # Forward w/ teacher forcing
    def forward_sentence(self, wrd_emb, wrd_seq_len, vis_repr, last_state, cat):
        """
        wrd_emb: (batch_size, len of input question (padded), wrd_embed_size)
        wrd_seq_len: (batch_size)
        vis_repr: (batch_size, num_glimpses * wrd_embed_size)
        """
        cat_emb = self.cat_embed(cat.squeeze())
        lstm_input = torch.cat(
            [wrd_emb, vis_repr.unsqueeze(1).repeat(1, wrd_emb.size(1), 1),
             cat_emb.unsqueeze(1).repeat(1, wrd_emb.size(1), 1)],
            dim=-1)
        lstm_input = nn.utils.rnn.pack_padded_sequence(
            lstm_input, wrd_seq_len, batch_first=True, enforce_sorted=False)
        lstm_hidden, state = self.lstm(lstm_input, tuple(last_state))
        # (batch_size, seq_len, lstm_hidden_size)
        lstm_hidden, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_hidden, batch_first=True, total_length=wrd_emb.size(1))
        state = torch.stack(state)
        return lstm_hidden, state

    # Forward w/ teacher forcing
    def forward_dialog(self, questions, question_len, answers, img_feats, bboxs, category, cat_len,pi=None,
                       bboxs_mask=None,
                       end_turn=None, cat_input=None):
        obj_feats = self.vis_fc(img_feats, bboxs)
        batch_size = obj_feats.size(0)
        num_bboxs = obj_feats.size(1)
        device = obj_feats.device
        max_num_turns = questions.size(1)
        # find_cat = np.zeros(batch_size,dtype=int).reshape(-1, 1)
        find_cat = torch.zeros(batch_size).long().to(device)
        q_emb = self.wrd_embed(questions)

        entropy = torch.zeros(batch_size).to(device)
        result_logits = []
        # lstm的输入(h_0, c_0),h_0=(1,bs,hiddensize)=c_0,所以这里多了一维度2
        last_state = torch.zeros(2, 1, batch_size, self.lstm_hidden_size).to(device)
        # if cat_sort is None:
        #     cat_sort = []
        #     for i in range(batch_size):
        #         sort=list(Counter(category[i]).keys())
        #         for j in range(len(sort)):
        #             sort[j]=self.tokenizer.encode(sort[j])
        #         cat_sort.append(sort)
        # input_cat=torch.tensor([cat_sort[i].pop(0) for i in range(batch_size)])

        # object 概率初始化
        if pi is None:
            pi = torch.ones(batch_size, num_bboxs).to(device)
            true_num_bboxs = num_bboxs
            if bboxs_mask is not None:
                pi[~bboxs_mask] = 0
                true_num_bboxs = bboxs_mask.sum(dim=-1).unsqueeze(1)
            pi = (pi / true_num_bboxs)

        final_pi_logits = torch.zeros_like(pi)
        not_find_cat = torch.ones(batch_size).to(device)

        for t in range(max_num_turns):
            # Update object representations
            # (batch_size, num_bboxs, obj_feat_dim)  # pi.unsqueeze(-1)=[batch_size, num_bboxs,1]
            obj_repr = obj_feats * pi.unsqueeze(-1)
            # 通过Object-self Difference Attention 获得vis_repr最终视觉表示
            vis_repr = self.self_diff_attention(obj_repr, pi, bboxs_mask)

            cat_input = torch.gather(category, dim=1, index=find_cat.view(batch_size, 1))

            # HACK: RNN can not forward w/ len == 0
            # 取出bs个游戏的所有第t个对话长度 长度等于0的对话，将它设置为1
            fake_len = question_len[:, t].clone()
            fake_len[fake_len == 0] = 1
            # LSTM
            # q_emb[:, t]取出第t个游戏的所有对话
            lstm_hidden, state = self.forward_sentence(
                q_emb[:, t],
                fake_len,
                vis_repr,
                last_state,
                cat_input
            )  # lstm_hidden[bs,token数，hidden size](2,6,512) state[2,1,bs,hidden size]
            logits = self.proj(lstm_hidden)  # (batch_size, seq_len, num_wrds)[2, 6, 30522]
            # not_finished=[bs个Ture/Flase] 因为如果是0表明这个bs的对话结束 ，保留还没结束的bs的state
            not_finished = question_len[:, t] != 0
            if self.keep_lstm_state:
                last_state = last_state.clone()
                last_state[:, :, not_finished] = state[:, :, not_finished]
            result_logits.append(logits)
            # update pi
            if t != max_num_turns - 1:
                pi, pi_logits, not_find_cat, find_cat = self.refresh_pi(pi, answers[:, t], last_state[0, 0], obj_repr,
                                                                        self.noid,self.naid, find_cat, not_find_cat,
                                                                        cat_len,return_logits=True)
                entropy[not_finished] = 0.95 * entropy[not_finished] + \
                                        torch.distributions.Categorical(probs=pi).entropy()[not_finished].sum()
                if end_turn is not None:
                    end = t == end_turn
                    final_pi_logits[end] = pi_logits[end]

        result_logits = torch.stack(result_logits).transpose(0, 1)
        return result_logits, final_pi_logits, entropy.sum()

    # w/o teacher forcing
    def generate_word(self, wrd, vis_repr, state,cat):

        cat_emb = self.cat_embed(cat.squeeze())
        batch_size = wrd.size(0)
        wrd_embed = self.wrd_embed(wrd)
        # (batch_size, 1, (num_glimpses+1) * wrd_embed_size)
        lstm_input = torch.cat([wrd_embed, vis_repr,cat_emb], dim=-1).unsqueeze(1)
        lstm_hidden, state = self.lstm(lstm_input, tuple(state))
        lstm_hidden = lstm_hidden.view(batch_size, -1)
        logit = self.proj(lstm_hidden)
        # (2, ...)
        state = torch.stack(state)
        return logit, state

    # w/o teacher forcing
    def generate_sentence(
            self, last_wrd, img_feats, bboxs, eoq_token, eod_token, end_of_dialog, category,find_cat,
            max_q_len,pi=None, last_state=None, greedy=True, bboxs_mask=None):
        last_wrd = last_wrd.clone()
        batch_size = last_wrd.size(0)
        device = last_wrd.device
        # Have generated <eoq> or <eod>
        # Shape: (batch_size)
        finished = end_of_dialog.clone()
        # Shape: (batch_size)
        actual_length = torch.zeros_like(last_wrd).long()
        if last_state is None:
            # (h, c)
            last_state = torch.zeros(2, 1, batch_size, self.lstm_hidden_size).to(device)

        if pi is None:
            pi = torch.ones(batch_size, self.num_bboxs).to(device)
            true_num_bboxs = self.num_bboxs
            if bboxs_mask is not None:
                pi[~bboxs_mask] = 0
                true_num_bboxs = bboxs_mask.sum(dim=-1).unsqueeze(1)
            pi = (pi / true_num_bboxs)

        # UoDR + OsDA
        # init_obj_repr = self.img_mlp(obj_feats)
        obj_feats = self.vis_fc(img_feats, bboxs)
        obj_repr = obj_feats * pi.unsqueeze(-1)
        # 通过Object-self Difference Attention 获得vis_repr最终视觉表示
        vis_repr = self.self_diff_attention(obj_repr, pi, bboxs_mask)
        cat_input = torch.gather(category, dim=1, index=find_cat.view(batch_size, 1))
        _q_tokens = []
        for t in range(max_q_len):
            logit, state = self.generate_word(last_wrd, vis_repr, last_state,cat_input)
            if greedy:
                q_t = logit.argmax(dim=-1)
            else:
                q_t = torch.multinomial(self.softmax(logit), 1).view(-1)

            # Only update those not finished
            updated_indices = torch.logical_not(finished).nonzero().view(-1)  # 对话未结束的bs的索引
            actual_length[updated_indices] = actual_length[updated_indices] + 1
            last_state[:, :, updated_indices] = state[:, :, updated_indices]
            last_wrd[updated_indices] = q_t[updated_indices]
            # _q_tokens.append(last_wrd) # Bad because this "last_wrd" will be altered in-place
            _q_tokens.append(q_t)
            # Update finished flags
            new_end_of_question = (q_t == eoq_token)
            new_end_of_dialog = (q_t == eod_token)
            new_finished_indices = (new_end_of_question | new_end_of_dialog).nonzero().view(-1)
            finished[new_finished_indices] = 1  # bool类型1代表True
            end_of_dialog = end_of_dialog | new_end_of_dialog
            if finished.sum() == batch_size:
                # all finished
                break

        _q_tokens = torch.stack(_q_tokens).transpose(0, 1)
        q_tokens = []
        for _q_tok, act_len in zip(_q_tokens, actual_length):
            q_tokens.append(_q_tok[:act_len])
        return q_tokens, actual_length, last_state, obj_repr, end_of_dialog
