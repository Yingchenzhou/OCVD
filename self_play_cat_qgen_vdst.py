# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
import copy

import torch
import torch.nn as nn
from src.model.category_vdst import QGenModel
from src.model.oracle import OracleModel
from src.model.guesser import GuesserModel
from torch.nn.utils.rnn import pad_sequence
from .FeatEncoder import VisualFeatEncoder


class SelfPlayModel(nn.Module):
    def __init__(
            self,
            yesid,
            noid,
            naid,
            qgen_kwargs,
            oracle_kwargs,
            guesser_kwargs,
    ):
        super(SelfPlayModel, self).__init__()
        self.qgen = None
        self.noid = noid
        self.naid = naid
        if qgen_kwargs is not None:
            self.qgen = QGenModel(yesid=yesid, noid=noid, naid=naid, **qgen_kwargs)
        self.oracle = OracleModel(**oracle_kwargs)
        self.guesser = GuesserModel(**guesser_kwargs)
        self.vis_fc = VisualFeatEncoder(1024, 8, 512)

    def load_player(self, player, path, map_location="cpu"):
        """
        Usage:
            self_play_obj.load_play("guesser", ckpt_path)
        """
        assert player in ['qgen', 'oracle', 'guesser'], \
            "`player` should be one of ('qgen', 'oracle', 'guesser')."
        getattr(self, player).load_state_dict(
            torch.load(path, map_location=map_location)['model']
        )
        return "Load %s from %s" % (player, path)

    def play(self, img_feats, qgen_bboxs, tgt_cat, tgt_bbox, cats, bboxs, bboxs_mask, qgen_cat, cat_len,
             sos_token, pad_token, eoq_token, eod_token,
             answer2id, answer2token, max_q_len, greedy=True, max_turns=8,log=None):
        log=[]
        cat_log=[]

        device = img_feats.device

        batch_size = img_feats.size(0)
        num_bboxs = img_feats.size(1)
        end_of_dialog = torch.zeros(batch_size).bool().to(device)
        last_wrd = torch.zeros(batch_size).fill_(sos_token).long().to(device)
        last_state = torch.zeros(2, 1, batch_size, self.qgen.lstm_hidden_size).to(device)

        pi = (torch.ones(batch_size, num_bboxs) / num_bboxs).to(device)
        dialog = [torch.LongTensor(0).to(device) for _ in range(batch_size)]
        # BS个[]
        q_log = [[] for _ in range(batch_size)]
        # bs个[]
        a_log = [[] for _ in range(batch_size)]
        a_conf_log = [[] for _ in range(batch_size)]
        not_find_cat = torch.ones(batch_size).to(device)
        find_cat = torch.zeros(batch_size).long().to(device)

        for turn in range(max_turns):
            q, q_len, state, obj_repr, end_of_dialog_next = self.qgen.generate_sentence(
                last_wrd, img_feats, qgen_bboxs, eoq_token, eod_token, end_of_dialog,
                qgen_cat,find_cat,max_q_len,pi=pi, last_state=last_state, greedy=greedy
            )

            pad_q = pad_sequence(q, batch_first=True, padding_value=pad_token)
            # HACK: length == 0 can not forward in RNN
            fake_q_len = q_len.clone()
            fake_q_len[q_len == 0] = 1
            a = self.oracle(pad_q, tgt_cat, tgt_bbox, fake_q_len)
            a_confidence = nn.functional.softmax(a, dim=-1)
            a_idx = a.argmax(dim=-1)
            a = oracle_output_to_answer_token(a_idx, answer2id, answer2token)
            for b in range(batch_size):
                if not end_of_dialog[b]:
                    _q = q[b][:q_len[b]] #q_len对话轮数
                    q_log[b].append(_q)
                    dialog[b] = torch.cat([dialog[b], _q])
                if not end_of_dialog_next[b]:
                    _a = a[b].view(-1)
                    a_log[b].append(_a)
                    a_conf_log[b].append(a_confidence[b, a_idx[b]])
                    dialog[b] = torch.cat([dialog[b], _a])

            if end_of_dialog_next.sum().item() == batch_size:
                break
            end_of_dialog = end_of_dialog_next
            last_wrd = a
            # 执行
            if self.qgen.keep_lstm_state:
                last_state = state
            # print(find_cat[:][7])
            # print(qgen_cat[7])
            pi,not_find_cat,find_cat= self.qgen.refresh_pi(pi, a, last_state[0, 0], obj_repr,self.noid,self.naid,
                                      find_cat, not_find_cat,cat_len, input_token=True)

        dial_len = torch.LongTensor([len(dial) for dial in dialog]).to(device)
        dial_pad = pad_sequence(dialog, batch_first=True, padding_value=pad_token)
        guess = self.guesser(dial_pad, dial_len, cats, bboxs, bboxs_mask)
        return guess, dialog, q_log, a_log, a_conf_log,log,cat_log


def oracle_output_to_answer_token(oracle_output, answer2id, answer2token):
    oracle_output = oracle_output.clone()
    yes_indices = oracle_output == answer2id['Yes']
    no_indices = oracle_output == answer2id['No']
    na_indices = oracle_output == answer2id['N/A']
    oracle_output[yes_indices] = answer2token['Yes']
    oracle_output[no_indices] = answer2token['No']
    oracle_output[na_indices] = answer2token['N/A']
    return oracle_output
