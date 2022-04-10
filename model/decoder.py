import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import io
import numpy as np


def find_k_max(tensor, k):
    '''
    :param tensor: (N,)
    :param k:
    :return: indices (k,),  value (k,)
    '''
    indices = []
    x = tensor.clone()
    for i in range(k):
        _, now_index = torch.max(x, dim=0)
        indices.append(now_index)
        x[now_index] -= float('inf')

    indices = torch.tensor(indices, dtype=torch.int64)
    value = tensor[indices]

    return indices, value


class Decoder(nn.Module):
    def __init__(self,
                 word2idx=None,
                 image_feature_num=14 * 14,
                 original_image_feature_size = 2048,
                 image_feature_size=512,
                 att_lstm_dim=512,
                 lan_lstm_dim=512,
                 dropout=0.5,
                 embed_dim=200,
                 max_length=20,
                 att_hid_dim=512,
                 device=torch.device('cuda'),
                 ):
        '''
        :param image_feature_size: out put dim from encoder. (14,14 440)

        by the way, loader's output:
        torch.Size([1, 440]) torch.Size([1, 440, 14, 14]) torch.Size([1, 5, 20])
        '''
        super(Decoder, self).__init__()
        # store parameters
        self.device = device
        self.att_lstm_dim = att_lstm_dim
        self.lan_lstm_dim = lan_lstm_dim
        self.att_hid_dim = att_hid_dim
        self.word2idx = word2idx
        if self.word2idx is None:
            self.load_word2idx()
        self.max_length = max_length

        #image feature things
        self.fc_embed = nn.Sequential(
            nn.Linear(original_image_feature_size, image_feature_size),
            nn.ReLU(),
            nn.Dropout(p = dropout)
        )
        self.att_embed = nn.Sequential(
            nn.Linear(original_image_feature_size, image_feature_size),
            nn.ReLU(),
            nn.Dropout(p = dropout)
        )

        # define model
        self.total_words = len(self.word2idx)
        self.embedding = nn.Sequential(
            nn.Embedding(self.total_words, embed_dim),
            nn.ReLU(),
            nn.Dropout(p = dropout)
        )
        self.attention_lstm = nn.LSTMCell(input_size=lan_lstm_dim + image_feature_size + embed_dim,
                                          hidden_size=att_lstm_dim)

        # language_lstm things
        self.language_lstm = nn.LSTMCell(input_size=image_feature_size + att_lstm_dim ,
                                         hidden_size=lan_lstm_dim)



        # final out
        self.classifier = nn.Sequential(
            nn.Dropout(p = dropout),
            nn.Linear(lan_lstm_dim, self.total_words),
        )

        # attention parameters
        self.W_va = nn.Linear(image_feature_size, att_hid_dim)
        self.W_ha = nn.Linear(att_lstm_dim, att_hid_dim)
        self.W_a = nn.Linear(att_hid_dim, 1)

        # load

    def attention_lstm_step(self, h1, c1, h2, v_bar, prev_out_word_idx):
        '''
        :param h1: (N,att_lstm_dim)
        :param c1: (N, att_lstm_dim)
        :param c1: (N, att_lstm_dim)
        :param h2: (N, lan_lstm_dim)
        :param v_bar: (N, img_feature_dim)
        :param prev_out_word_idx:(N,), len(xxx.shape) = 1
        :return:
        '''
        embeded = self.embedding(prev_out_word_idx)  # N, embed dim
        input = torch.cat((h2, v_bar, embeded), dim=1)  # N, K

        h, c = self.attention_lstm(input, (h1, c1))
        return h, c

    def language_lstm_step(self, h2, c2, v_hat, h1, prev_out_word_idx):
        '''
        :param h2: (N, lan_lstm_dim)   tuple of 3
        :param c2: (N, lan_lstm_dim)   tuple of 3
        :param v_hat: (N, img_feature_dim)
        :param h1: (N,att_lstm_dim)
        :return:
        '''
 #       embeded = self.embedding(prev_out_word_idx)
        input = torch.cat((v_hat, h1,), dim=1)
        h, c = self.language_lstm(input, (h2, c2))

        return h, c

    def attention_step(self, att_image_features, h1, ):
        '''
        :param att_image_features: (N, feature_num, img_feature_dim)
        :param h1: (N,att_lstm_dim)
        :return: (N, img_feature_dim)
        '''
        # (N, feature_num, att_hidden)              (N,att_hid_dim)
        new_att_image_features = self.W_va(att_image_features)  # (N, feature_num, att_hidden)
        inside = new_att_image_features + self.W_ha(h1).unsqueeze(1).expand_as(
            new_att_image_features)  # (N, feature_num, att_hidden)
        inside = torch.tanh(inside)  # (N, feature_num, att_hidden)
        att_scores = self.W_a(inside).squeeze(2)  # N, feature_num
        att_scores = F.softmax(att_scores, dim=1).unsqueeze(1)  # N, 1, feature_num
        # print(att_scores.shape, att_image_features.shape)
        result = torch.bmm(att_scores, att_image_features).squeeze(1)
        return result

    def teacher_forcing_forward(self, fc_img_feature, att_img_feature, word=None):
        '''
        :param fc_img_feature: (N, img_feature_dim)
        :param att_img_feature: (N,img_feature_dim, 14, 14)
        :param word:ground truth. (N, T)
        :return: scores (N, total_step, scores)
        '''
        # (N, 14*14, img_feature_dim)
        att_img_feature = att_img_feature.permute(0, 2, 3, 1)
        att_img_feature = att_img_feature.view(att_img_feature.shape[0], -1, att_img_feature.shape[3])
        v_bar = self.fc_embed(fc_img_feature)
        att_img_feature = self.att_embed(att_img_feature)

        # need h1 c1 h2 c2
        N = att_img_feature.shape[0]
        h2, c2 = torch.zeros(N, self.lan_lstm_dim, device=self.device), torch.zeros(N, self.lan_lstm_dim,
                                                                                      device=self.device)
        h1, c1 = torch.zeros(N, self.att_lstm_dim, device=self.device), torch.zeros(N, self.lan_lstm_dim,
                                                                                    device=self.device)


        prev_out_word_idx = torch.zeros(N, dtype=torch.int32, device=self.device)
        prev_out_word_idx += self.word2idx['<START>']
        scores = torch.zeros(N, self.max_length, self.total_words, device=self.device)

        for step in range(self.max_length):

            h1, c1 = self.attention_lstm_step(h1, c1, h2, v_bar, prev_out_word_idx)

            # print(att_img_feature.device, h1.device)
            v_hat = self.attention_step(att_img_feature, h1)
            h2, c2 = self.language_lstm_step(h2, c2, v_hat, h1, prev_out_word_idx)

            # (N, total_words)
            now_score = self.classifier(h2)
            scores[:, step, :] = now_score

            _, prev_out_word_idx = torch.max(now_score, dim=1)
            # use ground truth to train
            if word is not None:
                prev_out_word_idx = word[:, step]

        return scores

    def save_model(self):
        torch.save(self.state_dict(), 'decoder.ckpt')

    def load_model(self):
        if os.path.exists('decoder.ckpt'):
            ckpt = torch.load('decoder.ckpt', map_location=torch.device('cpu'))
            self.load_state_dict(ckpt)
            print('managed to load the model')

        print('_________________________________________________')

    def load_word2idx(self, out_feature_path='./data/features/'):
        self.word2idx = np.load(os.path.join(out_feature_path, 'dictionary.npy'), allow_pickle=True).item()
        return self.word2idx

    def forward_step_for_beam_search(self, h1, c1, h2, c2, v_bar, att_img_feature, prev_out_word_idx, ):
        h1, c1 = self.attention_lstm_step(h1, c1, h2, v_bar, prev_out_word_idx)
        # print(att_img_feature.device, h1.device)
        v_hat = self.attention_step(att_img_feature, h1)
        h2, c2 = self.language_lstm_step(h2, c2, v_hat, h1, prev_out_word_idx)

        # (N, total_words)
        now_score = self.classifier(h2)

        # _, prev_out_word_idx = torch.max(now_score, dim=1)
        now_score = F.softmax(now_score, dim=1)
        now_score = torch.log(now_score)

        return now_score, (h1, c1, h2, c2),

    def predict_one(self, fc_img_feature, att_img_feature, beam_size=10):
        '''
        batch size should be 1!!!
        :param fc_img_feature: (1, img_feature_dim)
        :param att_img_feature: (1,img_feature_dim, 14, 14)
        :param beam_size:
        :return:(T,)
        '''
        # (1, 14*14, img_feature_dim)
        att_img_feature = att_img_feature.permute(0, 2, 3, 1)
        att_img_feature = att_img_feature.view(att_img_feature.shape[0], -1, att_img_feature.shape[3])
        v_bar = self.fc_embed(fc_img_feature)
        att_img_feature = self.att_embed(att_img_feature)

        # need h1 c1 h2 c2
        # K, D
        h2, c2 = torch.zeros(1, self.lan_lstm_dim, device=self.device), torch.zeros(1, self.lan_lstm_dim,
                                                                                      device=self.device)
        h1, c1 = torch.zeros(1, self.att_lstm_dim, device=self.device), torch.zeros(1,
                                                                                    self.lan_lstm_dim,
                                                                                    device=self.device)
        prev_out_word_idx = torch.zeros(1, dtype=torch.int32, device=self.device)
        prev_out_word_idx += self.word2idx['<START>']

        scores = torch.zeros(beam_size)  # K,

        choice_record = torch.zeros((self.max_length, beam_size), dtype=torch.int64)

        # 保存索引
        r_scores = scores
        r_choice_record = choice_record

        # the first iteration

        now_scores, cache = self.forward_step_for_beam_search(h1, c1, h2, c2, v_bar, att_img_feature,
                                                              prev_out_word_idx)

        h1, c1, h2, c2 = cache
        indices, selected_scores = find_k_max(now_scores.squeeze(0), beam_size)
        prev_out_word_idx = indices
        choice_record[0, :] = indices
        scores[:] = selected_scores
        h1 = h1.expand(beam_size, h1.shape[1])
        c1 = c1.expand(beam_size, c1.shape[1])
        h2 = h2.expand(beam_size, h2.shape[1])
        c2 = c2.expand(beam_size, c2.shape[1])

        v_bar = v_bar.expand(beam_size, v_bar.shape[1])
        att_img_feature = att_img_feature.expand(beam_size, att_img_feature.shape[1], att_img_feature.shape[2])

        for step in range(1, self.max_length):
            # scores (K, out_dim)       cache   (K,hid_dim)
            now_scores, cache = self.forward_step_for_beam_search(h1, c1, h2, c2, v_bar, att_img_feature,
                                                                  prev_out_word_idx)
            # compute total prob, now_scores (K, out_dim)
            now_scores = now_scores + scores.unsqueeze(1).expand_as(now_scores)

            K, out_dim = now_scores.shape
            now_scores = now_scores.reshape(-1)  # (K*out_dim,)
            indices, selected_scores = find_k_max(now_scores, beam_size)  # indices (K,)

            prev_word_idx = indices // out_dim  # (K,)       索引 选哪个
            next_word_idx = indices % out_dim  # (K, )       输出词的idx

            # update record and scores
            choice_record[step, :] = next_word_idx
            choice_record[step - 1, :] = choice_record[step - 1, prev_word_idx]
            scores = scores[prev_word_idx] + selected_scores

            # update idx   更新所有变量索引
            prev_out_word_idx = next_word_idx

            # update cache
            h1, c1, h2, c2 = cache  # (K, ?)
            h1 = h1[prev_word_idx, :]
            c1 = c1[prev_word_idx, :]
            h2 = h2[prev_word_idx, :]
            c2 = c2[prev_word_idx, :]


        # print(r_choice_record)
        final_choice, _ = find_k_max(r_scores, 1)
        choice_record = r_choice_record[:, final_choice]

        return choice_record.squeeze(1)
