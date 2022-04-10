import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from DataProcess import get_test_loader
from model.decoder import Decoder
from tqdm import tqdm


def compute_number_of_same_element(a, b, device):
    '''
    :param a:  tensor (N,)
    :param b:  tensor (N,)
    :return: number_of_same_element
    '''
    if device == torch.device('cpu'):
        a = a.numpy().tolist()
        b = b.numpy().tolist()
    else:
        a = a.cpu().numpy().tolist()
        b = b.cpu().numpy().tolist()

    same_count = len(set(a) - set(b))
    count = len(a) - same_count

    return count


def compute_BLEU_for_one_sentence(pre, target, device, word2idx):
    '''
    :param pre: (T, D)
    :param target: (T)
    :param device:
    :param word2idx:
    :return: bleu for this sentence
    '''
    pre = F.softmax(pre, dim=1)
    mask = (target != word2idx['<PAD>'])
    pre = pre[mask, :]  # T', D
    target = target[mask]  # T'
    log_prob = torch.log(pre)  # T', D

    pre = torch.multinomial(pre, 1)  # (T',1)
    choice = pre.squeeze(1)  # T'

    # compute sample prob
    sampled_log_prob = log_prob.gather(1, pre)
    sum_log_prob = torch.sum(sampled_log_prob)

    reward = compute_number_of_same_element(choice, target, device) / target.shape[0]

    return sum_log_prob, reward


# def roughly_estimate_bleu_in_batch(pre, target, device, word2idx):
#     N, T, D = pre.shape
#     pre = pre.reshape(N * T, D)
#     target = target.reshape(N * T)
#     return compute_BLEU_for_one_sentence(pre, target, device, word2idx)


class BLEULoss():
    def __init__(self, momentum=0.9, word2idx=None, device=None):
        self.momentum = momentum

        if word2idx is None:
            self.load_word2idx()
        else:
            self.word2idx = word2idx

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.mean_reward = 0

    def forward(self, pre, target, ):
        '''
        :param pre:  logits
        :param target:
        :return:
        '''
        N, T, D = pre.shape
        loss = 0
        for i in range(N):
            now_prob, now_reward = compute_BLEU_for_one_sentence(pre[i, :, :], target[i, :], self.device, self.word2idx)
            # self.update_mean(now_reward)
            loss += now_prob * (now_reward - self.mean_reward)

        loss *= -1

        return loss

    def load_word2idx(self, out_feature_path='./data/features/'):
        self.word2idx = np.load(os.path.join(out_feature_path, 'dictionary.npy'), allow_pickle=True).item()
        return self.word2idx

    def update_mean(self, new_data):
        # print(type(self.mean_reward), type(new_data), type(self.momentum))
        self.mean_reward = self.momentum * self.mean_reward + (1 - self.momentum) * new_data


def estimate_bleu(total_interations=100, momentum=0.99):
    result = 0
    decoder = Decoder(device=torch.device('cpu'))
    decoder.load_model()
    loader, word2idx = get_test_loader()
    loader = iter(loader)

    for i in tqdm(range(total_interations)):
        fc, att, word, name = next(loader)

        pre = decoder.predict_one(fc, att, beam_size=1)
        word = word.squeeze(0)

        now_bleu = compute_number_of_same_element(pre, word, device=torch.device('cpu')) / 20

        result += now_bleu


    return result/total_interations


if __name__ == '__main__':
    estimate_bleu()
