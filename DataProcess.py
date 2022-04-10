import h5py
import torch
from dataUtils import *
from model.encoder import Encoder
import os
import skimage
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import random


class Embedding():
    def __init__(self, max_len=20):
        self.max_len = max_len
        self.word2idx = {}
        self.word2idx['<START>'] = 0
        self.word2idx['<PAD>'] = 1
        self.word2idx['<UNK>'] = 2
        self.word2idx['<EOF>'] = 3
        pass

    def get_one_word_idx(self, x):
        '''
        :param x: str
        :return:
        '''
        if x in self.word2idx:
            return self.word2idx[x]

        self.word2idx[x] = len(self.word2idx)
        return self.word2idx[x]

    def get_one_sentence_to_idx(self, x):
        '''
        :param x: a list of words with lower case:
    ['a', 'man', 'in', 'street', ............]
        :return:  numpy array (1,2,3,..)
        '''
        result = []
        for word in x:
            result.append(self.get_one_word_idx(word))

        while len(result) > self.max_len:
            result.pop(-1)

        while len(result) < self.max_len:
            result.append(self.word2idx['<PAD>'])

        return np.array(result, dtype=np.float64)

    def get_a_list_of_sentence_to_idx(self, x):
        '''
        :param x:  a list of sentence
        :return:
        '''
        result = []
        for sentence in x:
            t = split_sentence_into_words(sentence)
            result.append(self.get_one_sentence_to_idx(t))

        return np.stack(result)

    def save_dictionary(self, save_path='./data/features/dictionary.npy'):
        np.save(save_path, self.word2idx)

        # when loading, use dic = np.load(path, allow_pickle = True)


def extract_feature(image_path='./data/Flickr8k_images/images/',
                    lemma_path='./data/Flickr8k_text/Flickr8k.lemma.token.txt',
                    out_feature_path='./data/features/'):
    embed = Embedding()
    attribute_dic = read_into_dic(file=lemma_path)
    encoder = Encoder()
    imgs = os.listdir(image_path)
    imgs.sort()
    print('I am doing data processing, the process:')
    if not os.path.exists(out_feature_path):
        os.makedirs(out_feature_path)

    with h5py.File(os.path.join(out_feature_path, 'fc_feature.h5'), 'w') as fc_f, \
            h5py.File(os.path.join(out_feature_path, 'att_feature.h5'), 'w') as att_f, \
            h5py.File(os.path.join(out_feature_path, 'word.h5'), 'w') as word_f:
        for image_name in tqdm(imgs):
            img = skimage.io.imread(os.path.join(image_path, image_name))
            with torch.no_grad():
                fc, att = encoder.forward(img)

            word = attribute_dic[image_name]
            word = embed.get_a_list_of_sentence_to_idx(word)
            fc_f.create_dataset(image_name, data=fc.float().numpy())
            att_f.create_dataset(image_name, data=att.float().numpy())
            word_f.create_dataset(image_name, data=word)

        # after done collections
        print('Data process has been done!!')
        fc_f.close()
        att_f.close()
        word_f.close()
        embed.save_dictionary()
        print('--------------------------------------------------------------------')


class MyDataSet(Dataset):
    def __init__(self,
                 image_path='./data/Flickr8k_images/images/',
                 out_feature_path='./data/features/',
                 mode='train',
                 image_names_path='./data/features/image_names.npy',
                 require_name=False):
        self.dic = np.load(os.path.join(out_feature_path, 'dictionary.npy'), allow_pickle=True).item()
        self.requre_name = require_name
        if os.path.exists(image_names_path):
            self.image_names = np.load(image_names_path)
            self.image_names = self.image_names.tolist()
        else:
            self.image_names = os.listdir(image_path)
            np.save(image_names_path, np.array(self.image_names))

        self.fc_f = h5py.File(os.path.join(out_feature_path, 'fc_feature.h5'), 'a')
        self.att_f = h5py.File(os.path.join(out_feature_path, 'att_feature.h5'), 'a')
        self.word_f = h5py.File(os.path.join(out_feature_path, 'word.h5'), 'a')
        self.mode = mode

        if mode == 'train':
            self.image_names = self.image_names[:7000]

        if mode == 'valid':
            self.image_names = self.image_names[7000:7500]

        if mode == 'test' or mode == 'CIDEr':
            self.image_names = self.image_names[7500:]

    def __getitem__(self, item):
        name = self.image_names[item]

        # N, K
        num_sentences = self.word_f[name][:].shape[0]
        fc = torch.tensor(self.fc_f[name][:]).to(torch.float32)
        att = torch.tensor(self.att_f[name][:]).to(torch.float32)

        if self.mode == 'CIDEr':
            word = torch.from_numpy(np.array([self.word_f[name][:]]))
            word = word.to(torch.int64).squeeze(0)
            return fc, att, word, name

        word = torch.from_numpy(np.array([self.word_f[name][:][random.randint(0, num_sentences - 1)]]))
        word = word.to(torch.int64).squeeze(0)

        if self.mode == 'test' or self.requre_name == True:
            return fc, att, word, name

        return fc, att, word

    def __len__(self):
        return len(self.image_names)

    def get_dic(self):
        return self.dic

    def get_image_names(self):
        return self.image_names


def get_loader(batch_size=128, require_name=False):
    train_set = MyDataSet(require_name=require_name)
    valid_set = MyDataSet(mode='valid', require_name=require_name)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, )

    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, train_set.get_dic()


def get_test_loader(batch_size=1):
    test = MyDataSet(mode='test')

    train_loader = DataLoader(test, batch_size=batch_size, shuffle=True, )

    return train_loader, test.get_dic()


def get_cider_loader(total_image = 500):
    test = MyDataSet(mode='CIDEr')
    loader = DataLoader(test, batch_size=1, shuffle=False)
    return loader, test.get_image_names()


if __name__ == '__main__':
    loader = get_cider_loader()
    for fc, att, word, name in loader:
        print(fc.shape, word.shape, name)
