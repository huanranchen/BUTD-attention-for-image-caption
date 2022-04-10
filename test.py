import torch
import torch.nn as nn
from DataProcess import get_test_loader, get_loader
from model.decoder import Decoder
import os
from PIL import Image
import numpy as np
from model.encoder import Encoder
import skimage


def invert_dict(d):
    return dict([(v, k) for (k, v) in d.items()])


def test():
    loader, word2idx = get_test_loader()
    idx2word = invert_dict(word2idx)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'our device is {device}!!!')
    model = Decoder(word2idx=word2idx, device=device).to(device)
    model.load_model()
    for fc, att, word,name in loader:
        with torch.no_grad():
            fc = fc.to(device)
            att = att.to(device)
            word = word.to(device)
            scores = model.teacher_forcing_forward(fc, att)
            scores[:, :, word2idx['<PAD>']] -= float('inf')
        # print(scores.shape)  torch.Size([1, 20, 6783])

        _, result = torch.max(scores, dim=2)
        result = result.squeeze(0)  # 20
        print(f'image name is {name}')
        for i in range(result.shape[0]):
            now_word = idx2word[result[i].int().item()]
            print(now_word)
            # if now_word == '<EOF>':
            #     break
        print('__________________________________________')
        word = word.squeeze(0)
        for i in range(word.shape[0]):
            now_word = idx2word[word[i].int().item()]
            print(now_word)
            if now_word == '<EOF>':
                break
        assert False



def test_on_train():
    _, loader, word2idx = get_loader(batch_size=1, require_name=True)
    idx2word = invert_dict(word2idx)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'our device is {device}!!!')
    model = Decoder(word2idx=word2idx, device=device).to(device)
    model.load_model()
    for fc, att, word,name in loader:
        fc = fc.to(device)
        att = att.to(device)
        word = word.to(device)
        scores = model.predict_one(fc, att,)
        result = scores
        print(f'image name is {name}')
        for i in range(result.shape[0]):
            now_word = idx2word[result[i].int().item()]
            print(now_word)
            # if now_word == '<EOF>':
            #     break
        print('__________________above is beam search________________________')




        scores=model.teacher_forcing_forward(fc, att)
        scores[:, :, word2idx['<PAD>']] -= float('inf')
        _, result = torch.max(scores, dim=2)
        result = result.squeeze(0)  # 20
        print(f'image name is {name}')
        for i in range(result.shape[0]):
            now_word = idx2word[result[i].int().item()]
            print(now_word)
            # if now_word == '<EOF>':
            #     break
        print('________________above is greedy decoding__________________________')



        assert False
        word = word.squeeze(0)
        for i in range(word.shape[0]):
            now_word = idx2word[word[i].int().item()]
            print(now_word)
            if now_word == '<EOF>':
                break
        assert False


def test_outside_image(filepath = 'outsideimg.jpg'):
    img = skimage.io.imread(filepath)
    print()
    encoder = Encoder()
    decoder = Decoder(device=torch.device('cpu'))
    fc, att = encoder(img)
    fc = fc.unsqueeze(0)
    att=att.unsqueeze(0)
    decoder.load_model()
    word2idx = decoder.load_word2idx()
    idx2word = invert_dict(word2idx)
    scores = decoder.teacher_forcing_forward(fc, att,)
    _, result = torch.max(scores, dim=2)
    result = result.squeeze(0)  # 20
    for i in range(result.shape[0]):
        now_word = idx2word[result[i].int().item()]
        print(now_word)
        if now_word == '<EOF>':
            break
    assert False


if __name__ == '__main__':
   test()
