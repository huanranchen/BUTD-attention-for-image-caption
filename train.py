import os.path
from test import invert_dict
import torch
import torch.nn as nn
from model.decoder import Decoder
from DataProcess import get_loader
from tqdm import tqdm
from criterions.BLEU import BLEULoss


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def compute_cross_entropy_loss(pre, target, device, word2idx):
    '''
    :param target: (N, total_step)
    :param pre: (N, total_step, scores)
    :return: total loss
    '''
    N, T, D = pre.shape
    mask = (target != word2idx['<PAD>'])  # N, T
    target = target.reshape(N * T)
    mask = mask.reshape(N * T)
    pre = pre.reshape(N * T, D)

    criterion = nn.CrossEntropyLoss().to(device)

    # idx2word = invert_dict(word2idx)
    # target = target[mask]
    # print(target)
    # for i in range(target.shape[0]):
    #     print(idx2word[target[i].item()])
    # assert False

    loss = criterion(pre[mask, :], target[mask])

    return loss


def teacher_forcing_train(batch_size=32,
                          total_epoch=100,
                          lr=4e-4,
                          weight_decay=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, valid_loader, dic = get_loader(batch_size=batch_size)
    model = Decoder(word2idx=dic, device=device).to(device)
    if os.path.exists('decoder.ckpt'):
        model.load_state_dict(torch.load('decoder.ckpt', map_location=torch.device('cpu')))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(total_epoch):
        # train part
        train_loss = 0
        valid_loss = 0
        model.train()
        for fc, att, word in tqdm(train_loader):
            fc = fc.to(device)
            att = att.to(device)
            word = word.to(device)
            scores = model.teacher_forcing_forward(fc, att, word)
            loss = compute_cross_entropy_loss(scores, word, device, dic)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()


            clip_gradient(optimizer, 0.1)

            # print(model.language_lstm.weight_hh.grad)
            #
            # for i in model.language_lstm.parameters():
            #     print(torch.max(i.grad), i.data)
            #
            # assert False

            optimizer.step()

        train_loss /= len(train_loader)
        if epoch % 2 == 0:
            print(f'epoch {epoch}, training loss = {train_loss}')

        model.eval()
        for fc, att, word in tqdm(valid_loader):
            with torch.no_grad():
                fc = fc.to(device)
                att = att.to(device)
                word = word.to(device)
                scores = model.teacher_forcing_forward(fc, att, word)
                loss = compute_cross_entropy_loss(scores, word, device, dic)
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        if epoch % 2 == 0:
            print(f'epoch {epoch}, validating loss = {valid_loss}')
            model.save_model()


def overfit_small_dataset(batch_size=32,
                          total_epoch=100,
                          lr=4e-4,
                          weight_decay=1e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, valid_loader, dic = get_loader(batch_size=batch_size)
    model = Decoder(word2idx=dic, device=device).to(device)
    if os.path.exists('decoder.ckpt'):
        model.load_state_dict(torch.load('decoder.ckpt', map_location=torch.device('cpu')))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(total_epoch):
        # train part
        train_loss = 0
        for fc, att, word in tqdm(valid_loader):
            fc = fc.to(device)
            att = att.to(device)
            word = word.to(device)
            optimizer.zero_grad()
            scores = model.teacher_forcing_forward(fc, att, word)
            loss = compute_cross_entropy_loss(scores, word, device, dic)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        print(f'epoch {epoch}, training loss = {train_loss}')
        model.save_model()


def RL_train(batch_size=32,
             total_epoch=100,
             lr=4e-4,
             weight_decay=1e-5,
             rl_criterion='BLEU'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, valid_loader, dic = get_loader(batch_size=batch_size)
    model = Decoder(word2idx=dic, device=device).to(device)
    if os.path.exists('decoder.ckpt'):
        model.load_state_dict(torch.load('decoder.ckpt', map_location=torch.device('cpu')))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if rl_criterion == 'BLEU':
        criterion = BLEULoss(device=device)

    for epoch in range(total_epoch):
        # train part
        train_loss = 0
        valid_loss = 0
        model.train()
        for fc, att, word in tqdm(train_loader):
            fc = fc.to(device)
            att = att.to(device)
            word = word.to(device)
            optimizer.zero_grad()
            scores = model.teacher_forcing_forward(fc, att, word)
            loss = criterion.forward(scores, word)
            train_loss += loss.item()
            loss.backward()

            # print(model.language_lstm.weight_hh.grad)
            #
            # for i in model.language_lstm.parameters():
            #     print(torch.max(i.grad), i.data)
            #
            # assert False

            optimizer.step()

        train_loss /= len(train_loader)
        if epoch % 2 == 0:
            print(f'epoch {epoch}, training loss = {train_loss}')

        model.eval()
        for fc, att, word in tqdm(valid_loader):
            with torch.no_grad():
                fc = fc.to(device)
                att = att.to(device)
                word = word.to(device)
                scores = model.teacher_forcing_forward(fc, att)
                loss = criterion.forward(scores, word, )
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        if epoch % 2 == 0:
            print(f'epoch {epoch}, validating loss = {valid_loss}')
            model.save_model()


if __name__ == '__main__':
    teacher_forcing_train(batch_size=32)
