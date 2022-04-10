import torch
import numpy as np
from tqdm import tqdm
import os
import math


class CIDEr():
    def __init__(self, loader, model, image_names, n_grams=4, ):
        '''
        大致思路：为每个句子构建一个字典。为了方便查找这些句子，为candidate再构建一个name->dic的字典，为ground truth 构建一个name+index->dic的字典
        :param loader: dataloader
        :param model: decoder
        :param image_names: list of image names
        :param n_grams: 4
        '''
        self.n_grams = n_grams
        self.loader = loader
        self.model = model
        self.image_names = image_names

        # class variables
        self.tokens = {}  # n_grams -> token

        self.ground_truth_dic = {}  # image_name ->[dic1,dic2,dic3,dic4,dic5]
        self.candicate_dic = {}  # image_name -> dic

    def synthesize_dic_for_one_sentence(self, x):
        '''
        :param x: tensor [20]
        :return: a dic(token -> times) for this sentence
        '''
        result = {}
        sentence = x.cpu().numpy().tolist()
        for i in range(len(sentence) - self.n_grams):
            now_token = tuple(sentence[i:i + self.n_grams])

            # convert to true token
            if now_token not in self.tokens:
                self.tokens[now_token] = len(self.tokens)
            now_token = self.tokens[now_token]

            # calculate <result> dictionary
            if now_token not in result:
                result[now_token] = 1
            else:
                result[now_token] += 1

        return result

    def process_data(self):
        '''
        compute self.ground_truth_dic and self.candicate_dic
        '''
        print('now we are processing all the candidate and ground truth')

        for fc, att, ground_truth, name in tqdm(self.loader):
            # first process ground truth. ground truth tensor (1, N, D)
            now_ground_truth_dic = []
            ground_truth = ground_truth.squeeze(0)
            for i in range(ground_truth.shape[0]):
                now_ground_truth_dic.append(self.synthesize_dic_for_one_sentence(ground_truth[i]))
            self.ground_truth_dic[name] = now_ground_truth_dic
            # then process candidate
            candidate = self.model.predict_one(fc, att)
            self.candicate_dic[name] = self.synthesize_dic_for_one_sentence(candidate)

        print('now we finish processing !!!!!!!!')
        print('___________________________________________________________________')

    def save_processed_data(self, save_path='./criterions/CIDEr_materials/'):
        '''
        using numpy save self.tokens, candidate dic, ground truth dic
        self.tokens is a dic.
        candidate dic is a dic where all the values are dic
        ground truth dic is a dic where all the values are list [dic1, dic2, dic3, dic4, dic5]
        '''
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(save_path + 'tokens.npy', self.tokens)
        np.save(save_path + 'candidate.npy', self.candicate_dic)
        np.save(save_path + 'ground_truth.npy', self.ground_truth_dic)
        print('managed to save the data !!!!!!!!')
        print('___________________________________________________________________')

    def load_processed_data(self, load_path='./criterions/CIDEr_materials/'):
        '''
        using numpy load self.tokens, candidate dic, ground truth dic
        '''
        self.tokens = np.load(load_path + 'tokens.npy', allow_pickle=True).item()
        self.candicate_dic = np.load(load_path + 'candidate.npy', allow_pickle=True).item()
        self.ground_truth_dic = np.load(load_path + 'ground_truth.npy', allow_pickle=True).item()
        print('managed to load the data !!!!!!!!')
        print('___________________________________________________________________')

    def compute_cider_for_one_sentence(self, x_dic, y_dic):
        '''
        :param x_dic: one candidate dic
        :param y_dic: one ground truth dic
        :return:CIDEr
        '''
        keys = list(set(list(x_dic.keys())) & set(list(y_dic.keys())))
        print(x_dic.keys(), y_dic.keys())
        if len(keys) == 0:
            return 0
        x_vec = []
        y_vec = []
        for key in keys:
            tf = x_dic[key] / sum(list(x_dic.values()))
            idf = sum([key in i for i in list(self.candicate_dic.values())])
            idf = len(self.image_names) / idf
            idf = math.log(idf)
            x_vec.append(tf * idf)

        for key in keys:
            tf = y_dic[key] / sum(list(y_dic.values()))
            idf = 0
            for i in list(self.ground_truth_dic.values()):
                for j in i:
                    if keys in j:
                        idf += 1
                        break
            idf = len(self.image_names) / idf
            idf = math.log(idf)
            y_vec.append(tf * idf)

        x_vec = np.array(x_vec)
        y_vec = np.array(y_vec)

        return (x_vec @ y_vec) / (np.linalg.norm(x_vec) * np.linalg.norm(y_vec))

    def compute_cider_for_one_image(self, name):
        now_ground_truth_dics = self.ground_truth_dic[name]  # list
        now_cancidate = self.candicate_dic[name]

        result = [self.compute_cider_for_one_sentence(now_cancidate, ground_truth) for ground_truth in
                  now_ground_truth_dics]
        result = sum(result) / len(now_ground_truth_dics)

        return result

    def estimate_entire_cider(self):
        result = 0
        for i in self.image_names:
            result += self.compute_cider_for_one_image((i,))

        # result /= len(self.image_names)
        return result
