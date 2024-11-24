import torch
import numpy as np
from torch import nn


path = 'D:/Work/Project/Github/PresRecRF/data'


class PresRecRF(torch.nn.Module):
    def __init__(self, batch_size, embedding_dim, symptom_cnt, herb_cnt, drop_ratio, semantic='Bert', molecular='HSP'):
        super(PresRecRF, self).__init__()
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.semantic = semantic
        self.molecular = molecular

        self.sym_random = torch.nn.Embedding(self.batch_size, self.embedding_dim)
        self.herb_random = torch.nn.Embedding(self.batch_size, self.embedding_dim)

        # 1. 初始语义embedding (1.1 1.2二选一)
        # 1.1 读入bert向量
        if self.semantic == 'Bert':
            self.bert_sym_embedding = nn.Embedding.from_pretrained(
                 torch.from_numpy(np.load(path+'/bert_emb'+'/bert_sym.npy')))

            self.bert_herb_embedding = nn.Embedding.from_pretrained(
                 torch.from_numpy(np.load(path+'/bert_emb'+'/bert_herb.npy')))

        # 1.2 读入GPT3.5向量-240104
        elif self.semantic == 'GPT35':
            self.bert_sym_embedding = nn.Embedding.from_pretrained(
                 torch.as_tensor(torch.from_numpy(np.load(path+'/Ada'+'/symptom_vectors_ada_240104.npy')), dtype=torch.float32))

            self.bert_herb_embedding = nn.Embedding.from_pretrained(
                 torch.as_tensor(torch.from_numpy(np.load(path+'/Ada'+'/herb_vectors_ada_240104.npy')), dtype=torch.float32))

        else:
            self.bert_sym_embedding = nn.Embedding.from_pretrained(
                torch.from_numpy(np.load(path + '/bert_emb' + '/bert_sym.npy')))

            self.bert_herb_embedding = nn.Embedding.from_pretrained(
                torch.from_numpy(np.load(path + '/bert_emb' + '/bert_herb.npy')))

        # 2. 初始结构embedding
        # 2.1 仅药症的embedding
        if self.molecular == 'HS':
            self.sym_embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(torch.from_numpy(np.load(path+'/method'+'/deepwalk-sym.npy')),dtype=torch.float32))
            self.herb_embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(torch.from_numpy(np.load(path+'/method'+'/deepwalk-herb.npy')),dtype=torch.float32))

        # 2.2 补充了分子信息后的embedding
        elif self.molecular == 'HSP':
            self.sym_embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(torch.from_numpy(
                    np.load(path + r'/method/deepwalk-sym_AddG_230911.npy')
                ), dtype=torch.float32))
            self.herb_embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(torch.from_numpy(
                    np.load(path + r'/method/deepwalk-herb_AddG_230911.npy')
                ), dtype=torch.float32))

        else:
            self.sym_embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(torch.from_numpy(np.load(path + '/method' + '/deepwalk-sym.npy')),
                                dtype=torch.float32))
            self.herb_embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(torch.from_numpy(np.load(path + '/method' + '/deepwalk-herb.npy')),
                                dtype=torch.float32))

        #症状
        self.mlp_sym_1 = torch.nn.Linear(self.embedding_dim, 256)
        self.tanh_1 = torch.nn.Tanh()

        self.mlp_sym_2 = torch.nn.Linear(256, 256)
        self.tanh_2 = torch.nn.Tanh()

        self.mlp_sym_3 = torch.nn.Linear(256, self.embedding_dim)
        # self.relu = torch.nn.ReLU()

        # bert向量变化--sym
        if self.semantic == 'Bert':
            self.mlp_bert_1 = torch.nn.Linear(768, 256)
        # # Ada向量变化--sym
        elif self.semantic == 'GPT35':
            self.mlp_bert_1 = torch.nn.Linear(1536, 256)
        else:
            self.mlp_bert_1 = torch.nn.Linear(768, 256)

        self.mlp_bert_2 = torch.nn.Linear(256, self.embedding_dim)

        #药物
        self.mlp_herb_1 = torch.nn.Linear(self.embedding_dim, 256)
        self.tanh_herb_1 = torch.nn.Tanh()

        self.mlp_herb_2 = torch.nn.Linear(256, 256)
        self.tanh_herb_2 = torch.nn.Tanh()

        self.mlp_herb_3 = torch.nn.Linear(256, self.embedding_dim)

        # bert向量变化--药物
        if self.semantic == 'Bert':
            self.mlp_bert_herb_1 = torch.nn.Linear(768, 256)
        # # ada向量变化-药物
        elif self.semantic == 'GPT35':
            self.mlp_bert_herb_1 = torch.nn.Linear(1536, 256)
        else:
            self.mlp_bert_herb_1 = torch.nn.Linear(768, 256)
        self.mlp_bert_herb_2 = torch.nn.Linear(256, self.embedding_dim)

        self.dropout = torch.nn.Dropout(drop_ratio)

        self.mlp_dosage = torch.nn.Linear(herb_cnt, herb_cnt)
        self.mlp_dosage_2 = torch.nn.Linear(herb_cnt, herb_cnt)
        self.mlp_dosage_3 = torch.nn.Linear(herb_cnt, herb_cnt)
        self.relu = torch.nn.ReLU()

    def forward(self, symptom_OH):
        # 1. symptom embedding
        get_sym = symptom_OH  # 128*1804

        bert_sym = self.mlp_bert_1(self.bert_sym_embedding.weight)  # 1804*256
        bert_sym = self.mlp_bert_2(bert_sym)  # 1804*dim-128
        bert_sym = self.dropout(bert_sym)

        bert_herb = self.mlp_bert_herb_1(self.bert_herb_embedding.weight)  # 410*256
        bert_herb = self.mlp_bert_herb_2(bert_herb)  # 410*128

        # # add sym s
        get_sym = torch.mm(get_sym, torch.add(bert_sym, self.sym_embedding.weight))
        # print(get_sym.shape)  # 32*128
        # get_sym = torch.mm(get_sym, bert_sym)  # no struct
        # get_sym = torch.mm(get_sym, torch.add(bert_sym, self.sym_embedding.weight))  # no seman

        sym_emb = self.mlp_sym_1(get_sym)
        sym_emb = self.tanh_1(sym_emb)
        sym_emb = self.dropout(sym_emb)
        sym_emb = self.mlp_sym_2(sym_emb)
        sym_emb = self.tanh_2(sym_emb)
        sym_emb = self.dropout(sym_emb)
        sym_agg = self.mlp_sym_3(sym_emb)  # 64*128 same dim trans mlp,

        # add herb S
        herb_emb = torch.add(bert_herb, self.herb_embedding.weight)
        # print(herb_emb.shape)  # 410*128
        # herb_emb = bert_herb  # no struct
        # herb_emb = self.herb_embedding.weight  # no seman

        herb_emb = self.mlp_herb_1(herb_emb)
        herb_emb = self.tanh_herb_1(herb_emb)
        herb_emb = self.mlp_herb_2(herb_emb)
        herb_emb = self.tanh_herb_2(herb_emb)
        herb_emb = self.mlp_herb_3(herb_emb)  # 64*128 same dim trans mlp,

        # 4. judge herb
        judge_herb = torch.mm(sym_agg, herb_emb.T)  # 64*128 * 128*410 => 64*721

        # 5. dosage
        judge_dosage = self.mlp_dosage(judge_herb)
        judge_dosage = self.mlp_dosage(self.relu(judge_dosage))

        return judge_herb, judge_dosage