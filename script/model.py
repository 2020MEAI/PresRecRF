import torch
import numpy as np
from torch import nn


path = '/Users/xindong/Documents/Work/Projects/GitHub-Work/PresRecRF/data'


class PresRecRF(torch.nn.Module):
    def __init__(self, batch_size, embedding_dim, symptom_cnt, herb_cnt, drop_ratio, semantic='BERT'):
        super(PresRecRF, self).__init__()
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.semantic = semantic

        self.sym_random = torch.nn.Embedding(self.batch_size, self.embedding_dim)
        self.herb_random = torch.nn.Embedding(self.batch_size, self.embedding_dim)

        # 1. Initialize Semantic Embedding
        # 1.1 Load Semantic Embedding (BERT version)
        if self.semantic == 'BERT':
            self.bert_sym_embedding = nn.Embedding.from_pretrained(
                 torch.from_numpy(np.load(path+'/Semantic_BERT'+'/Sem-symptom.npy')))

            self.bert_herb_embedding = nn.Embedding.from_pretrained(
                 torch.from_numpy(np.load(path+'/Semantic_BERT'+'/Sem-herb.npy')))

        # 1.2 Load Semantic Embedding (LLM version)
        elif self.semantic == 'LLM':
            self.bert_sym_embedding = nn.Embedding.from_pretrained(
                 torch.as_tensor(torch.from_numpy(np.load(path+'/Semantic_LLM'+'/LLM-symptom.npy')), dtype=torch.float32))

            self.bert_herb_embedding = nn.Embedding.from_pretrained(
                 torch.as_tensor(torch.from_numpy(np.load(path+'/Semantic_LLM'+'/LLM-herb.npy')), dtype=torch.float32))

        else:
            self.bert_sym_embedding = nn.Embedding.from_pretrained(
                torch.from_numpy(np.load(path + '/Semantic_BERT' + '/Sem-symptom.npy')))

            self.bert_herb_embedding = nn.Embedding.from_pretrained(
                torch.from_numpy(np.load(path + '/Semantic_BERT' + '/Sem-herb.npy')))

        # 2. Initialize Structural Embedding
        self.sym_embedding = nn.Embedding.from_pretrained(
            torch.as_tensor(torch.from_numpy(
                np.load(path + r'/Structural_Network/Net-HSP-symptom.npy')
            ), dtype=torch.float32))
        self.herb_embedding = nn.Embedding.from_pretrained(
            torch.as_tensor(torch.from_numpy(
                np.load(path + r'/Structural_Network/Net-HSP-herb.npy')
            ), dtype=torch.float32))

        # 3. Form Symptom Embedding
        self.mlp_sym_1 = torch.nn.Linear(self.embedding_dim, 256)
        self.tanh_1 = torch.nn.Tanh()

        self.mlp_sym_2 = torch.nn.Linear(256, 256)
        self.tanh_2 = torch.nn.Tanh()

        self.mlp_sym_3 = torch.nn.Linear(256, self.embedding_dim)
        # self.relu = torch.nn.ReLU()

        if self.semantic == 'BERT':
            self.mlp_bert_1 = torch.nn.Linear(768, 256)
        elif self.semantic == 'LLM':
            self.mlp_bert_1 = torch.nn.Linear(1536, 256)
        else:
            self.mlp_bert_1 = torch.nn.Linear(768, 256)

        self.mlp_bert_2 = torch.nn.Linear(256, self.embedding_dim)

        # 4. Form Herb Embedding
        self.mlp_herb_1 = torch.nn.Linear(self.embedding_dim, 256)
        self.tanh_herb_1 = torch.nn.Tanh()

        self.mlp_herb_2 = torch.nn.Linear(256, 256)
        self.tanh_herb_2 = torch.nn.Tanh()

        self.mlp_herb_3 = torch.nn.Linear(256, self.embedding_dim)

        if self.semantic == 'BERT':
            self.mlp_bert_herb_1 = torch.nn.Linear(768, 256)
        elif self.semantic == 'LLM':
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