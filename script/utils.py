#!/usr/bin/env python

import sys
import numpy as np
import torch
from gensim.models import Word2Vec
import pandas as pd
import networkx as nx


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
            Saves model when validation loss decrease.
        """
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class PreDataset(torch.utils.data.Dataset):
    def __init__(self, a, b, c, d):
        self.S_array, self.Sd_array, self.T_array, self.H_array = a, b, c, d

    def __getitem__(self, idx):
        sid = self.S_array[idx]
        sdid = self.Sd_array[idx]
        tid = self.T_array[idx]
        hid = self.H_array[idx]
        return sid, sdid, tid, hid

    def __len__(self):
        return self.S_array.shape[0]


class PreDatasetLung(torch.utils.data.Dataset):
    def __init__(self, a, b, c):
        self.S_array, self.Sd_array, self.H_array = a, b, c

    def __getitem__(self, idx):
        sid = self.S_array[idx]
        sdid = self.Sd_array[idx]
        hid = self.H_array[idx]
        return sid, sdid, hid

    def __len__(self):
        return self.S_array.shape[0]


class PreDatasetPTM(torch.utils.data.Dataset):
    def __init__(self, a, b):
        self.S_array, self.H_array = a, b

    def __getitem__(self, idx):
        sid = self.S_array[idx]
        hid = self.H_array[idx]
        return sid, hid

    def __len__(self):
        return self.S_array.shape[0]


class PreDatasetDosage(torch.utils.data.Dataset):
    def __init__(self, a, b, c):
        self.S_array, self.H_array, self.H_d_array = a, b, c

    def __getitem__(self, idx):
        sid = self.S_array[idx]
        hid = self.H_array[idx]
        h_did = self.H_d_array[idx]
        return sid, hid, h_did

    def __len__(self):
        return self.S_array.shape[0]


class SubnetworkSymptomTermMapping(object):
    def __init__(self):
        self.embedding = Word2Vec.load('D:/Work/Project/SDKG/data/network/Deepwalk/word2vec_symptom_network_128.model')
        self.embed_list = self.embedding.wv.index_to_key
        self.symptom_network_data = pd.read_csv('D:/Work/Project/SDKG/data/network/Network.txt')
        self.symptom_network_data_id = pd.read_csv('D:/Work/Project/SDKG/data/network/Network_id.txt')
        self.degree_flag: int = 2

    def get_entity_from_graph(self, graph: nx.Graph):
        """
        from graph to entity list with filtered degree
        :param degree_flag:
        :param graph: connected graph
        :return: filtered entity list
        """
        degree_dict = dict(graph.degree)
        temp_list = []
        temp_list_append = temp_list.append
        for node in degree_dict.keys():
            # 20211130 add degree flag
            if degree_dict[node] > self.degree_flag:  # degree filter
                temp_list_append(node)
        return temp_list

    def sstm_filter(self, symptom_list):
        """
        Subnetwork-based Symptom Term Mapping (SSTM)
        input: Symptom list
        :return: SSTMed Symptom list
        """

        result_sstm_for_list = []

        for symptom in symptom_list:
            # 20211130 judge
            if symptom in self.embed_list:
                # print(symptom, 'in w2v')
                result_sstm_for_list.append(symptom)
            else:
                # 1. Symptom list -> symptom word set list
                symptom_word = set([word for word in symptom])
                # print(symptom_word)

                # 2. For set, query edges in symptom df, and construct graph
                subgraph_df = self.symptom_network_data.query('Source in @symptom_word or Target in @symptom_word')
                if len(subgraph_df) > 0:
                    subgraph = nx.from_pandas_edgelist(subgraph_df, 'Source', 'Target')

                    # 3. for "connected graph", get the symptom list with nodes' degree > degree_filter
                    if nx.is_connected(subgraph):  # connected graph
                        result_sstm_for_list += self.get_entity_from_graph(subgraph)
                    else:  # not connected graph
                        # print(f'Not Connected graph of {symptom} !')
                        extract_networks = sorted(nx.connected_components(subgraph), key=len, reverse=True)
                        for subnetwork in extract_networks:
                            graph_sub = subgraph.subgraph(subnetwork)
                            assert nx.is_connected(graph_sub) == True
                            result_sstm_for_list += self.get_entity_from_graph(graph_sub)
                else:
                    print(f"The symptom '{symptom}' doesn't have its subgraph in symptom network.")

        result = list(set(result_sstm_for_list))
        # print(len(result), len(result_sstm_for_list))
        return result
