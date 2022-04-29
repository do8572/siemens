import argparse
import hashlib
import json
import os
import shutil
import time
import scipy.stats as st

import numpy as np
import pandas as pd
import torch

from spotlight.cross_validation import (random_train_test_split,
                                        user_based_train_test_split)
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.sequence.representations import LSTMNet
from spotlight.layers import BloomEmbedding
from spotlight.evaluation import sequence_mrr_score

from get_dataset import get_transactions, time_based_train_test_split
from bayes_opt import BayesianOptimization

NUM_SAMPLES = 50

class deepFM:
    def __init__(self):
        self.hyperparams = None

    def fit(self, train, hyperparams=None):
        h = hyperparams if hyperparams != None else self.hyperparams
        
        item_embeddings = BloomEmbedding(train.num_items, round(h["embedding_dim"]),
                                    compression_ratio=h['compression_ratio'],
                                    num_hash_functions=4,
                                    padding_idx=0)

        network = LSTMNet(train.num_items, round(h["embedding_dim"]),
                item_embedding_layer=item_embeddings)

        model = ImplicitSequenceModel(loss='bpr',
                                    representation=network,
                                    n_iter=round(h["n_iter"]),
                                    batch_size=round(h["bsize"]),
                                    embedding_dim=round(h["embedding_dim"])/2,
                                    l2=h["l2"])
        model.fit(train, verbose=True)
        return model
    
    def predict(self):
        pass

    def eval(self, **hyperparams):
        print(hyperparams)
        model = self.fit(self.train, hyperparams)
        mrr = sequence_mrr_score(model, self.validation)
        return np.mean(mrr)

    # TODO: make sure hyperparameter search works (find optimal values)
    def set_hyperparams(self, train, validation):
        self.train = train
        self.validation = validation
        pbounds = {
            "embedding_dim": (30, 100),
            "compression_ratio": (0.4,0.5),
            "n_iter": (10, 10),
            "bsize": (16, 64),
            "l2": (0.0, 1e-6) # 1e-9
        }

        optimizer = BayesianOptimization(
            f=self.eval,
            pbounds=pbounds,
        )

        optimizer.maximize(
        init_points=2,
        n_iter=10,
        )
        self.hyperparams = optimizer.max["params"]
        return self

if __name__ == '__main__':
    val_frac = 0.8
    tot_hit_rate = []
    N_ITER = 45
    PROD_CAT = "family"

    results = pd.read_csv("results.csv")

    for i in range(6):
        train_start, train_stop = 125 * i, 125 * (i+4)
        test_start, test_stop = 125 * (i+4), 125 * (i+5)
        train_size = train_stop - train_start 
        val_start, val_stop = train_start + train_size * val_frac, train_stop
        dataset = get_transactions(train_start, test_stop, type=PROD_CAT)
        #print(np.unique(dataset.timestamps))

        max_sequence_length = 20 # 30
        min_sequence_length = 20
        step_size = max_sequence_length

        train, test = time_based_train_test_split(dataset,
                    train_start, train_stop, test_start, test_stop)
        ttrain, validation = time_based_train_test_split(train,
                    train_start, val_start, val_start, val_stop)
        train = ttrain.to_sequence(max_sequence_length=max_sequence_length,
                                    min_sequence_length=min_sequence_length,
                                    step_size=step_size)
        ttrain = ttrain.to_sequence(max_sequence_length=max_sequence_length,
                                    min_sequence_length=min_sequence_length,
                                    step_size=step_size)
        validation = validation.to_sequence(max_sequence_length=max_sequence_length,
                                    min_sequence_length=min_sequence_length,
                                    step_size=step_size)
        test = test.to_sequence(max_sequence_length=max_sequence_length,
                                    min_sequence_length=min_sequence_length,
                                    step_size=step_size)

        h = {
            "embedding_dim": 64, # 64
            "compression_ratio": 0.8, # 0.8
            "n_iter": N_ITER,
            "bsize": 64,
            "l2": 1e-8
        }

        model = deepFM()#.set_hyperparams(ttrain, validation)
        #model.hyperparams["n_iter"] = 100
        model = model.fit(train, h)
        torch.save(model, "saved_models/deepFM_{}_{}_{}".format(N_ITER, i, PROD_CAT))
        #model = torch.load("saved_models/deepFM")

        sequences = test.sequences[:, :-1]
        targets = test.sequences[:, -1:]

        pred = []
        hit5 = []

        for j in range(len(sequences)):
            predictions = -model.predict(sequences[j])
            #print(st.rankdata(predictions)[targets[i]] <=5)
            #print(st.rankdata(predictions))
            #print(targets.flatten())
            pred.append(predictions)
            hit5.append(st.rankdata(predictions)[targets[j]])
        hit5 = np.array(hit5)
        
        tot_hit_rate.append(np.mean(hit5 <= 5))
        print(f"Hit 5: {np.mean(hit5 <= 5)}")
        print(f"MRR: {np.mean(1/hit5)}")

        results = results.append({"type": PROD_CAT,
                                  "n_iter": N_ITER,
                                  "fold": j,
                                   "hit@5": np.mean(hit5 <= 5)},
                                    ignore_index=True)
        results.to_csv("results.csv", index=False)

    #mrr = sequence_mrr_score(model, test)
    #print(np.mean(mrr))
    print(np.mean(tot_hit_rate))
    print(np.std(tot_hit_rate))