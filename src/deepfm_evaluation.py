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

if __name__ == '__main__':
    val_frac = 0.8
    tot_hit_rate = []
    N_ITER = 45
    PROD_CAT = "variations"

    results = pd.read_csv("results.csv")

    for i in range(2):
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
        test_days, test_ids = test.timestamps, test.user_ids
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

        #model = deepFM()#.set_hyperparams(ttrain, validation)
        #model.hyperparams["n_iter"] = 100
        #model = model.fit(train, h)
        #torch.save(model, "saved_models/deepFM_{}_{}_{}".format(N_ITER, i, PROD_CAT))
        model = torch.load("saved_models/deepFM_{}_{}_{}".format(N_ITER, i, PROD_CAT))

        sequences = test.sequences[:, :-1]
        targets = test.sequences[:, -1:]

        pred = []
        hit5 = []

        j = 0
        topN = []
        while j < len(sequences):
            day, id = test_days[j], test_ids[j]
            arr = []
            prank = []
            while j < len(sequences) and day == test_days[j] and id == test_ids[j]:
                print(i, end="\r")
                predictions = -model.predict(sequences[j])
                prank.append(st.rankdata(predictions))
                arr.append(targets[j])
                j += 1
            arr = np.array(arr)
            m_rank = np.mean([(p[arr].min() <= 5) for p in prank])
            print(m_rank)
            topN.append(m_rank)

        print(f"top-N: {np.mean(topN)}")