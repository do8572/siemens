import os
import h5py
import pandas as pd

from spotlight.interactions import Interactions

def get_transactions(start, stop, type="variations"):
    df = pd.read_csv("transaction_history.csv")
    df = df[(df.day >= start) & (df.day < stop)]
    if type == "product":
        df[["drop1", "prod", "drop2"]] = df['prod'].str.split('_', expand=True)
    if type == "family":
        df[["prod", "drop1", "drop2"]] = df['prod'].str.split('_', expand=True)
    df["customer"] = df["customer"].astype(int)
    df["prod"] = df["prod"].astype('category').cat.codes

    gcustomer = df.groupby("customer").count().reset_index().drop_duplicates()
    gcostumers = gcustomer.loc[gcustomer["new_orders"] > 1, "customer"]
    df = df.loc[df["customer"].isin(gcostumers), :]

    print(df)

    return Interactions(df["customer"].to_numpy() + 1, df["prod"].to_numpy() + 1, \
        ratings=df["new_orders"].to_numpy(), \
        #weights=df["new_orders"].to_numpy(),
        #weights=df[["new_orders", "domestic", "state", "ind_code", "ind_seg_code"]].to_numpy(), \
        timestamps=df["day"].to_numpy())

def get_transactions_df():
    df = pd.read_csv("transaction_history.csv")
    df["customer"] = df["customer"].astype(int)
    #df[["prod", "cat", "id"]] = df["prod"].str.split("_")
    #df["prod"] = df["prod"].astype('category').cat.codes
    #df["cat"] = df["cat"].astype('category').cat.codes
    #df["id"] = df["id"].astype('category').cat.codes

    return df

def time_based_train_test_split(interactions, train_start, train_end, test_start, test_end):
    in_train = (interactions.timestamps >= train_start) & (interactions.timestamps < train_end)
    in_test = (interactions.timestamps >= test_start) & (interactions.timestamps < test_end)

    train = Interactions(interactions.user_ids[in_train],
                        interactions.item_ids[in_train],
                        ratings=interactions.ratings[in_train],
                        timestamps=interactions.timestamps[in_train],
                        weights= interactions.weights,
                        num_users=interactions.num_users,
                        num_items=interactions.num_items)
    
    test =  Interactions(interactions.user_ids[in_test],
                        interactions.item_ids[in_test],
                        ratings=interactions.ratings[in_test],
                        timestamps=interactions.timestamps[in_test],
                        weights= interactions.weights,
                        num_users=interactions.num_users,
                        num_items=interactions.num_items)
    
    return train, test


if __name__ == "__main__":
    transactions = get_transactions()
    train, test = time_based_train_test_split(transactions, 0, 1000, 1000, 1250)