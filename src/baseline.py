from turtle import pu
import pandas as pd
import numpy as np
import random

def most_common_category_per_customer(data):
    categories = {}
    for cust in data['customer'].unique():
        customer_purchases = data[data['customer'] == cust]
        values = customer_purchases['family'].value_counts()
        category = values.index[0]
        categories[cust] = category
    return categories

def most_popular_item_per_category(data):
    popular_items = {}
    for cat in data['family'].unique():
        category_purchases = data[data['family'] == cat]
        all_items = category_purchases['variation'].value_counts()
        items = all_items.index.tolist()
        popular_items[cat] = items
    return popular_items

def baseline_with_most_popular_item(train, test):
    popular_items = most_popular_item_per_category(train)
    categories = most_common_category_per_customer(train)
    
    predictions = 0
    correct_predictions = 0
    customers_not_in_train_set = 0
    for cust in test['customer'].unique():
        try:
            category = categories[cust]
            items = popular_items[category]
            purchased_items = test[test['customer'] == cust]['variation'].unique()
            for item in items[:5]:
                if item in purchased_items:
                    correct_predictions += 1
                    break
            predictions += 1
        except:
            customers_not_in_train_set += 1
            pass
    #print('customers in test set: ', len(test['customer'].unique()))
    #print('customers_not_in_train_set: ', customers_not_in_train_set)
    return correct_predictions / predictions

def baseline_with_most_popular_new_item(train, test):
    # top 5 variations include only those that a customer has not bought before
    popular_items = most_popular_item_per_category(train)
    categories = most_common_category_per_customer(train)
    
    predictions = 0
    correct_predictions = 0
    customers_not_in_train_set = 0
    for cust in test['customer'].unique():
        try:
            category = categories[cust]
            items = popular_items[category]
            purchased_items_test = test[test['customer'] == cust]['variation'].unique()
            purchased_items_train = train[train['customer'] == cust]['variation'].unique()
            j = 0
            for item in items:
                if j == 5:
                    break
                if item in purchased_items_train:
                    continue
                else:
                    j += 1
                    if item in purchased_items_test:
                        correct_predictions += 1
                        break
            predictions += 1
        except:
            customers_not_in_train_set += 1
            pass
    #print('customers in test set: ', len(test['customer'].unique()))
    #print('customers_not_in_train_set: ', customers_not_in_train_set)
    return correct_predictions / predictions

def predict_random_item(items):
    return random.sample(items, k=5)

def baseline_with_random_item(train, test):
    all_items = list(train['variation'].unique())
    
    predictions = 0
    correct_predictions = 0
    for cust in test['customer'].unique():
        items = predict_random_item(all_items)
        purchased_items = test[test['customer'] == cust]['variation'].unique()
        for item in items:
            if item in purchased_items:
                correct_predictions += 1
                break
        predictions += 1
    return correct_predictions / predictions

def one_split(data, split):
    print('ONE SPLIT')
    train = data[data['day'] <= split]
    test = data[data['day'] > split]

    ac = baseline_with_most_popular_item(train, test)
    print('popular variations', ac)

    acc = baseline_with_most_popular_new_item(train, test)
    print('popular new variations', acc)

    accc = baseline_with_random_item(train, test)
    print('random variations', accc)

def sliding_window(data, train_size, test_size):
    print('SLIDING WINDOW')
    iterations = (1250 - train_size) // test_size
    popular = []
    popular_new = []
    rand = []
    for i in range(iterations):
        start = test_size * i
        split = test_size * i + train_size
        end = test_size * i + train_size + test_size
        train = data[(data['day'] > start) & (data['day'] <= split)]
        test = data[(data['day'] > split) & (data['day'] <= end)]

        print('window ', start, split, end)

        ac = baseline_with_most_popular_item(train, test)
        popular.append(ac)
        print('popular variations', ac)

        acc = baseline_with_most_popular_new_item(train, test)
        popular_new.append(acc)
        print('popular new variations', acc)
        
        accc = baseline_with_random_item(train, test)
        rand.append(accc)
        print('random variations', accc)

    print('mean values and standard deviation')
    print('popular variations', np.mean(popular), np.std(popular))
    print('popular new variations', np.mean(popular_new), np.std(popular_new))
    print('random variations', np.mean(rand), np.std(rand)) 

if __name__ == '__main__':
    data = pd.read_csv('transaction_history.csv', sep=',', header=0)
    data['family'] = data['prod'].str[:4]
    data['product'] = data['prod'].str[:9]
    data['variation'] = data['prod'].str[:14]

    one_split(data, 1000)
    sliding_window(data, 500, 125)
    

