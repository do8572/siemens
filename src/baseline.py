from turtle import pu
import pandas as pd
import numpy as np
import random

def most_common_category_per_customer(data):
    categories = {}
    for cust in data['customer'].unique():
        customer_purchases = data[data['customer'] == cust]
        values = customer_purchases['family'].value_counts()
        category = values.index.tolist()
        categories[cust] = category
    return categories

def most_popular_variation_and_product_per_category(data):
    popular_variations = {}
    popular_products = {}
    for fam in data['family'].unique():
        category_purchases = data[data['family'] == fam]

        all_variations = category_purchases['variation'].value_counts()
        variations = all_variations.index.tolist()
        popular_variations[fam] = variations

        all_products = category_purchases['product'].value_counts()
        products = all_products.index.tolist()
        popular_products[fam] = products
    return popular_variations, popular_products

def baseline_most_popular(train, test, prediction, top=5):
    popular_variations, popular_products = most_popular_variation_and_product_per_category(train)
    categories = most_common_category_per_customer(train)
    top_categories = train['family'].value_counts().index.tolist()

    predictions = 0
    correct_predictions = 0
    for cust in test['customer'].unique():
        try:
            category = categories[cust][0]
            if prediction == 'variation':
                items = popular_variations[category]
            elif prediction == 'product':
                items = popular_products[category]
            elif prediction == 'family':
                items = top_categories
            purchased_items = test[test['customer'] == cust][prediction].unique()
            for item in items[:top]:
                if item in purchased_items:
                    correct_predictions += 1
                    break
            predictions += 1
        except:
            pass
    return correct_predictions / predictions

def baseline_most_popular_new(train, test, prediction, top=5):
    popular_variations, popular_products = most_popular_variation_and_product_per_category(train)
    categories = most_common_category_per_customer(train)
    top_categories = train['family'].value_counts().index.tolist()
    
    predictions = 0
    correct_predictions = 0
    for cust in test['customer'].unique():
        try:
            category = categories[cust][0]
            if prediction == 'variation':
                items = popular_variations[category]
            elif prediction == 'product':
                items = popular_products[category]
            elif prediction == 'family':
                items = top_categories
            purchased_items_test = test[test['customer'] == cust][prediction].unique()
            purchased_items_train = train[train['customer'] == cust][prediction].unique()
            j = 0
            for item in items:
                if j == top:
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
            pass
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

    ac = baseline_most_popular(train, test, prediction='variation')
    print('popular variations', ac)

    acc = baseline_most_popular_new(train, test, prediction='variation')
    print('popular new variations', acc)

    accc = baseline_with_random_item(train, test)
    print('random variations', accc)

def sliding_window(data, train_size, test_size):
    print('SLIDING WINDOW')
    iterations = (1250 - train_size) // test_size
    
    for top in [1,2,3,5,10,15,20,30]:
        print(f'------------- top {top} items ----------------')
        popular = {'variation':[], 'product':[], 'family':[]}
        popular_new = {'variation':[], 'product':[], 'family':[]}
        for i in range(iterations):
            start = test_size * i
            split = test_size * i + train_size
            end = test_size * i + train_size + test_size
            train = data[(data['day'] > start) & (data['day'] <= split)]
            test = data[(data['day'] > split) & (data['day'] <= end)]

            print('window ', start, split, end)
            for prediction in ['variation', 'product','family']:
                ac = baseline_most_popular(train, test, prediction=prediction, top=top)
                popular[prediction].append(ac)
                #print(f'popular {prediction}', ac)

                acc = baseline_most_popular_new(train, test, prediction=prediction, top=top)
                popular_new[prediction].append(acc)
                #print(f'popular new {prediction}', acc)

        print('--------------------------------------')
        print(f'mean values and standard deviation for top {top} items')
        for prediction in ['variation', 'product', 'family']:
            print(f'popular {prediction}', np.mean(popular[prediction]), np.std(popular[prediction]))
            print(f'popular new {prediction}', np.mean(popular_new[prediction]), np.std(popular_new[prediction]))

if __name__ == '__main__':
    data = pd.read_csv('transaction_history.csv', sep=',', header=0)
    data['family'] = data['prod'].str[:4]
    data['product'] = data['prod'].str[:9]
    data['variation'] = data['prod'].str[:14]

    #one_split(data, 1000)
    sliding_window(data, 500, 125)
    