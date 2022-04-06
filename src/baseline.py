import pandas as pd
import random

def most_common_category_per_customer(data):
    categories = {}
    for cust in data['customer'].unique():
        customer_purchases = data[data['customer'] == cust]
        values = customer_purchases['cat'].value_counts()
        category = values.index[0]
        categories[cust] = category
    return categories

def most_popular_item_per_category(data):
    items = {}
    for cat in data['cat'].unique():
        category_purchases = data[data['cat'] == cat]
        all_items = category_purchases['prod'].value_counts()
        item = all_items.index[0]
        items[cat] = item
    return items

def baseline_with_most_popular_item(train, test):
    items = most_popular_item_per_category(train)
    categories = most_common_category_per_customer(train)
    
    predictions = 0
    correct_predictions = 0
    customers_not_in_train_set = 0
    for cust in test['customer'].unique():
        try:
            category = categories[cust]
            item = items[category]
            purchased_items = test[test['customer'] == cust]['prod'].unique()
            if item in purchased_items:
                correct_predictions += 1
            predictions += 1
        except:
            customers_not_in_train_set += 1
            pass
    print('customers in test set: ', len(test['customer'].unique()))
    print('customers_not_in_train_set: ', customers_not_in_train_set)
    return correct_predictions / predictions

def predict_random_item(items):
    return random.choice(items)

def baseline_with_random_item(train, test):
    items = train['prod'].unique()
    
    predictions = 0
    correct_predictions = 0
    for cust in test['customer'].unique():
        item = predict_random_item(items)
        purchased_items = test[test['customer'] == cust]['prod'].unique()
        if item in purchased_items:
            correct_predictions += 1
        predictions += 1
    return correct_predictions / predictions

if __name__ == '__main__':
    data = pd.read_csv('transaction_history.csv', sep=',', header=0)
    data[['cat', 'subcat', 'prod']] = data['prod'].str.split('_', expand=True)

    train = data[data['day'] <= 1000]
    test = data[data['day'] > 1000]

    acc = baseline_with_most_popular_item(train, test)
    print(acc)

    accc = baseline_with_random_item(train, test)
    print(accc)

