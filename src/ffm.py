import pandas as pd
import xlearn as xl
import random
from recommenders.datasets.pandas_df_utils import LibffmConverter
from recommenders.utils.timer import Timer

def negative_sampling(dataset):
    all_items = list(dataset['prod'].unique())
    final_dataset = pd.DataFrame(columns=['customer', 'state', 'ind_code', 'ind_seg_code', 'family', 'prod', 'variation', 'bought'])
    rows = []
    for user in dataset['customer'].unique():
        user_data = dataset[dataset['customer'] == user]
        n = user_data['prod'].unique().shape[0]
        user_items = set(user_data['prod'].unique())
        negative_samples = set()
        i = 0
        while i < n:
            item = random.sample(all_items, 1)[0]
            if item in user_items or item in negative_samples:
                continue
            else:
                negative_samples.add(item)
                i += 1
        
        for item in user_items:
            row = [user, user_data['state'].iloc[0], user_data['ind_code'].iloc[0], user_data['ind_seg_code'].iloc[0], item[:4], item[:9], item, 1]
            rows.append(row)

        for item in negative_samples:
            row = [user, user_data['state'].iloc[0], user_data['ind_code'].iloc[0], user_data['ind_seg_code'].iloc[0], item[:4], item[:9], item, 0]
            rows.append(row)

    rows = pd.DataFrame(rows, columns=['customer', 'state', 'ind_code', 'ind_seg_code', 'family', 'prod', 'variation', 'bought'])
    final_dataset = final_dataset.append(rows)
    return final_dataset

def predicting_data(dataset):
    all_items = list(dataset['prod'].unique())
    final_dataset = pd.DataFrame(columns=['customer', 'state', 'ind_code', 'ind_seg_code', 'family', 'prod', 'variation', 'bought'])
    rows = []
    m = len(dataset['customer'].unique())
    for i, user in enumerate(dataset['customer'].unique()):
        print(i, '/', m)
        if i == 100:
            break
        user_data = dataset[dataset['customer'] == user]
        user_items = set(user_data['prod'].unique())
        for item in all_items:
            if item not in user_items:
                row = [user, user_data['state'].iloc[0], user_data['ind_code'].iloc[0], user_data['ind_seg_code'].iloc[0], item[:4], item[:9], item, 0]
                rows.append(row)

    rows = pd.DataFrame(rows, columns=['customer', 'state', 'ind_code', 'ind_seg_code', 'family', 'prod', 'variation', 'bought'])
    final_dataset = final_dataset.append(rows)
    return final_dataset

                

def ffm(train, test, valid, i=0):
    n_train = negative_sampling(train)
    n_valid = negative_sampling(valid)

    converter_train = LibffmConverter().fit(n_train, col_rating='bought')
    train_out = converter_train.transform(n_train)
    train_out.to_csv(r'train.txt', header=None, index=None, sep=' ', mode='w')

    converter_valid = LibffmConverter().fit(n_valid, col_rating='bought')
    valid_out = converter_valid.transform(n_valid)
    valid_out.to_csv(r'valid.txt', header=None, index=None, sep=' ', mode='w')

    # Training task
    ffm_model = xl.create_ffm() # Use field-aware factorization machine (ffm)
    ffm_model.setSigmoid()        # Convert output to 0-1             
    ffm_model.setTrain("train.txt")     # Set the path of training dataset
    ffm_model.setValidate("valid.txt")  # Set the path of validation dataset

    LEARNING_RATE = 0.2
    LAMBDA = 0.002
    EPOCH = 10 # number of epoches
    OPT_METHOD = "adagrad" # options are "sgd", "adagrad" and "ftrl"

    # The metrics for binary classification options are "acc", "prec", "f1" and "auc"
    # for regression, options are "rmse", "mae", "mape"
    METRIC = "auc" 

    param = {"task":"binary", 
            "lr": LEARNING_RATE, 
            "lambda": LAMBDA, 
            "metric": METRIC,
            "epoch": EPOCH,
            "opt": OPT_METHOD
            }

    # Start to train
    # The trained model will be stored in model.out
    with Timer() as time_train:
        ffm_model.fit(param, "model.out")
    print(f"Training time: {time_train}")

    # PREDICT
    n_test = predicting_data(train)

    converter_test = LibffmConverter().fit(n_test, col_rating='bought')
    test_out = converter_test.transform(n_test)
    print(test_out)
    test_out.to_csv(r'test_{}.txt'.format(i), header=None, index=None, sep=' ', mode='w')

    ffm_model.setTest(f"test_{i}.txt")  # Set the path of test dataset

    # Start to predict
    # The output result will be stored in output.txt
    with Timer() as time_predict:
        ffm_model.predict("model.out", "output.txt")
    print(f"Prediction time: {time_predict}")



def sliding_window(data, train_size, test_size):
    iterations = (1250 - train_size) // test_size
    
    for i in range(iterations):
        start = test_size * i
        split = test_size * i + train_size
        end = test_size * i + train_size + test_size
        valid = int((split - start) * 0.2)
        train = data[(data['day'] > start) & (data['day'] <= split - valid)]
        valid = data[(data['day'] > split - valid) & (data['day'] <= split)]
        test = data[(data['day'] > split) & (data['day'] <= end)]

        ffm(train, test, valid, i)


    
    

if __name__ == '__main__':
    dataset = pd.read_csv("transaction_history.csv")
    dataset = dataset.fillna('')
    sliding_window(dataset, 500, 125)


