#coding=utf-8

try:
    import numpy as np
except:
    pass

try:
    import pandas as pd
except:
    pass

try:
    from os.path import exists
except:
    pass

try:
    import itertools
except:
    pass

try:
    import tensorflow as tf
except:
    pass

try:
    import tensorflow.keras as keras
except:
    pass

try:
    from tensorflow.keras import regularizers
except:
    pass

try:
    from keras.wrappers.scikit_learn import KerasClassifier
except:
    pass

try:
    from sklearn.model_selection import GridSearchCV
except:
    pass

try:
    from tensorflow.keras.preprocessing.text import Tokenizer
except:
    pass

try:
    from keras.utils.vis_utils import plot_model
except:
    pass

try:
    import tensorflow.keras.backend as K
except:
    pass

try:
    from tensorflow.keras.models import Model
except:
    pass

try:
    from tensorflow.keras.layers import *
except:
    pass

try:
    from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
except:
    pass

try:
    from hyperopt import Trials, STATUS_OK, tpe
except:
    pass

try:
    from keras.utils import np_utils
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import choice, uniform
except:
    pass

try:
    import matplotlib.pyplot as plt
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
def deep_fm_model(k, dnn_dim, dnn_dr, k_reg=(1e-4, 1e-4), a_reg=(1e-4, 1e-4), act_fun="relu"):
    
    inputs = define_input_layers(transformed)
    
    y_fm_1d = fm_1d(inputs)
    y_fm_2d, embed_2d = fm_2d(inputs, k)
    y_dnn = dnn_part(embed_2d, dnn_dim, dnn_dr, k_reg, a_reg, act_fun)
    
    # combinded deep and fm parts
    y = Concatenate()([y_fm_1d, y_fm_2d, y_dnn])
    y = Dense(2, name = "concat")(y)
    y = Softmax(name = 'deepfm_output')(y)
    deep_fm_model = Model(inputs, y)
    
    return deep_fm_model

def df2xy(df, model):
    x = [df[inp.name].values for inp in model.inputs]
    y = df.label.values
    return x,np.asarray(y).astype('float32')

deepfm_model = <keras.engine.functional.Functional object at 0x00000242843D3AF0>
"""
Data Generator function for Fold.
"""
print("Fuck hyperas")
train = pd.read_csv("chad_library.csv")
val_day = np.max(train.day.unique())
train_data, val_data = train[train.day < val_day], train[train.day == val_day]
train_data_x, train_data_y = df2xy(train_data, deepfm_model)
val_data_x, val_data_y = df2xy(val_data, deepfm_model)


def keras_fmin_fnct(space):

    """
    Wrapper function for hyperparameter tunning.
    """
    deepfm_model = deep_fm_model(**deepFM_params)
    train_x, train_y = df2xy(train, fm_model_1d)
    train_y = pd.get_dummies(pd.Series(train_y)).to_numpy()
    
    fm_model_1d.compile(loss = 'binary_crossentropy', optimizer='adam')
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    model_ckp = ModelCheckpoint(filepath=f'./models/1d_{i}.h5', 
                                        monitor='val_loss',
                                        save_weights_only=True, 
                                        save_best_only=True)
    callbacks = [early_stop, model_ckp]
    
    train_history = fm_model_1d.fit(train_x, train_y, 
                                              epochs=300, batch_size=256, 
                                              validation_split=0.1, 
                                              callbacks = callbacks)
    
    validation_entrophy = np.amin(result.history['val_loss'])
    return {"loss": validation_entrophy, 'status': STATUS_OK, 'model': fm_model_1d}

def get_space():
    return {
    }
