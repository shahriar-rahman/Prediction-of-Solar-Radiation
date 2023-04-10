import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras import activations
from utils import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l1_l2
import numpy as np


class solarPredictionV1:
    def __init__(self):     # pandas data set extractor
        self.df = pd.read_excel('C:/Users/USER/PycharmProjects/Neural_Network/Prediction_SolarRadiation/dataset/Rad2.xlsx')

    def manipulate_data_frame(self):
        self.df_updated = self.df.iloc[:, 2:6]      # Select just columns 3 - 11 & remove the rest
        #self.df_updated = self.df_updated
        self.df_norm = (self.df_updated - self.df_updated.mean()) / self.df_updated.std()  # Normalize Features & Label
        print(self.df_updated.head)

    def convert_label(self, predictions):
        y_mean = self.df_updated['ETR (Wh/m^2)'].mean()    # Find the mean value of just Radiation
        y_std = self.df_updated['ETR (Wh/m^2)'].std()      # Find the standard deviation of just Radiation
        return (predictions*y_std+y_mean)               # Return the converted actual values of label

    def feature_label_selection(self):
        self.x = self.df_norm.iloc[:, 1:]           # Choose all except the 1st Column
        self.y = self.df_norm.iloc[:, :1]           # Choose only 1st Column
        print(self.x.head())
        print(self.y.head())

    def modify_feature_label_values(self):
        self.x_array = self.x.values                # Store as an array from list
        self.y_array = self.y.values
        print(self.x_array.shape)
        print(self.y_array.shape)

    def train_test(self):
        self.x_train, self.x_val_test, self.y_train,  self.y_val_test = train_test_split(self.x_array, self.y_array,
                                                                                 test_size=0.20, random_state=0)
        self.x_val, self.x_test, self.y_val, self.y_test = train_test_split(self.x_val_test, self.y_val_test,
                                                                                 test_size=0.50, random_state=0)
        # Feature and Label arrays spilt into 70/15/15 into train, validation & test set
        print('X Train Shape: ', self.x_train.shape)
        print('Y Train Shape: ', self.y_train.shape)
        print('X Test Shape:', self.x_test.shape)
        print('Y Test Shape: ', self.y_test.shape)

    def get_model(self):
        # Base Structure of the model
        # Learning Rate Decay
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2,
            decay_steps=10000,
            decay_rate=0.09)
        Adam = keras.optimizers.Adam(learning_rate=lr_schedule)  # Adam with learning_rate_decay
        SGD = keras.optimizers.SGD(learning_rate=0.001, momentum=0.0, nesterov=False)
        RMSprop = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False)
        Adadelta = keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07)
        Adagrad = keras.optimizers.Adagrad(learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07)
        Adamax = keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        Nadam = keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        ftrl = keras.optimizers.Ftrl(learning_rate=0.001, learning_rate_power=-0.5, initial_accumulator_value=0.1,
                                                        l1_regularization_strength=0.0, l2_regularization_strength=0.0)
        model = Sequential([
            Dense(20, input_shape=(3, ), activation=activations.relu),
            Dense(30, activation=activations.relu),
            Dense(40, activation=activations.relu),
            Dense(50, activation=activations.relu),
            Dense(50, activation=activations.relu, kernel_regularizer=l2(0.009), bias_regularizer=l2(0.009)),
            Dropout(0.1),
            Dense(40, activation=activations.relu, kernel_regularizer=l2(0.009), bias_regularizer=l2(0.009)),
            Dropout(0.1),
            Dense(30, activation=activations.relu, kernel_regularizer=l2(0.009), bias_regularizer=l2(0.009)),
            Dense(20, activation=activations.relu, kernel_regularizer=l2(0.009), bias_regularizer=l2(0.009)),
            Dense(1, activation=activations.linear)
        ])
        # Compile used to load the structure with loss & optimizers
        model.compile(loss='mean_squared_error', optimizer=Adam, metrics=['mse'])
        return model

    def train_model(self):
        es_cb = EarlyStopping(monitor='val_loss', patience=8)   # Early Stopping Conditions (mainly for analyzing)
        model = self.get_model()                                 # Calls the function that returns a model
        history = model.fit(                                     # Information stored in Variable
            self.x_train, self.y_train,                          # Fit the data in model & Train
            validation_data=(self.x_val_test, self.y_val_test),          # Validation on Test data
            epochs=100,                                          # Epochs
            batch_size=128,
            callbacks=[es_cb]                                   # Callbacks to identify the ideal epochs
        )

        # View History
        print(history.history)

        # Predictions made on Validation Set
        preds_on_trained = model.predict(self.x_test)
        # Predicted Values on Validation Set
        predicttions_trained = [self.convert_label(y) for y in preds_on_trained]
        Validation_results = [self.convert_label(y) for y in self.y_val_test]
        print(predicttions_trained)
        print(Validation_results)

        # Predictions made on Unseen / Real Set
        preds_on_untrained = model.predict(np.array(self.x_test))
        predicttions_untrained = [self.convert_label(y) for y in preds_on_untrained]

        Unseen_Values = [self.convert_label(y) for y in self.y_test]

        # Predicted Values on Unseen Set
        for predicted, actual in zip(predicttions_untrained, Unseen_Values):
            print("'{}'".format(predicted))
            print(actual.astype(int))
            print("")

        # visualizing losses and accuracy
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()


p = solarPredictionV1()
p.manipulate_data_frame()
p.feature_label_selection()
p.modify_feature_label_values()
p.train_test()
p.get_model().summary()
p.train_model()

