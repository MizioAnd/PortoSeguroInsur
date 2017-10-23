# porto_seguro_insur.py
#  Assumes python vers. 3.6

__author__ = 'mizio'
import csv as csv
import numpy as np
import pandas as pd
import pylab as plt
from fancyimpute import MICE
import random
from sklearn.model_selection import cross_val_score
import tensorflow as tf

class PortoSeguroIsur:
    ''' Pandas DataFrame '''
    # For .read_csv, always use header=0 when you know row 0 is the header row
    df = pd.read_csv('../input/train.csv', header=0)
    df_test = pd.read_csv('../input/test.csv', header=0)

    @staticmethod
    def features_with_null_logical(df, axis=1):
        row_length = len(df._get_axis(0))
        # Axis to count non null values in. aggregate_axis=0 implies counting for every feature
        aggregate_axis = 1 - axis
        features_non_null_series = df.count(axis=aggregate_axis)
        # Whenever count() differs from row_length it implies a null value exists in feature column and a False in mask
        mask = row_length == features_non_null_series
        return mask

    def missing_values_in_dataframe(self, df):
        mask = self.features_with_null_logical(df)
        print(df[mask[mask == 0].index.values].isnull().sum())
        print('\n')

    def clean_data(self, df, is_train_data=1):
        df = df.copy()
        if df.isnull().sum().sum() > 0:
            if is_train_data:
                df = df.dropna()
            else:
                df = df.dropna(1)
        return df




def main():
    porto_seguro_insur = PortoSeguroIsur()
    df = porto_seguro_insur.df.copy()
    df_test = porto_seguro_insur.df_test.copy()

    df = df.replace(-1, np.NaN)
    df_test = df_test.replace(-1, np.NaN)

    print(df.shape)
    print(df_test.shape)
    df = porto_seguro_insur.clean_data(df)
    df_test = porto_seguro_insur.clean_data(df_test, is_train_data=0)
    print("After dropping NaN")
    print(df.shape)
    print(df_test.shape)


    is_explore_data = 1
    if is_explore_data:
        # Overview of train data
        print('\n TRAINING DATA:----------------------------------------------- \n')
        print(df.head(3))
        print('\n')
        print(df.info())
        print('\n')
        print(df.describe())
        print('\n')
        print(df.dtypes)
        print(df.get_dtype_counts())

        # missing_values
        print('All df set missing values')
        porto_seguro_insur.missing_values_in_dataframe(df)

    is_prediction = 1
    if is_prediction:
        #Todo: Implement tensorflow
        pass


if __name__ == '__main__':
    main()