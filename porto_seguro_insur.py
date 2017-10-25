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

class PortoSeguroInsur:
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

    def reformat_data(self, dataset, labels, num_columns, num_labels):
        # reshape dataset to have dim (remaining)x(number of features)**2. remaining is set by -1 values in reshape().
        # dataset = dataset.reshape((-1, num_columns**2)).astype(np.float64)
        # Map labels/target value to one-hot-encoded frame. None is same as implying newaxis() just replicating array
        # if num_labels > 2:
        labels = (np.arange(num_labels) == labels[:, None]).astype(np.float64)
        return dataset, labels

    def accuracy(self, predictions, labels):
        # Sum the number of cases where the predictions are correct and divide by the number of predictions
        number_of_correct_predictions = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
        return 100*number_of_correct_predictions/predictions.shape[0]



def main():
    porto_seguro_insur = PortoSeguroInsur()
    df = porto_seguro_insur.df.copy()
    df_test = porto_seguro_insur.df_test.copy()

    df = df.replace(-1, np.NaN)
    df_test = df_test.replace(-1, np.NaN)

    print(df.shape)
    print(df_test.shape)
    df = porto_seguro_insur.clean_data(df)
    df_test = porto_seguro_insur.clean_data(df_test)
    # df_test = porto_seguro_insur.clean_data(df_test, is_train_data=0)
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

        print('Uniques')
        uniques_in_id = np.unique(df.id.values).shape[0]
        print(uniques_in_id)
        print('uniques_in_id == df.shape[0]')
        print(uniques_in_id == df.shape[0])


    is_prediction = 1
    if is_prediction:
        #Todo: Implement tensorflow
        # Subset the data to make it run faster
        # (595212, 59)
        # (892816, 58)
        # After dropping NaN
        # (124931, 59)
        # (186567, 58)
        subset_size = 10000

        num_labels = np.unique(df.loc[:subset_size, 'target'].values).shape[0]
        num_columns = df[(df.columns[(df.columns != 'target') & (df.columns != 'id')])].shape[1]
        # Reformat datasets
        x_train = df.loc[:subset_size, (df.columns[(df.columns != 'target') & (df.columns != 'id')])].values
        y_train = df.loc[:subset_size, 'target'].values
        # We only need to one-hot-encode our labels since otherwise they will not match the dimension of the
        # logits in our later computation.
        y_train = porto_seguro_insur.reformat_data(x_train, y_train, num_columns=num_columns, num_labels=num_labels)[1]

        # Todo: we need testdata with labels to benchmark test results.
        # Hack. Use subset of training data (not used in training model) as test data, since it has a label/target value
        # In case there are duplicates in training data it may imply that test results are too good, when using
        # a subset of training data for test.
        x_test = df.loc[subset_size:2*subset_size, (df.columns[(df.columns != 'target') & (df.columns != 'id')])].values
        # x_test = df_test.loc[:subset_size, (df_test.columns[(df_test.columns != 'id')])].values
        y_test = df.loc[subset_size:2*subset_size, 'target'].values
        y_test = porto_seguro_insur.reformat_data(x_test, y_test, num_columns=num_columns, num_labels=num_labels)[1]

        # Todo: we need validation data with labels to perform crossvalidation while training and get a better result.


        # Tensorflow uses a dataflow graph to represent your computations in terms of dependencies.
        graph = tf.Graph()
        with graph.as_default():
            # Load training and test data into constants that are attached to the graph
            tf_train = tf.constant(x_train)
            tf_train_labels = tf.constant(y_train)
            tf_test = tf.constant(x_test)
            tf_test_labels = tf.constant(y_test)

            # As in a neural network the goal is to compute the cross-entropy D(S(w,x), L)
            # x, input training data
            # w_ij, are elements of the weight matrix
            # L, labels or target values of the training data (classification problem)
            # S(), is softmax function
            # Do the Multinomial Logistic Classification
            # step 1.
            # Compute y from the linear model y = WX + b, where b is the bias and W is randomly chosen on
            # Gaussian distribution and bias is set to zero. The result is called the logit.
            # step 2.
            # Compute the softmax function S(Y) which gives distribution
            # step 3.
            # Compute cross-entropy D(S, L) = - Sum_i L_i*log(S_i)
            # step 4.
            # Compute loss L = 1/N * D(S, L)
            # step 5.
            # Use gradient-descent to find minimum of loss wrt. w and b by minimizing L(w,b).
            # Update your weight and bias until minimum of loss function is reached
            # w_i -> w_i - alpha*delta_w L
            # b -> b - alpha*delta_b L
            # OBS. step 5 is faster optimized if you have transformed the data to have zero mean and equal variance
            # mu(x_i) = 0
            # sigma(x_i) = sigma(x_j)
            # This transformation makes it a well conditioned problem.

            # Make a 2-layer Neural network (count number of layers of adaptive weights) with num_columns nodes
            # in hidden layer.
            # Initialize weights on truncated normal distribution. Initialize biases to zero.
            weights_1_layer = tf.Variable(tf.truncated_normal([num_columns, num_columns], dtype=np.float64))
            biases_1_layer = tf.Variable(tf.zeros([num_columns], dtype=np.float64))
            weights_2_layer = tf.Variable(tf.truncated_normal([num_columns, num_labels], dtype=np.float64))
            biases_2_layer = tf.Variable(tf.zeros([num_labels], dtype=np.float64))

            # Logits and loss function.
            logits_hidden_1_layer = tf.matmul(tf_train, weights_1_layer) + biases_1_layer
            # Output unit activations of first layer
            a_1_layer = tf.nn.softmax(logits_hidden_1_layer)
            logits_2_layer = tf.matmul(a_1_layer, weights_2_layer) + biases_2_layer
            loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,
                                                                                   logits=logits_2_layer))

            # Find minimum of loss function using gradient-descent.
            optimized_weights_and_bias = tf.train.GradientDescentOptimizer(0.5).minimize(loss=loss_function)

            # Accuracy variables using the initial values for weights and bias of our linear model.
            train_prediction = tf.nn.softmax(logits_2_layer)
            test_prediction = tf.nn.softmax(tf.matmul(tf.nn.softmax(tf.matmul(tf_test, weights_1_layer) +
                                                                    biases_1_layer), weights_2_layer) + biases_2_layer)
            # Using Rectifier as activation function. Rectified linear unit (ReLU). Compared to sigmoid or other
            # activation functions it allows for faster and effective training of neural architectures
            # train_prediction = tf.nn.relu(logits)
            # test_prediction = tf.nn.relu(tf.matmul(tf_test, weights) + biases)


        number_of_iterations = 900
        # Creating a tensorflow session to effeciently run same computation multiple times using definitions in defined
        # dataflow graph.
        with tf.Session(graph=graph) as session:
            # Ensure that variables are initialized as done in our graph defining the dataflow.
            tf.global_variables_initializer().run()
            for ite in range(number_of_iterations):
                # Compute loss and predictions
                loss, predictions = session.run([optimized_weights_and_bias, loss_function, train_prediction])[1:3]
                if (ite % 100 == 0):
                    print('Loss at iteration %d: %f' % (ite, loss))
                    print('Training accuracy: %.1f%%' % porto_seguro_insur.accuracy(predictions, y_train))
            print('Test accuracy: %.1f%%' % porto_seguro_insur.accuracy(test_prediction.eval(), y_test))

        pass


if __name__ == '__main__':
    main()