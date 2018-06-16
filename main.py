import tensorflow as tf
import pandas as pd
import numpy as np

DATA_PATH = '~/Datasets/'
DATASET_PATH = DATA_PATH + 'stackoverflow/survey_results_public.csv'
DATA_COLUMNS = ['Hobby', 'OpenSource', 'Country', 'Student',
                'Employment', 'FormalEducation', 'UndergradMajor',
                'YearsCodingProf', 'YearsCoding', 'CareerSatisfaction',
                'JobSatisfaction', 'Salary']
COUNTRIES = []


def load_data():
    data = pd.read_csv(DATASET_PATH, usecols=DATA_COLUMNS, skiprows=0, header=0)
    data = data.dropna()
       
    rows_in_train = int(0.7 * data.shape[0])
    train = data.iloc[0:rows_in_train]
    test = data.iloc[rows_in_train:]

    train_x, train_y = train.drop('Salary', 1), train['Salary'].to_frame()
    test_x, test_y = test.drop('Salary', 1), test['Salary'].to_frame()

    return (train_x, train_y), (test_x, test_y)


def train_input(features, results):
    features = dict(features)
    inputs = (features, results)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(40000)

    return dataset

def get_unique_values(*args):
    list_of_all_values = []
    for argument in args:
        list_of_all_values += argument.values.T.tolist()
    return list(set(list_of_all_values))

def main(argv):
    (train_x, train_y), (test_x, test_y) = load_data()

    feature_columns = []
    for key in train_x.keys():
        feature_columns.append(
            tf.feature_column.categorical_column_with_vocabulary_list(
                key=key,
                vocabulary_list=get_unique_values(train_x[key],
                                                  test_x[key]))
        )
    classifier = tf.estimator.LinearRegressor(feature_columns=feature_columns)

    classifier.train(
        input_fn=lambda:train_input(train_x, train_y),
        steps=100)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
