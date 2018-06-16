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


def convert_salary(x):
    x = x.replace(",", "")
    return float(x)

def load_data():
    data = pd.read_csv(DATASET_PATH, usecols=DATA_COLUMNS, skiprows=0, header=0)
    data = data.dropna()

    data['Salary'] = data['Salary'].apply(convert_salary)

    rows_in_train = int(0.7 * data.shape[0])
    train = data.iloc[0:rows_in_train]
    test = data.iloc[rows_in_train:]

    train_x, train_y = train.drop('Salary', 1), train['Salary']
    test_x, test_y = test.drop('Salary', 1), test['Salary']

    return (train_x, train_y), (test_x, test_y)


def train_input(features, results):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), results))
    dataset = dataset.shuffle(1000).batch(1).repeat()

    return dataset.make_one_shot_iterator().get_next()


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
        steps=10000)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
