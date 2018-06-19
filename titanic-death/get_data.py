import tensorflow as tf
import pandas as pd


CSV_COLUMNS = ['Survived', 'Pclass', 'Sex', 'Age',
               'Fare', 'Embarked', 'SibSp', 'Parch']

DATA = '~/Datasets/'
TRAIN_DATASET = DATA + 'titanic/train.csv'
TEST_DATASET = DATA + 'titanic/train.csv'


def load_data():
    train_data = pd.read_csv(TRAIN_DATASET, usecols=CSV_COLUMNS,
                             header=0).dropna()
    test_data = pd.read_csv(TEST_DATASET, usecols=CSV_COLUMNS,
                            header=0).dropna()

    train_x, train_y = train_data.drop('Survived', 1), train_data['Survived']
    test_x, test_y = test_data.drop('Survived', 1), test_data['Survived']

    return (train_x, train_y), (test_x, test_y)


def inp(features, labels):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.batch(712).repeat(15)

    return dataset
