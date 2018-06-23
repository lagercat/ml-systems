import tensorflow as tf
import pandas as pd


CSV_COLUMNS = ['Survived', 'Pclass', 'Sex', 'Age',
               'Fare', 'Embarked', 'SibSp', 'Parch']


DATA = '~/Datasets/'
DATASET = DATA + 'titanic/train.csv'
SUBMIT_DATASET = DATA + 'titanic/test.csv'
fill_dict = {
    'Pclass': '',
    'Sex': '',
    'Age': 0,
    'Fare': 0,
    'Embarked': '',
    'SibSp': '',
    'Parch': ''
}


def load_data():
    data = pd.read_csv(DATASET, usecols=CSV_COLUMNS,
                       header=0).fillna(fill_dict)
    rows = data.shape[0]

    train_data = data.head(int(0.7 * rows))
    test_data = data.head(rows - int(0.7 * rows))

    train_x, train_y = train_data.drop('Survived', 1), train_data['Survived']
    test_x, test_y = test_data.drop('Survived', 1), test_data['Survived']

    return (train_x, train_y), (test_x, test_y)


def load_submit():
    data = pd.read_csv(SUBMIT_DATASET, usecols=CSV_COLUMNS[1:],
                       header=0).fillna(fill_dict)
    return data


def inp(features, labels, mode):
    if mode == 'TRAIN' or mode == 'EVAL':
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        dataset = dataset.repeat(100).batch(32)
    elif mode == 'PREDICT':
        dataset = tf.data.Dataset.from_tensor_slices(dict(features)).batch(417)
    return dataset


def write_to_file(predicts):
    with open('submit.csv', 'w') as submit_file:
        submit_file.write('PassengerId,Survived\n')
        for idx, result in enumerate(predicts):
            submit_file.write(str(idx + 892) + ',' + str(result) + '\n')
