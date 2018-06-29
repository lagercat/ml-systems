import tensorflow as tf
import pandas as pd
import numpy as np

CSV_COLUMNS = ['Survived', 'Pclass', 'Sex', 'Age', 'Cabin',
               'Name', 'Fare', 'Embarked', 'SibSp', 'Parch']


DATA = '~/Datasets/'
DATASET = DATA + 'titanic/train.csv'
SUBMIT_DATASET = DATA + 'titanic/test.csv'
fill_dict = {
    'Pclass': 0,
    'Sex': '',
    'Age': 0,
    'Fare': 0,
    'Embarked': '',
    'SibSp': 0,
    'Parch': 0,
    'agcl': 0,
    'fsize': 0,
    'title': '',
    'Cabin': '',
    'deck': '',
    'Fare_Per_Person': 0
}

title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
              'Dr', 'Ms', 'Mlle', 'Col', 'Capt', 'Mme', 'Countess',
              'Don', 'Jonkheer']
cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', '']


def replace_titles(x):
    title = x['title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title == 'Dr':
        if x['Sex'] == 'Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title


def substring_in_string(big_string, substrings):
    if big_string is not np.nan:
        for substring in substrings:
            if substring in big_string or substring.lower() in big_string:
                return substring
    return np.nan


def feature_engineering(data):
    data['agcl'] = data['Age'] * data['Pclass']
    data['fsize'] = data['SibSp'] + data['Parch']
    data['title'] = data['Name'].map(lambda x: substring_in_string(x,
                                     title_list))
    data['title'] = data.apply(replace_titles, axis=1)
    data['deck'] = data['Cabin'].map(lambda x: substring_in_string(x,
                                     cabin_list))
    data['Fare_Per_Person'] = data['Fare']/(data['fsize']+1)
    data = data.drop('Name', 1)
    data = data.drop('Cabin', 1)
    data = data.fillna(fill_dict)
    return data


def load_data(ratio=0.7, data=812):
    data = pd.read_csv(DATASET, usecols=CSV_COLUMNS,
                       header=0)
    data = feature_engineering(data)
    rows = data.shape[0]

    train_data = data.head(int(ratio * rows))
    test_data = data.head(rows - int(ratio * rows))

    train_x, train_y = train_data.drop('Survived', 1), train_data['Survived']
    test_x, test_y = test_data.drop('Survived', 1), test_data['Survived']

    return (train_x, train_y), (test_x, test_y)


def load_submit():
    data = pd.read_csv(SUBMIT_DATASET, usecols=CSV_COLUMNS[1:],
                       header=0)
    data = feature_engineering(data)
    return data


def inp(features, labels, mode, rep=2000):
    if mode == 'TRAIN' or mode == 'EVAL':
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        dataset = dataset.batch(32).repeat(rep)
    elif mode == 'PREDICT':
        dataset = tf.data.Dataset.from_tensor_slices(dict(features)).batch(417)
    return dataset


def write_to_file(predicts):
    with open('submit.csv', 'w') as submit_file:
        submit_file.write('PassengerId,Survived\n')
        for idx, result in enumerate(predicts):
            submit_file.write(str(idx + 892) + ',' + str(result) + '\n')
