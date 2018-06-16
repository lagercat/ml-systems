import tensorflow as tf
import pandas as pd

DATA_PATH = '~/Datasets/'
DATASET_PATH = DATA_PATH + 'stackoverflow/survey_results_public.csv'
DATA_COLUMNS = ['Hobby', 'OpenSource', 'Country', 'Student',
                'Employment', 'FormalEducation', 'UndergradMajor',
                'YearsCodingProf', 'YearsCoding', 'CareerSatisfaction',
                'JobSatisfaction', 'Salary']
COUNTRIES = []


def load_data():
    data = pd.read_csv(DATASET_PATH, names=DATA_COLUMNS, header=0)
    rows_in_train = int(0.7 * data.shape[0])

    train = data.iloc[0:rows_in_train]
    test = data.iloc[rows_in_train:]

    train_x, train_y = train.drop('Salary', 1), train['Salary'].to_frame()
    test_x, test_y = test.drop('Salary', 1), test['Salary'].to_frame()

    return (train_x, train_y), (test_x, test_y)


def get_unique_values(*args):
    for argument in args:
        print(1)


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

    feature_columns = [
        tf.feature_column.categorical_column_with_vocabulary_list(
            key='Hobby',
            vocabulary_list=['Yes', 'No']),
        tf.feature_column.categorical_column_with_vocabulary_list(
            key='OpenSource',
            vocabulary_file=['Yes', 'No']),
        tf.feature_column.categorical_column_with_vocabulary_list(
            key='Country',
            vocabulary_file=COUNTRIES),
    ]


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
