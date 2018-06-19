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

def main(argv):
    (train_x, train_y), (test_x, test_y) = load_data()

    feature_columns = [
        tf.feature_column.categorical_column_with_identity(key='Pclass',
            num_buckets=4),
        tf.feature_column.categorical_column_with_vocabulary_list(key='Sex',
            vocabulary_list=['male', 'female']),
        tf.feature_column.numeric_column(key='Age'),
        tf.feature_column.numeric_column(key='Fare'),
        tf.feature_column.categorical_column_with_vocabulary_list(key='Embarked',
            vocabulary_list=['C', 'Q', 'S']),
        tf.feature_column.numeric_column(key='SibSp'),
        tf.feature_column.numeric_column(key='Parch')
    ]
    optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=0.1)

    model = tf.estimator.LinearClassifier(feature_columns=feature_columns,
            optimizer=optimizer)
    model.train(input_fn=lambda: inp(test_x, test_y), steps=10000)

    eval_result = model.evaluate(input_fn=lambda: inp(test_x, test_y))

    average_loss = eval_result['average_loss']
    print('\n' + 80 * '*')
    print('Error: ${:.0f}'.format(average_loss ** 0.5))
    print()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
