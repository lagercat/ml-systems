import tensorflow as tf

from get_data import load_data, inp


def main(argv):
    (train_x, train_y), (test_x, test_y) = load_data()

    feature_columns = [
        tf.feature_column.categorical_column_with_identity(key='Pclass',
                                                           num_buckets=4),
        tf.feature_column.categorical_column_with_vocabulary_list(
            key='Sex',
            vocabulary_list=['male', 'female']),
        tf.feature_column.numeric_column(key='Age'),
        tf.feature_column.numeric_column(key='Fare'),
        tf.feature_column.categorical_column_with_vocabulary_list(
            key='Embarked',
            vocabulary_list=['C', 'Q', 'S']),
        tf.feature_column.numeric_column(key='SibSp'),
        tf.feature_column.numeric_column(key='Parch')
    ]
    optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=0.001)

    model = tf.estimator.LinearClassifier(
            feature_columns=feature_columns,
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
