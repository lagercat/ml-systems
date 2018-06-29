import tensorflow as tf
from get_data import inp, load_data, load_submit, write_to_file, cabin_list


def main(argv):

    feature_columns = [
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_identity(
                key='Pclass', num_buckets=5)),
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                key='Sex',
                vocabulary_list=['male', 'female'])),
        tf.feature_column.numeric_column(key='Age'),
        tf.feature_column.numeric_column(key='Fare'),
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                key='Embarked',
                vocabulary_list=['C', 'Q', 'S'])),
        tf.feature_column.numeric_column(key='SibSp'),
        tf.feature_column.numeric_column(key='Parch'),
        tf.feature_column.numeric_column(key='agcl'),
        tf.feature_column.numeric_column(key='fsize'),
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                key='title',
                vocabulary_list=['Mr', 'Mrs', 'Miss']),
            dimension=3),
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                key='deck',
                vocabulary_list=cabin_list),
            dimension=3),
        tf.feature_column.numeric_column(key='Fare_Per_Person')
    ]

    (train_x, train_y), (test_x, test_y) = load_data()

    number_of_units = [10, 15, 20, 25, 30, 35, 40]
    learning_rate = [0.01, 0.03, 0.1, 0.3, 1, 3]
    best_units = 0
    best_rate = 0
    best_error = 2 ** 60

    for rate in learning_rate:
        for unit in number_of_units:
            units = 2 * [unit]
            optimizer = tf.train.AdagradOptimizer(learning_rate=rate)
            model = tf.estimator.DNNClassifier(hidden_units=units,
                                               feature_columns=feature_columns,
                                               optimizer=optimizer,
                                               activation_fn=tf.nn.sigmoid)

            model.train(input_fn=lambda: inp(train_x, train_y, 'TRAIN'),
                        steps=20000)

            eval_result = model.evaluate(input_fn=lambda: inp(test_x, test_y,
                                         'EVAL'))
            average_loss = eval_result['average_loss']

            print('Average loss: ' + str(average_loss))
            if average_loss < best_error:
                best_error = average_loss
                best_units = number_of_units
                best_rate = rate

    print('Best units:', best_units)
    print('Best rate:', best_rate)

    '''
    brute_results = model.predict(input_fn=lambda: inp(load_submit(),
                                  (), 'PREDICT'))
    net_results = []
    for line in brute_results:
        net_results.append(line['class_ids'][0])
    write_to_file(net_results)
    '''


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
