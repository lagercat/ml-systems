import tensorflow as tf
from get_data import inp, load_data, load_submit, write_to_file


def main(argv):

    feature_columns = [
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(
                key='Pclass', num_buckets=5),
            dimension=5),
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                key='Sex',
                vocabulary_list=['male', 'female']), dimension=4),
        tf.feature_column.numeric_column(key='Age'),
        tf.feature_column.numeric_column(key='Fare'),
        tf.feature_column.numeric_column(key='SibSp'),
        tf.feature_column.numeric_column(key='Parch')
    ]

    (train_x, train_y), (test_x, test_y) = load_data(train_test_ration=0.7)

    units = 6 * [20] + 2 * [10]
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.15)
    model = tf.estimator.DNNClassifier(hidden_units=units,
                                       feature_columns=feature_columns,
                                       optimizer=optimizer)

    model.train(input_fn=lambda: inp(train_x, train_y, 'TRAIN'),
                steps=3000)

    eval_result = model.evaluate(input_fn=lambda: inp(test_x, test_y,
                                 'EVAL'))
    average_loss = eval_result['average_loss']
    print('Average loss: ' + str(average_loss))

    brute_results = model.predict(input_fn=lambda: inp(load_submit(),
                                  (), 'PREDICT'))
    net_results = []
    for line in brute_results:
        net_results.append(line['class_ids'][0])
    write_to_file(net_results)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
