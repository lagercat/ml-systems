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

    (train_x, train_y), (test_x, test_y) = load_data(ratio=0.7)

    units = 2 * [30]
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    model = tf.estimator.DNNClassifier(hidden_units=units,
                                       feature_columns=feature_columns,
                                       optimizer=optimizer,
                                       activation_fn=tf.nn.sigmoid)

    model.train(input_fn=lambda: inp(train_x, train_y, 'TRAIN', rep=3000),
                steps=600000)

    eval_result = model.evaluate(input_fn=lambda: inp(test_x, test_y,
                                 'EVAL', rep=1))
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
