import tensorflow as tf

from get_data import load_data, inp


def dnn_model_fn(features, labels, mode, params):

    top = tf.feature_column.input_layer(features, params['feature_columns'])

    for units in params.get('hidden_units', [20]):
        top = tf.layers.dense(inputs=top, units=units, activation=tf.nn.relu)

    output_layer = tf.layers.dense(inputs=top, units=1)

    predictions = tf.squeeze(output_layer, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={'death':
                                          predictions})

    average_loss = tf.losses.mean_squared_error(labels, predictions)

    batch_size = tf.shape(labels)[0]
    total_loss = tf.to_float(batch_size) * average_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = params.get('optimizer', tf.train.AdamOptimizer)
        optimizer = optimizer(params.get('learning_rate', None))
        train_op = optimizer.minimize(loss=average_loss,
                global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss,
                train_op=train_op)


def main(argv):
    (train_x, train_y), (test_x, test_y) = load_data()

    feature_columns = [
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(
                key='Pclass', num_buckets=4),
            dimension=3),
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                key='Sex',
                vocabulary_list=['male', 'female']), dimension=3),
        tf.feature_column.numeric_column(key='Age'),
        tf.feature_column.numeric_column(key='Fare'),
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                key='Embarked',
                vocabulary_list=['C', 'Q', 'S']),
            dimension=3),
        tf.feature_column.numeric_column(key='SibSp'),
        tf.feature_column.numeric_column(key='Parch')
    ]




if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
