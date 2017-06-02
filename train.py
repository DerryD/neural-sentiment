# -*- coding: utf-8 -*-
"""

"""
from six.moves import xrange
import numpy as np
import os
import time
import tensorflow as tf
from utils.data_processor import build_data
from models.sentiment import SentimentModel, SentInput
from utils.vocab_mapping import VocabMapping
# import crash_on_ipy
import logging

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)
# tf.logging.set_verbosity(tf.logging.ERROR)
path = 'data/processed/'
tf.flags.DEFINE_float('learning_rate', 0.0003, "initial learning rate")            # TODO: 0.009
tf.flags.DEFINE_integer('num_layers', 1, "number of stacked LSTM cells")         # TODO: 2
tf.flags.DEFINE_integer('embedding_dims', 60, "embedded size")                   # TODO: 50
tf.flags.DEFINE_float('keep_prob', 0.5, "keeping probability in dropout")        # TODO: 0.5
tf.flags.DEFINE_float('lr_decay', 0.7, "learning rate decay")
tf.flags.DEFINE_integer('batch_size', 100, "number of batches per epoch")
tf.flags.DEFINE_boolean('use_gru', False,                                        # TODO: True
                        'whether to use GRU instead of LSTM')
tf.flags.DEFINE_integer('fact_size', 0, 'factor size if using factorized RNN')   # TODO: 40
tf.flags.DEFINE_integer('max_epoch', 50, 'number of epochs')                     # TODO: 20
FLAGS = tf.flags.FLAGS


class Config(object):
    def __init__(self):
        self.learning_rate = FLAGS.learning_rate  # 0.001
        self.max_grad_norm = 5
        self.num_layers = FLAGS.num_layers  # number of stacked LSTM cells
        self.embedding_dims = FLAGS.embedding_dims  # embedded size
        self.max_epoch = FLAGS.max_epoch  # Number of epochs for iteration
        self.keep_prob = FLAGS.keep_prob
        self.lr_decay = FLAGS.lr_decay
        self.batch_size = FLAGS.batch_size
        self.num_classes = 2
        self.vocab_size = 20000
        self.max_seq_len = 200
        self.use_gru = FLAGS.use_gru
        self.fact_size = FLAGS.fact_size


def main(_):
    config = Config()
    build_data(config.max_seq_len,
               config.vocab_size)

    # create model
    tb_log_dir = '/tmp/tb_logs/emb-{:d}_layers-{:d}_dropout-{:.2f}_epoch-{:d}'.format(
        FLAGS.embedding_dims, FLAGS.num_layers, FLAGS.keep_prob, FLAGS.max_epoch)
    if config.use_gru:
        tb_log_dir += '_use-gru'
    if config.fact_size:
        tb_log_dir += '_fact-size-%d' % config.fact_size
    tb_log_dir += '_%d' % int(time.time())
    logging.info('To visualize on tensorboard, run:\ntensorboard --logdir=%s' % tb_log_dir)
    logging.info('Model params: number of hidden layers: %d;'
                 ' number of units per layer: %d; dropout: %.2f.' % (
                     config.num_layers, config.embedding_dims, config.keep_prob))
    vocab_mapping = VocabMapping()
    vocab_size = vocab_mapping.get_size()
    logging.info("Vocab size is: {0}".format(vocab_size))
    infile = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    # randomize data order
    print infile
    data = np.vstack((np.load(os.path.join(path, f)) for f in infile))
    np.random.shuffle(data)
    # data = data[:1000]
    n_samples = len(data)
    test_count = int(n_samples * 0.2)
    test_data = data[:test_count]
    train_data = data[test_count:]
    valid_count = int(n_samples * 0.2)
    valid_data = train_data[:valid_count]
    train_data = train_data[valid_count:]
    logging.info('Shape of data: %s' % str(data.shape))
    train_input = SentInput(config, train_data)
    valid_input = SentInput(config, valid_data)
    test_input = SentInput(config, test_data)
    batches_per_epoch = train_input.num_batches
    logging.info('Number of training examples per batch: {0}; '
                 'number of batches per epoch: {1}.'.format(config.batch_size,
                                                            batches_per_epoch))
    with tf.Graph().as_default():
        logging.info('Creating model...')
        initializer = tf.random_uniform_initializer(-1, 1)
        logging.info('Building training model...')
        with tf.name_scope('Train'):
            with tf.variable_scope('Model', reuse=None, initializer=initializer):
                model = SentimentModel(config, train_input, True)
        logging.info('Building validation model...')
        with tf.name_scope('Valid'):
            with tf.variable_scope('Model', reuse=True, initializer=initializer):
                m_valid = SentimentModel(config, valid_input, is_training=False)
        with tf.name_scope('Test'):
            with tf.variable_scope('Model', reuse=True, initializer=initializer):
                m_test = SentimentModel(config, test_input, is_training=False)
        logging.info('Initializing...')
        global_init = tf.global_variables_initializer()
        # sv = tf.train.Supervisor()  # logdir=tb_log_dir
        with tf.Session() as sess:
            sess.run(global_init)
            writer = tf.summary.FileWriter(tb_log_dir, sess.graph)
            learning_rate = config.learning_rate
            lr_decay = config.lr_decay
            logging.info('Model creation completed.')
            # train model and save to checkpoint
            epoch_num = config.max_epoch
            logging.info('Maximum number of epochs to train for: '
                         '{0}; Batch size: {0}; Starting learning '
                         'rate: {0}; Learning rate decay factor: '
                         '{0}'.format((epoch_num, config.batch_size,
                                       config.learning_rate, config.lr_decay)))
            batch_time, train_loss = 0.0, 0.0
            previous_losses = []
            # Total number of batches to pass through.
            step = 0
            logging.info('Training...')
            # starting at step 1 to prevent test set from running after first batch
            for epoch in xrange(epoch_num):
                for batch in xrange(batches_per_epoch):
                    step += 1
                    # if step == 2:
                    #     pass
                    # Get a batch and make a step.
                    start_time = time.time()
                    inputs, targets, seq_lengths = model.input_data.next_batch()
                    batch_loss, _, accuracy, train_summary = model.run_batch(
                        sess, inputs, targets, seq_lengths, True)
                    # average training time for each batch
                    batch_time += (time.time() - start_time) / batches_per_epoch
                    train_loss += batch_loss / batches_per_epoch
                    writer.add_summary(train_summary, step)
                    # Once in a while, we run evals.
                # Print statistics for the previous epoch.
                logging.info('Epoch [%d/%d]: learning rate %.7f, step-time %.2f, loss %.4f'
                             % (epoch + 1, epoch_num, sess.run(model.learning_rate),
                                batch_time, train_loss))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and train_loss > max(previous_losses[-3:]):
                    # sess.run(model.learning_rate_decay_op)
                    learning_rate *= lr_decay
                    model.assign_lr(sess, learning_rate)
                previous_losses.append(train_loss)
                # Save checkpoint and zero timer and loss.
                batch_time, val_loss, val_acc = 0.0, 0.0, 0.0
                # Run evals on test set and print their accuracy.
                for _ in xrange(m_valid.input_data.num_batches):
                    inputs, targets, seq_lengths = m_valid.input_data.next_batch()
                    valid_loss, _, valid_accuracy, valid_summary = m_valid.run_batch(
                        sess, inputs, targets, seq_lengths, False)
                    val_loss += valid_loss
                    val_acc += valid_accuracy
                norm_valid_loss = val_loss / m_valid.input_data.num_batches
                norm_valid_accuracy = val_acc / m_valid.input_data.num_batches
                # noinspection PyUnboundLocalVariable
                writer.add_summary(valid_summary, step)
                logging.info('Avg valid loss: {0}, Avg valid accuracy: {1}.'.format(
                    norm_valid_loss, norm_valid_accuracy))
                train_loss = 0.0
            test_loss, test_acc = 0.0, 0.0
            for _ in xrange(m_test.input_data.num_batches):
                inputs, targets, seq_lengths = m_test.input_data.next_batch()
                loss, _, acc, _ = m_test.run_batch(
                    sess, inputs, targets, seq_lengths, False)
                test_loss += loss
                test_acc += acc
            norm_test_loss = test_loss / m_test.input_data.num_batches
            norm_test_accuracy = test_acc / m_test.input_data.num_batches
            logging.info('Avg Test Loss: {0}, Avg Test Accuracy: {1}'.format(
                norm_test_loss, norm_test_accuracy))


if __name__ == '__main__':
    # crash_on_ipy.init()
    tf.app.run()
