# -*- coding: utf-8 -*-
"""
export PYTHONPATH="/home/dairui/workspace/neural-sentiment/:$PYTHONPATH"
python -u train.py > /tmp/ns.log 2>&1 &
tensorboard --logdir=/tmp/tb_logs
rm -rf /tmp/tb_logs
"""
from six.moves import xrange
import numpy as np
import os
import time
import tensorflow as tf
from utils.data_processor import build_data
from models.sentiment import SentimentModel, SentimentInput
from utils.vocab_mapping import VocabMapping
# import crash_on_ipy
import logging

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)
# tf.logging.set_verbosity(tf.logging.ERROR)
path = 'data/processed/'
tf.flags.DEFINE_float('learning_rate', 0.009, "initial learning rate")
tf.flags.DEFINE_integer('num_layers', 2, "number of stacked LSTM cells")
tf.flags.DEFINE_integer('embedding_dims', 50, "embedded size")
tf.flags.DEFINE_float('keep_prob', 0.5, "keeping probability in dropout")
tf.flags.DEFINE_float('lr_decay', 0.7, "learning rate decay")
tf.flags.DEFINE_integer('batch_size', 200, "number of batches per epoch")
tf.flags.DEFINE_boolean('use_gru', True,
                        'whether to use GRU instead of LSTM')
tf.flags.DEFINE_integer('fact_size', 40, 'factor size if using factorized RNN')
FLAGS = tf.flags.FLAGS


class Config(object):
    def __init__(self):
        self.learning_rate = FLAGS.learning_rate  # 0.001
        self.max_grad_norm = 5
        self.num_layers = FLAGS.num_layers  # number of stacked LSTM cells
        self.embedding_dims = FLAGS.embedding_dims  # embedded size
        self.max_epoch = 20  # Number of epochs for iteration
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
    tb_log_dir = '/tmp/tb_logs/emb-size-{:d}_num-layers-{:d}_keep-prob-{:.2f}_{:d}'.format(
        FLAGS.embedding_dims, FLAGS.num_layers, FLAGS.keep_prob, int(time.time()))
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
    logging.info('Shape of data: %s' % str(data.shape))
    sent_input = SentimentInput(config, data)
    batches_per_epoch = sent_input.num_train_batches
    logging.info('Number of training examples per batch: {0}; '
                 'number of batches per epoch: {1}.'.format(config.batch_size,
                                                            batches_per_epoch))
    with tf.Graph().as_default():
        logging.info('creating model...')
        initializer = tf.random_uniform_initializer(-1, 1)
        with tf.name_scope('Train'):
            with tf.variable_scope('Model', reuse=None, initializer=initializer):
                model = SentimentModel(config, sent_input, True)
        with tf.name_scope('Valid'):
            with tf.variable_scope('Model', reuse=True, initializer=initializer):
                m_valid = SentimentModel(config, sent_input, is_training=False)
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
            print 'Maximum number of epochs to train for: {0}'.format(epoch_num)
            print 'Batch size: {0}'.format(config.batch_size)
            print 'Starting learning rate: {0}'.format(config.learning_rate)
            print 'Learning rate decay factor: {0}'.format(config.lr_decay)
            batch_time, train_loss = 0.0, 0.0
            previous_losses = []
            # Total number of batches to pass through.
            step = 0
            logging.info('Training...')
            # starting at step 1 to prevent test set from running after first batch
            for epoch in range(epoch_num):
                for batch in range(batches_per_epoch):
                    step += 1
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
                             % (epoch, epoch_num, sess.run(model.learning_rate),
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
                for _ in xrange(m_valid.input_data.num_valid_batches):
                    inputs, targets, seq_lengths = m_valid.input_data.next_batch(False)
                    valid_loss, _, valid_accuracy, valid_summary = m_valid.run_batch(
                        sess, inputs, targets, seq_lengths, False)
                    val_loss += valid_loss
                    val_acc += valid_accuracy
                norm_valid_loss = val_loss / m_valid.input_data.num_valid_batches
                norm_valid_accuracy = val_acc / m_valid.input_data.num_valid_batches
                # noinspection PyUnboundLocalVariable
                writer.add_summary(valid_summary, step)
                logging.info('Avg Test Loss: {0}, Avg Test Accuracy: {1}'.format(
                    norm_valid_loss, norm_valid_accuracy))
                train_loss = 0.0


if __name__ == '__main__':
    # crash_on_ipy.init()
    tf.app.run()
