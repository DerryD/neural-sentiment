# -*- coding: utf-8 -*-
"""
export PYTHONPATH="/home/dairui/workspace/neural-sentiment/:$PYTHONPATH"
python -u train.py > /tmp/ns.log 2>&1 &
tensorboard --logdir=/tmp/tb_logs
"""
from six.moves import xrange
import numpy as np
import sys
import os
import time
import tensorflow as tf
from utils.data_processor import build_data
from utils.config import Config
from models.sentiment import SentimentModel, SentimentInput
from utils.vocab_mapping import VocabMapping
import logging

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)
tf.logging.set_verbosity(tf.logging.ERROR)
path = "data/processed/"


def main():
    config = Config()
    build_data(config.max_seq_len,
               config.vocab_size)

    # create model
    logging.info('\nCreating model with:\n\tNumber of hidden layers: %d\n'
                 '\tNumber of units per layer: %d\n\tDropout: %f' % (
                    config.num_layers, config.embedding_dims, config.keep_prob))
    vocab_mapping = VocabMapping()
    vocab_size = vocab_mapping.get_size()
    print "Vocab size is: {0}".format(vocab_size)
    infile = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    # randomize data order
    print infile
    data = np.vstack((np.load(os.path.join(path, f)) for f in infile))
    np.random.shuffle(data)
    # data = data[:10000]
    logging.info('Shape of data: %s' % str(data.shape))
    num_batches = len(data) / config.batch_size

    logging.info("Number of training examples per batch: {0},\n"
                 "Number of batches per epoch: {1}".format(config.batch_size,
                                                           num_batches))
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("/tmp/tb_logs", sess.graph)
        print 'creating model...'
        sent_input = SentimentInput(config, data)
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None):
                model = SentimentModel(config, sent_input)
        with tf.name_scope("Test"):
            with tf.variable_scope("Model", reuse=True):
                m_test = SentimentModel(config, sent_input)
        sess.run(tf.global_variables_initializer())
        # model = create_model(config, sess)
        learning_rate = config.learning_rate
        lr_decay = config.lr_decay
        print 'model creation completed.'
        # train model and save to checkpoint
        print "Training started..."
        print "Maximum number of epochs to train for: {0}".format(config.max_epoch)
        print "Batch size: {0}".format(config.batch_size)
        print "Starting learning rate: {0}".format(config.learning_rate)
        print "Learning rate decay factor: {0}".format(config.lr_decay)

        step_time, loss = 0.0, 0.0
        previous_losses = []
        # Total number of batches to pass through.
        tot_steps = num_batches * config.max_epoch
        # starting at step 1 to prevent test set from running after first batch
        for step in xrange(1, tot_steps):
            # Get a batch and make a step.
            start_time = time.time()

            inputs, targets, seq_lengths = sent_input.get_batch()
            str_summary, step_loss, _ = model.step(sess, inputs, targets, seq_lengths, True)
            steps_per_checkpoint = 100
            step_time += (time.time() - start_time) / steps_per_checkpoint
            loss += step_loss / steps_per_checkpoint

            # Once in a while, we run evals.
            if step % steps_per_checkpoint == 0:
                writer.add_summary(str_summary, step)
                # Print statistics for the previous epoch.
                print ("global step %d learning rate %.7f step-time %.2f loss %.4f"
                       % (model.global_step.eval(), model.learning_rate.eval(),
                          step_time, loss))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    # sess.run(model.learning_rate_decay_op)
                    learning_rate *= lr_decay
                    model.assign_lr(sess, learning_rate)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                step_time, loss, test_accuracy = 0.0, 0.0, 0.0
                # Run evals on test set and print their accuracy.
                print "Running test set"
                for _ in xrange(len(m_test.input_data.test_data)):
                    inputs, targets, seq_lengths = sent_input.get_batch(False)
                    str_summary, test_loss, _, accuracy = m_test.step(
                        sess, inputs, targets, seq_lengths, False)
                    loss += test_loss
                    test_accuracy += accuracy
                normalized_test_loss, normalized_test_accuracy = loss / len(
                    m_test.input_data.test_data), test_accuracy / len(m_test.input_data.test_data)
                writer.add_summary(str_summary, step)
                print "Avg Test Loss: {0}, Avg Test Accuracy: {1}".format(
                    normalized_test_loss, normalized_test_accuracy)
                print "-------Step {0}/{1}------".format(step, tot_steps)
                loss = 0.0
                sys.stdout.flush()


if __name__ == '__main__':
    main()
