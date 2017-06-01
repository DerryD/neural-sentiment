import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import os
import nltk
import utils.data_processor
import models.sentiment
import utils.vocab_mapping
import ConfigParser
import pickle

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_dir', 'data/checkpoints/', 'Directory to store/restore checkpoints')
flags.DEFINE_string('text', 'Hello World!', 'Text to sample with.')
flags.DEFINE_string('config_file', 'config.ini', 'Path to configuration file.')


def main():
    vocab_mapping = utils.vocab_mapping.VocabMapping()
    with tf.Session() as sess:
        model = load_model(sess, vocab_mapping.get_size())
        if model is None:
            return
        max_seq_length = model.max_seq_length
        test_data = [FLAGS.text.lower()]
        for text in test_data:
            data, seq_lengths, targets = prepare_text(text, max_seq_length, vocab_mapping)
            input_feed = {
                model.seq_input.name: data,
                model.target.name: targets,
                model.seq_lengths.name: seq_lengths,
                model.dropout_keep_prob_embedding.name: model.dropout,
                model.dropout_keep_prob_lstm_input.name: model.dropout,
                model.dropout_keep_prob_lstm_output.name: model.dropout
            }
            output_feed = [model.softmax]
            outputs = sess.run(output_feed, input_feed)
            score = np.argmax(outputs[0])
            probability = outputs[0].max(axis=1)[0]
            print "Value of sentiment: {0} with probability: {1}".format(score, probability)


def prepare_text(text, max_seq_length, vocab_mapping):
    """
    Input:
    text_list: a list of strings

    Returns:
    inputs, seq_lengths, targets
    """
    data = np.array([i for i in range(max_seq_length)])
    targets = []
    seq_lengths = []
    tokens = tokenize(text)
    if len(tokens) > max_seq_length:
        tokens = tokens[0: max_seq_length]

    indices = [vocab_mapping.get_index(j) for j in tokens]
    if len(indices) < max_seq_length:
        indices += [vocab_mapping.get_index("<PAD>")
                    for _ in range(max_seq_length - len(indices))]
    else:
        indices = indices[0:max_seq_length]
    seq_lengths.append(len(tokens))

    data = np.vstack((data, indices))
    targets.append(1)

    one_hot = np.zeros((len(targets), 2))
    one_hot[np.arange(len(targets)), targets] = 1
    return data[1::], np.array(seq_lengths), one_hot


def load_model(session, vocab_size):
    hyper_params_path = os.path.join(FLAGS.checkpoint_dir, 'hyperparams.p')
    with open(hyper_params_path, 'rb') as hp:
        hyper_params = pickle.load(hp)
    # hyper_params = read_config_file()
    model = models.sentiment.SentimentModel(vocab_size=vocab_size,
                                            embedding_dims=hyper_params["hidden_size"],
                                            dropout=1.0,
                                            num_layers=hyper_params["num_layers"],
                                            max_gradient_norm=hyper_params["grad_clip"],
                                            max_seq_length=hyper_params["max_seq_length"],
                                            learning_rate=hyper_params["learning_rate"],
                                            batch_size=1,
                                            is_training=True)
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
        print "Reading model parameters from {0}".format(ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print "Double check you got the checkpoint_dir right..."
        print "Model not found..."
        model = None
    return model


def read_config_file():
    """
    Reads in config file, returns dictionary of network params
    """
    config = ConfigParser.ConfigParser()
    config.read(FLAGS.config_file)
    sentiment_section = "sentiment_network_params"
    general_section = "general"
    dic = {
        "num_layers": config.getint(sentiment_section, "num_layers"),
        "hidden_size": config.getint(sentiment_section, "hidden_size"),
        "dropout": config.getfloat(sentiment_section, "dropout"),
        "batch_size": config.getint(sentiment_section, "batch_size"),
        "train_frac": config.getfloat(sentiment_section, "train_frac"),
        "learning_rate": config.getfloat(sentiment_section, "learning_rate"),
        "lr_decay_factor": config.getfloat(
            sentiment_section, "lr_decay_factor"),
        "grad_clip": config.getint(sentiment_section, "grad_clip"),
        "use_config_file_if_checkpoint_exists": config.getboolean(
            general_section, "use_config_file_if_checkpoint_exists"),
        "max_epoch": config.getint(sentiment_section, "max_epoch"),
        "max_vocab_size": config.getint(sentiment_section, "max_vocab_size"),
        "max_seq_length": config.getint(general_section, "max_seq_length"),
        "steps_per_checkpoint": config.getint(
            general_section, "steps_per_checkpoint")
    }
    return dic


def tokenize(text):
    text = text.decode('utf-8')
    return nltk.word_tokenize(text)


if __name__ == "__main__":
    main()
