import tensorflow as tf
import numpy as np


class SentimentInput(object):
    def __init__(self, config, data):
        self.batch_size = config.batch_size
        # 70/30 split for train/test
        train_start_end_index = [0, int(0.7 * len(data))]
        test_start_end_index = [int(0.7 * len(data)) + 1, len(data) - 1]
        targets = data[:, -2]
        one_hot = np.zeros((len(targets), 2))
        one_hot[np.arange(len(targets)), targets] = 1
        sequence_lengths = data[:, -1]
        data = data[:, :-2]
        self.train_data = data[train_start_end_index[0]: train_start_end_index[1]]
        self.test_data = data[test_start_end_index[0]:test_start_end_index[1]]
        self.test_num_batch = len(self.test_data) / self.batch_size
        num_train_batches = len(self.train_data) / self.batch_size
        num_test_batches = len(self.test_data) / self.batch_size
        train_cutoff = len(self.train_data) - (len(self.train_data) % self.batch_size)
        test_cutoff = len(self.test_data) - (len(self.test_data) % self.batch_size)
        self.train_data = self.train_data[:train_cutoff]
        self.test_data = self.test_data[:test_cutoff]

        self.train_sequence_lengths = \
            sequence_lengths[train_start_end_index[0]: train_start_end_index[1]][:train_cutoff]
        self.train_sequence_lengths = np.split(self.train_sequence_lengths, num_train_batches)
        self.train_targets = one_hot[train_start_end_index[0]: train_start_end_index[1]][:train_cutoff]
        self.train_targets = np.split(self.train_targets, num_train_batches)
        self.train_data = np.split(self.train_data, num_train_batches)

        print "Test size is: {0}, splitting into {1} batches".format(
            len(self.test_data), num_test_batches)
        self.test_data = np.split(self.test_data, num_test_batches)
        self.test_targets = one_hot[test_start_end_index[0]: test_start_end_index[1]][:test_cutoff]
        self.test_targets = np.split(self.test_targets, num_test_batches)
        self.test_sequence_lengths = \
            sequence_lengths[test_start_end_index[0]: test_start_end_index[1]][:test_cutoff]
        self.test_sequence_lengths = np.split(self.test_sequence_lengths, num_test_batches)
        self.test_batch_pointer = 0
        self.train_batch_pointer = 0

    def get_batch(self, is_training=True):
        """
        Get a random batch of data to preprocess for a step
        not sure how efficient this is...

        Input:
        data: shuffled batch * n * m numpy array of data
        train_data: flag indicating whether or not to increment batch pointer, in other
            word whether to return the next training batch, or cross val data

        Returns:
        A numpy arrays for inputs, target, and seq_lengths

        """
        if is_training:
            batch_inputs = self.train_data[self.train_batch_pointer]
            targets = self.train_targets[self.train_batch_pointer]
            seq_lengths = self.train_sequence_lengths[self.train_batch_pointer]
            self.train_batch_pointer += 1
            self.train_batch_pointer = self.train_batch_pointer % len(
                self.train_data)
            return batch_inputs, targets, seq_lengths
        else:
            batch_inputs = self.test_data[self.test_batch_pointer]
            targets = self.test_targets[self.test_batch_pointer]
            seq_lengths = self.test_sequence_lengths[self.test_batch_pointer]
            self.test_batch_pointer += 1
            self.test_batch_pointer = self.test_batch_pointer % len(self.test_data)
            return batch_inputs, targets, seq_lengths


class SentimentModel(object):
    """
    Sentiment Model
    params:
    vocab_size: size of vocabulary
    hidden_size: number of units in a hidden layer
    num_layers: number of hidden lstm layers
    max_gradient_norm: maximum size of gradient
    max_seq_length: the maximum length of the input sequence
    learning_rate: the learning rate to use in param adjustment
    lr_decay: rate at which to decay learning rate
    is_training: whether to run backward pass or not
    """

    def __init__(self, config, data, is_training=True):
        """
        
        :param config: 
        :type data: SentimentInput
        :param is_training: 
        """
        self.input_data = data
        self.num_classes = 2
        self.dropout = config.keep_prob
        self.vocab_size = config.vocab_size
        self.learning_rate = tf.Variable(config.learning_rate, trainable=False)
        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self.learning_rate, self._new_lr)
        initializer = tf.random_uniform_initializer(-1, 1)
        self.batch_pointer = 0
        self.batch_size = config.batch_size
        self.max_grad_norm = config.max_grad_norm
        self.global_step = tf.Variable(0, trainable=False)
        self.max_seq_length = config.max_seq_len

        # seq_input: list of tensors, each tensor is size max_seq_length
        # target: a list of values between 0 and 1 indicating target scores
        # seq_lengths:the early stop lengths of each input tensor
        self.seq_input = tf.placeholder(
            tf.int32, shape=[None, config.max_seq_len], name="input")
        self.target = tf.placeholder(
            tf.float32, name="target", shape=[None, self.num_classes])
        self.seq_lengths = tf.placeholder(tf.int32, shape=[None],
                                          name="early_stop")

        with tf.variable_scope("embedding"), tf.device("/cpu:0"):
            # noinspection PyPep8Naming
            embedding = tf.get_variable("embedding", [self.vocab_size, config.embedding_dims],
                                        initializer=initializer)
            inputs = tf.nn.embedding_lookup(embedding, self.seq_input)
            if is_training and self.dropout < 1:
                inputs = tf.nn.dropout(
                    inputs, config.keep_prob)

        def lstm_cell():
            return tf.contrib.rnn.LSTMCell(
                config.embedding_dims,
                initializer=initializer,
                state_is_tuple=True,
                reuse=tf.get_variable_scope().reuse)

        if is_training and self.dropout < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(),
                    output_keep_prob=config.keep_prob
                )
        else:
            attn_cell = lstm_cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        initial_state = cell.zero_state(self.batch_size, tf.float32)
        with tf.variable_scope("lstm"):
            rnn_input = tf.unstack(inputs, num=self.max_seq_length, axis=1)
            rnn_output, rnn_state = tf.contrib.rnn.static_rnn(
                cell, rnn_input,
                initial_state=initial_state,
                sequence_length=self.seq_lengths,
                scope=tf.get_variable_scope()
            )

        with tf.variable_scope("softmax"):
            # noinspection PyPep8Naming
            softmax_w = tf.get_variable("softmax_w", [config.embedding_dims, self.num_classes],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
            softmax_b = tf.get_variable("softmax_b", [self.num_classes],
                                        initializer=tf.constant_initializer(0.1))
            # we use the cell memory state for information on sentence embedding
            self.scores = tf.nn.xw_plus_b(rnn_state[-1][0], softmax_w, softmax_b)
            self.y = tf.nn.softmax(self.scores)
            self.predictions = tf.argmax(self.scores, 1)

        with tf.variable_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores, labels=self.target, name="ce_losses")
            self.total_loss = tf.reduce_sum(self.losses)
            self.mean_loss = tf.reduce_mean(self.losses)

        with tf.variable_scope("accuracy"):
            self.correct_predictions = tf.equal(
                self.predictions, tf.argmax(self.target, 1))
            self.accuracy = tf.reduce_mean(tf.cast(
                self.correct_predictions, "float"), name="accuracy")

        params = tf.trainable_variables()
        if is_training:
            with tf.name_scope("train"):
                opt = tf.train.AdamOptimizer(self.learning_rate)
            gradients = tf.gradients(self.losses, params)
            clipped_gradients, norm = tf.clip_by_global_norm(
                gradients, self.max_grad_norm)
            with tf.name_scope("grad_norms"):
                tf.summary.scalar("grad_norms", norm)
            self.update = opt.apply_gradients(zip(
                clipped_gradients, params), global_step=self.global_step)

    def step(self, session, inputs, targets, seq_lengths, is_training=False):
        """
        Inputs:
        session: tensorflow session
        inputs: list of list of ints representing tokens in review of batch_size
        output: list of sentiment scores
        seq_lengths: list of sequence lengths, provided at runtime to prevent need for padding

        Returns:
        merged_tb_vars, loss, none
        or (in forward only):
        merged_tb_vars, loss, outputs
        """
        input_feed = {
            self.seq_input.name: inputs,
            self.target.name: targets,
            self.seq_lengths.name: seq_lengths
        }
        if is_training:
            output_feed = [self.mean_loss, self.update, self.accuracy]
        else:
            output_feed = [self.mean_loss, self.y, self.accuracy]
        outputs = session.run(output_feed, input_feed)
        return outputs[0], outputs[1], outputs[2]

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
