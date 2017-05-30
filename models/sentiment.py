import tensorflow as tf
import numpy as np


class SentimentInput(object):
    def __init__(self, config, data):
        self.batch_size = config.batch_size
        # 70/30 split for train/test
        train_start_end_index = [0, int(0.7 * len(data))]
        valid_start_end_index = [int(0.7 * len(data)) + 1, len(data) - 1]
        targets = data[:, -2]
        one_hot = np.zeros((len(targets), 2))
        one_hot[np.arange(len(targets)), targets] = 1
        sequence_lengths = data[:, -1]
        data = data[:, :-2]
        self.train_data = data[train_start_end_index[0]: train_start_end_index[1]]
        self.valid_data = data[valid_start_end_index[0]:valid_start_end_index[1]]
        self.valid_num_batch = len(self.valid_data) / self.batch_size
        num_train_batches = len(self.train_data) / self.batch_size
        num_valid_batches = len(self.valid_data) / self.batch_size
        train_cutoff = len(self.train_data) - (len(self.train_data) % self.batch_size)
        valid_cutoff = len(self.valid_data) - (len(self.valid_data) % self.batch_size)
        self.train_data = self.train_data[:train_cutoff]
        self.valid_data = self.valid_data[:valid_cutoff]

        self.train_sequence_lengths = \
            sequence_lengths[train_start_end_index[0]: train_start_end_index[1]][:train_cutoff]
        self.train_sequence_lengths = np.split(self.train_sequence_lengths, num_train_batches)
        self.train_targets = one_hot[train_start_end_index[0]: train_start_end_index[1]][:train_cutoff]
        self.train_targets = np.split(self.train_targets, num_train_batches)
        self.train_data = np.split(self.train_data, num_train_batches)

        print "Validation size is: {0}, splitting into {1} batches".format(
            len(self.valid_data), num_valid_batches)
        self.valid_data = np.split(self.valid_data, num_valid_batches)
        self.valid_targets = one_hot[valid_start_end_index[0]: valid_start_end_index[1]][:valid_cutoff]
        self.valid_targets = np.split(self.valid_targets, num_valid_batches)
        self.valid_seq_len = \
            sequence_lengths[valid_start_end_index[0]: valid_start_end_index[1]][:valid_cutoff]
        self.valid_seq_len = np.split(self.valid_seq_len, num_valid_batches)
        self.valid_batch_pointer = 0
        self.train_batch_pointer = 0
        # self.train_data = tf.convert_to_tensor(self.train_data, dtype=tf.int32)
        # self.train_targets = tf.convert_to_tensor(self.train_targets, dtype=tf.int32)
        # self.valid_data = tf.convert_to_tensor(self.valid_data, dtype=tf.int32)
        # self.valid_targets = tf.convert_to_tensor(self.valid_targets, dtype=tf.int32)

    def next_batch(self, is_training=True):
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
        else:
            batch_inputs = self.valid_data[self.valid_batch_pointer]
            targets = self.valid_targets[self.valid_batch_pointer]
            seq_lengths = self.valid_seq_len[self.valid_batch_pointer]
            self.valid_batch_pointer += 1
            self.valid_batch_pointer = self.valid_batch_pointer % len(self.valid_data)
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
        self.batch_pointer = 0
        # self.global_step = tf.Variable(0, trainable=False)
        self.max_seq_length = config.max_seq_len
        # seq_input: list of tensors, each tensor is size max_seq_length
        # target: a list of values between 0 and 1 indicating target scores
        # seq_lengths:the early stop lengths of each input tensor
        self.seq_input = tf.placeholder(
            tf.int32, shape=[None, config.max_seq_len])
        self.target = tf.placeholder(
            tf.float32, shape=[None, self.num_classes])
        self.seq_lengths = tf.placeholder(tf.int32, shape=[None])

        with tf.variable_scope("embedding"), tf.device("/cpu:0"):
            # noinspection PyPep8Naming
            embedding = tf.get_variable(
                name="embedding",
                shape=[self.vocab_size, config.embedding_dims]
            )
            inputs = tf.nn.embedding_lookup(embedding, self.seq_input)
            if is_training and self.dropout < 1:
                inputs = tf.nn.dropout(
                    inputs, config.keep_prob)

        def lstm_cell():
            if config.use_proj:
                return tf.contrib.rnn.LSTMCell(
                    num_units=config.hidden_size,    # n, size of $$ c_t $$
                    num_proj=config.embedding_dims,  # p, size of $$ h_t $$
                    reuse=tf.get_variable_scope().reuse)
            else:
                return tf.contrib.rnn.LSTMCell(
                    num_units=config.embedding_dims,
                    reuse=tf.get_variable_scope().reuse)

        def attn_cell():
            if is_training and self.dropout < 1:
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(),
                    # input_keep_prob=config.keep_prob,
                    output_keep_prob=config.keep_prob
                )
            else:
                return lstm_cell

        if config.num_layers >= 2:
            cell = tf.contrib.rnn.MultiRNNCell(
                [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
        else:
            cell = attn_cell()

        initial_state = cell.zero_state(config.batch_size, tf.float32)
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
            softmax_w = tf.get_variable(
                "softmax_w",
                [config.embedding_dims, self.num_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            softmax_b = tf.get_variable(
                "softmax_b", [self.num_classes],
                initializer=tf.random_uniform_initializer(0.1))
            # we use the cell memory state for information on sentence embedding
            if config.num_layers >= 2:
                scores = tf.nn.xw_plus_b(rnn_state[-1][0], softmax_w, softmax_b)
            else:
                scores = tf.nn.xw_plus_b(rnn_state[-1], softmax_w, softmax_b)
                # scores = tf.nn.xw_plus_b(rnn_output[-1], softmax_w, softmax_b)
            self.y = tf.nn.softmax(scores)
            predictions = tf.argmax(scores, 1)

        with tf.variable_scope("loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=scores, labels=self.target)
            self._cost = tf.reduce_mean(loss)

        with tf.variable_scope("accuracy"):
            correct_predictions = tf.equal(
                predictions, tf.argmax(self.target, 1))
            self.accuracy = tf.reduce_mean(tf.cast(
                correct_predictions, "float"))

        # self.str_summary_type = tf.placeholder(tf.string, name="str_summary_type")
        loss_summ = tf.summary.scalar("loss", self.cost)
        acc_summ = tf.summary.scalar("accuracy", self.accuracy)
        self.merged = tf.summary.merge([loss_summ, acc_summ])

        if not is_training:
            return
        self.learning_rate = tf.Variable(config.learning_rate, trainable=False)
        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self.learning_rate, self._new_lr)
        params = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(self.learning_rate)
        gradients = tf.gradients(loss, params)
        if config.max_grad_norm is not None:
            gradients, norm = tf.clip_by_global_norm(gradients, config.max_grad_norm)
        self.update = opt.apply_gradients(
            zip(gradients, params),
            global_step=tf.contrib.framework.get_or_create_global_step()
        )
        lr_summ = tf.summary.scalar("learning_rate", self.learning_rate)
        self.merged = tf.summary.merge([loss_summ, acc_summ, lr_summ])

    def step(self, session, inputs, targets, seq_lengths, is_training=True):
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
        feed_dict = {
            self.seq_input: inputs,
            self.target: targets,
            self.seq_lengths: seq_lengths
        }
        if is_training:
            fetches = [self.cost, self.update, self.accuracy, self.merged]
        else:
            fetches = [self.cost, self.y, self.accuracy, self.merged]
        outputs = session.run(fetches, feed_dict)
        return outputs

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def cost(self):
        return self._cost
