import tensorflow as tf
import numpy as np
from factorized_rnn import FLSTMCell, FGRUCell


# class SentimentInput(object):
#     def __init__(self, config, data, validation_split=0.25):
#         self.batch_size = config.batch_size
#         training_split = 1 - validation_split
#         train_start_end_index = [0, int(training_split * len(data))]
#         valid_start_end_index = [int(training_split * len(data)) + 1, len(data) - 1]
#         targets = data[:, -2]
#         one_hot = np.zeros((len(targets), 2))
#         one_hot[np.arange(len(targets)), targets] = 1
#         sequence_lengths = data[:, -1]
#         # tokenized text data
#         data = data[:, :-2]
#         # training data
#         self.train_data = data[train_start_end_index[0]: train_start_end_index[1]]
#         # validation data
#         self.valid_data = data[valid_start_end_index[0]:valid_start_end_index[1]]
#         self.num_train_batches = len(self.train_data) // self.batch_size
#         self.num_valid_batches = len(self.valid_data) // self.batch_size
#         train_cutoff = len(self.train_data) - (len(self.train_data) % self.batch_size)
#         valid_cutoff = len(self.valid_data) - (len(self.valid_data) % self.batch_size)
#         # cut off data
#         self.train_data = self.train_data[:train_cutoff]
#         self.valid_data = self.valid_data[:valid_cutoff]
#         # sequence lengths
#         self.train_sequence_lengths = \
#             sequence_lengths[train_start_end_index[0]: train_start_end_index[1]][:train_cutoff]
#         self.train_sequence_lengths = np.split(self.train_sequence_lengths, self.num_train_batches)
#         self.train_targets = one_hot[train_start_end_index[0]: train_start_end_index[1]][:train_cutoff]
#         self.train_targets = np.split(self.train_targets, self.num_train_batches)
#         self.train_data = np.split(self.train_data, self.num_train_batches)
#
#         print "Validation size is: {0}, splitting into {1} batches".format(
#             len(self.valid_data), self.num_valid_batches)
#         self.valid_data = np.split(self.valid_data, self.num_valid_batches)
#         self.valid_targets = one_hot[valid_start_end_index[0]: valid_start_end_index[1]][:valid_cutoff]
#         self.valid_targets = np.split(self.valid_targets, self.num_valid_batches)
#         self.valid_seq_len = \
#             sequence_lengths[valid_start_end_index[0]: valid_start_end_index[1]][:valid_cutoff]
#         self.valid_seq_len = np.split(self.valid_seq_len, self.num_valid_batches)
#         self.valid_batch_pointer = 0
#         self.train_batch_pointer = 0
#
#     def next_batch(self, is_training=True):
#         """
#         Get a random batch of data to preprocess for a step
#         not sure how efficient this is...
#
#         Input:
#         data: shuffled batch * n * m numpy array of data
#         train_data: flag indicating whether or not to increment batch pointer, in other
#             word whether to return the next training batch, or cross val data
#
#         Returns:
#         A numpy arrays for inputs, target, and seq_lengths
#
#         """
#         if is_training:
#             batch_inputs = self.train_data[self.train_batch_pointer]
#             targets = self.train_targets[self.train_batch_pointer]
#             seq_lengths = self.train_sequence_lengths[self.train_batch_pointer]
#             self.train_batch_pointer += 1
#             self.train_batch_pointer = self.train_batch_pointer % len(
#                 self.train_data)
#         else:
#             batch_inputs = self.valid_data[self.valid_batch_pointer]
#             targets = self.valid_targets[self.valid_batch_pointer]
#             seq_lengths = self.valid_seq_len[self.valid_batch_pointer]
#             self.valid_batch_pointer += 1
#             self.valid_batch_pointer = self.valid_batch_pointer % len(self.valid_data)
#         return batch_inputs, targets, seq_lengths


class SentInput(object):
    def __init__(self, config, data):
        self.batch_size = config.batch_size
        targets = data[:, -2]
        one_hot = np.zeros((len(targets), 2))
        one_hot[np.arange(len(targets)), targets] = 1
        sequence_lengths = data[:, -1]
        # tokenized text data
        data = data[:, :-2]
        # training data
        self.data = data
        # validation data
        self.num_batches = len(self.data) // self.batch_size
        cutoff = len(self.data) - (len(self.data) % self.batch_size)
        # cut off data
        self.data = self.data[:cutoff]
        # sequence lengths
        self.seq_len = sequence_lengths[:cutoff]
        self.seq_len = np.split(self.seq_len, self.num_batches)
        self.targets = one_hot[:cutoff]
        self.targets = np.split(self.targets, self.num_batches)
        self.data = np.split(self.data, self.num_batches)
        self.batch_pointer = 0

    def next_batch(self):
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
        batch_inputs = self.data[self.batch_pointer]
        targets = self.targets[self.batch_pointer]
        seq_lengths = self.seq_len[self.batch_pointer]
        self.batch_pointer += 1
        self.batch_pointer = self.batch_pointer % len(self.data)
        return batch_inputs, targets, seq_lengths


class SentimentModel(object):
    def __init__(self, config, data, is_training=True):
        """
        :type config: Config
        :type data: SentInput
        :type is_training: bool
        """
        self.input_data = data
        self.num_classes = 2
        self.dropout = config.keep_prob
        self.vocab_size = config.vocab_size
        self.batch_pointer = 0
        # self.global_step = tf.Variable(0, trainable=False)
        self.max_seq_length = config.max_seq_len
        # seq_input: list of tensors, each tensor is size max_seq_length
        # target: a list of values between 0 and 1 indicating target logits
        # seq_lengths:the early stop lengths of each input tensor
        self.seq_input = tf.placeholder(
            tf.int32, shape=[None, config.max_seq_len])
        self.target = tf.placeholder(
            tf.float32, shape=[None, self.num_classes])
        self.seq_lengths = tf.placeholder(tf.int32, shape=[None])

        with tf.variable_scope("embedding"), tf.device("/cpu:0"):
            embedding = tf.get_variable(
                name="embedding",
                shape=[self.vocab_size, config.embedding_dims]
            )
            inputs = tf.nn.embedding_lookup(embedding, self.seq_input)
            if is_training and self.dropout < 1:
                inputs = tf.nn.dropout(
                    inputs, config.keep_prob)

        def rnn_cell():
            if not config.use_gru:
                if config.fact_size:
                    return FLSTMCell(
                        num_units=config.embedding_dims,
                        factor_size=config.fact_size,
                        reuse=tf.get_variable_scope().reuse)
                else:
                    return tf.contrib.rnn.LSTMCell(
                            num_units=config.embedding_dims,
                            reuse=tf.get_variable_scope().reuse)
            else:
                if config.fact_size:
                    return FGRUCell(
                        num_units=config.embedding_dims,
                        factor_size=config.fact_size,
                        reuse=tf.get_variable_scope().reuse
                    )
                else:
                    return tf.contrib.rnn.GRUCell(
                        num_units=config.embedding_dims,
                        reuse=tf.get_variable_scope().reuse)
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    rnn_cell(),
                    output_keep_prob=config.keep_prob)
        else:
            attn_cell = rnn_cell

        if config.num_layers >= 2:
            cell = tf.contrib.rnn.MultiRNNCell(
                [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
        else:
            cell = attn_cell()

        initial_state = cell.zero_state(config.batch_size, tf.float32)
        with tf.variable_scope('rnn'):
            rnn_input = tf.unstack(inputs, num=self.max_seq_length, axis=1)
            # >>> lstm_state
            # (
            #     LSTMStateTuple(
            #         c=<tf.Tensor 'Train/Model/rnn/rnn/cond_199/Merge_1:0'
            #            shape=(batch_size, embedding_dims) dtype=float32>,
            #         h=<tf.Tensor 'Train/Model/rnn/rnn/cond_199/Merge_2:0'
            #            shape=(batch_size, embedding_dims) dtype=float32>
            #      ),
            #      LSTMStateTuple(
            #         c=<tf.Tensor 'Train/Model/rnn/rnn/cond_199/Merge_3:0'
            #            shape=(batch_size, embedding_dims) dtype=float32>,
            #         h=<tf.Tensor 'Train/Model/rnn/rnn/cond_199/Merge_4:0'
            #            shape=(batch_size, embedding_dims) dtype=float32>)
            # )
            #
            # >>> gru_state
            # (<tf.Tensor 'Train/Model/rnn/rnn/cond_199/Merge_1:0'
            #   shape=(batch_size, embedding_dims) dtype=float32>,
            #  <tf.Tensor 'Train/Model/rnn/rnn/cond_199/Merge_2:0'
            #   shape=(batch_size, embedding_dims) dtype=float32>)
            # each input leads to an output
            rnn_outputs, rnn_state = tf.contrib.rnn.static_rnn(
                cell, rnn_input,
                initial_state=initial_state,
                sequence_length=self.seq_lengths,
                scope=tf.get_variable_scope()
            )

        with tf.variable_scope('softmax'):
            softmax_w = tf.get_variable(
                'softmax_w',
                [config.embedding_dims, self.num_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            softmax_b = tf.get_variable(
                'softmax_b', [self.num_classes],
                initializer=tf.constant_initializer(0.1))
            # use constant initializer to avoid log zero
            # we use the cell memory state for information on sentence embedding

            if config.num_layers >= 2:
                if config.use_gru:
                    logits = tf.nn.xw_plus_b(rnn_state[-1], softmax_w, softmax_b)
                    # logits = tf.nn.xw_plus_b(rnn_outputs[-1], softmax_w, softmax_b)
                else:
                    # the last lstm layer, the state of c
                    logits = tf.nn.xw_plus_b(rnn_state[-1][0], softmax_w, softmax_b)
            else:
                if config.use_gru:
                    logits = tf.nn.xw_plus_b(rnn_state, softmax_w, softmax_b)
                    # logits = tf.nn.xw_plus_b(rnn_outputs[-1], softmax_w, softmax_b)
                else:
                    # the state of c is the first element of LSTMTuple(c, h)
                    logits = tf.nn.xw_plus_b(rnn_state[0], softmax_w, softmax_b)

            # shape = [logits.get_shape()[0], 2]
            # epsilon = tf.constant(value=1e-5, shape=shape)
            # logits += epsilon

            self.softmax = tf.nn.softmax(logits)
            predictions = tf.argmax(logits, 1)

        with tf.variable_scope('loss'):

            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=self.target)

            # loss = -tf.reduce_sum(self.target * tf.log(tf.clip_by_value(self.softmax, 1e-5, 5.0)))

            # log_softmax = tf.nn.log_softmax(logits)
            # loss = -tf.reduce_sum(self.target * log_softmax, reduction_indices=[1])

            self._cost = tf.reduce_mean(loss)

        with tf.variable_scope('accuracy'):
            correct_predictions = tf.equal(
                predictions, tf.argmax(self.target, 1))
            self.accuracy = tf.reduce_mean(tf.cast(
                correct_predictions, 'float'))

        loss_summ = tf.summary.scalar('loss', self.cost)
        acc_summ = tf.summary.scalar('accuracy', self.accuracy)
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

    def run_batch(self, session, inputs, targets, seq_lengths, is_training=True):
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
            fetches = [self.cost, self.softmax, self.accuracy, self.merged]
        outputs = session.run(fetches, feed_dict)
        # if math.isnan(outputs[0]):
        #     import pdb
        #     pdb.set_trace()
        return outputs

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def cost(self):
        return self._cost
