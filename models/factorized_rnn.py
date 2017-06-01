# -*- coding: utf-8 -*-
# noinspection PyProtectedMember
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
# noinspection PyProtectedMember
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear as linear
# noinspection PyProtectedMember
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _checked_scope as checked_scope
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import LSTMStateTuple
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import tanh, sigmoid
from tensorflow.python.platform import tf_logging as logging


class FLSTMCell(RNNCell):
    """Factorized LSTM cell described in "FACTORIZATION TRICKS FOR LSTM NETWORKS", ICLR 2017 workshop
    https://openreview.net/pdf?id=ByxWXyNFg.
    """

    def __init__(self, num_units,
                 factor_size,
                 initializer=None,
                 num_proj=None,
                 forget_bias=1.0,
                 activation=tanh,
                 reuse=None):
        """
        Initializes parameters of F-LSTM cell
        :param num_units: int, The number of units in the G-LSTM cell
        :param initializer: (optional) The initializer to use for the weight and
            projection matrices.
        :param num_proj: (optional) int, The output dimensionality for the projection
            matrices.  If None, no projection is performed.
        :param factor_size: factorization size
        :param forget_bias: Biases of the forget gate are initialized by default to 1
            in order to reduce the scale of forgetting at the beginning of
            the training.
        :param activation: Activation function of the inner states.
        """

        self._num_units = num_units
        self._initializer = initializer
        self._num_proj = num_proj
        self._forget_bias = forget_bias
        self._activation = activation
        self._factor_size = factor_size
        self._reuse = reuse

        assert (self._num_units > self._factor_size)
        if self._num_proj:
            assert (self._num_proj > self._factor_size)

        if num_proj:
            self._state_size = (LSTMStateTuple(num_units, num_proj))
            self._output_size = num_proj
        else:
            self._state_size = (LSTMStateTuple(num_units, num_units))
            self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):
        """Run one step of F-LSTM.

        Args:
          inputs: input Tensor, 2D, batch x num_units.
          state: this must be a tuple of state Tensors, both `2-D`, with column sizes `c_state` and `m_state`.
          scope: not used

        Returns:
          A tuple containing:

          - A `2-D, [batch x output_dim]`, Tensor representing the output of the
            F-LSTM after reading `inputs` when previous state was `state`.
            Here output_dim is:
               num_proj if num_proj was set,
               num_units otherwise.
          - Tensor(s) representing the new state of F-LSTM after reading `inputs` when
            the previous state was `state`.  Same type and shape(s) as `state`.

        Raises:
          ValueError: If input size cannot be inferred from inputs via
            static shape inference.
        """
        (c_prev, m_prev) = state

        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
        with checked_scope(self, scope or "flstm_cell",
                           initializer=self._initializer,
                           reuse=self._reuse):
            # Factorization
            with vs.variable_scope("factor"):
                # multiply with W1
                fact = linear([inputs, m_prev], self._factor_size, False)
            # multiply with W2
            concat = linear(fact, 4 * self._num_units, True)
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

        c = sigmoid(
            f + self._forget_bias) * c_prev + sigmoid(i) * tanh(j)
        m = sigmoid(o) * self._activation(c)

        if self._num_proj is not None:
            with vs.variable_scope("projection"):
                m = linear(m, self._num_proj, bias=False)

        new_state = LSTMStateTuple(c, m)
        return m, new_state


class FGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, num_units, factor_size, input_size=None, activation=tanh, reuse=None):
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._activation = activation
        self._reuse = reuse
        self._factor_size = factor_size

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with checked_scope(self, scope or "fgru_cell", reuse=self._reuse):
            with vs.variable_scope("gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                with vs.variable_scope("factor"):
                    # multiply with W1
                    gate_fact = linear([inputs, state], self._factor_size, False)
                value = sigmoid(linear(
                    gate_fact, 2 * self._num_units, True, 1.0))
                r, u = array_ops.split(
                    value=value,
                    num_or_size_splits=2,
                    axis=1)
            with vs.variable_scope("candidate"):
                if self._factor_size < 2./3 * self._num_units:
                    with vs.variable_scope("factor"):
                        c_fact = linear([inputs, r * state], self._factor_size, False)
                    c = self._activation(linear(c_fact, self._num_units, True))
                else:
                    c = self._activation(linear([inputs, r * state],
                                                self._num_units, True))
            new_h = u * state + (1 - u) * c
        return new_h, new_h


class FRNNCell(RNNCell):
    """The most basic RNN cell."""

    def __init__(self, num_units, factor_size, input_size=None, activation=tanh, reuse=None):
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._activation = activation
        self._reuse = reuse
        self._factor_size = factor_size

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
        with checked_scope(self, scope or "basic_rnn_cell", reuse=self._reuse):
            with vs.variable_scope("factor"):
                # multiply with W1
                fact = linear([inputs, state], self._factor_size, False)
            output = self._activation(
                linear(fact, self._num_units, True))
        return output, output
