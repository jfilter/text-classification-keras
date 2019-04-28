from __future__ import absolute_import

from keras.layers import LSTM, Bidirectional, Conv1D, Dropout, GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPooling1D, Dense, Flatten, GRU
from keras.layers.merge import Concatenate, concatenate

from .layers import AttentionLayer
from ..utils.format import to_fixed_digits


class SequenceEncoderBase(object):

    def __init__(self, dropout_rate=0.5):
        """Creates a new instance of sequence encoder.

        Args:
            dropout_rate: The final encoded output dropout.
        """
        self.dropout_rate = dropout_rate

    def __call__(self, x):
        """Build the actual model here.

        Args:
            x: The encoded or embedded input sequence.

        Returns:
            The model output tensor.
        """

        x = self.build_model(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)
        return x

    def build_model(self, x):
        """Build your model graph here.

        Args:
            x: The encoded or embedded input sequence.

        Returns:
            The model output tensor without the classification block.
        """
        raise NotImplementedError()

    def allows_dynamic_length(self):
        """Return a boolean indicating whether this model is capable of handling variable time steps per mini-batch.

        For example, this should be True for RNN models since you can use them with variable time steps per mini-batch.
        CNNs on the other hand expect fixed time steps across all mini-batches.
        """
        # Assume default as False. Should be overridden as necessary.
        return False


class YoonKimCNN(SequenceEncoderBase):

    def __init__(self, num_filters=64, filter_sizes=[3, 4, 5], dropout_rate=0.5, **conv_kwargs):
        """Yoon Kim's shallow cnn model: https://arxiv.org/pdf/1408.5882.pdf

        Args:
            num_filters: The number of filters to use per `filter_size`. (Default value = 64)
            filter_sizes: The filter sizes for each convolutional layer. (Default value = [3, 4, 5])
            **cnn_kwargs: Additional args for building the `Conv1D` layer.
        """
        super(YoonKimCNN, self).__init__(dropout_rate)
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.conv_kwargs = conv_kwargs

    def build_model(self, x):
        pooled_tensors = []
        for filter_size in self.filter_sizes:
            x_i = Conv1D(self.num_filters, filter_size,
                         activation='elu', **self.conv_kwargs)(x)
            x_i = GlobalMaxPooling1D()(x_i)
            pooled_tensors.append(x_i)

        x = pooled_tensors[0] if len(
            self.filter_sizes) == 1 else concatenate(pooled_tensors, axis=-1)
        return x

    def __str__(self):
        conv_kwargs_str = str(self.conv_kwargs) if len(
            self.conv_kwargs) > 0 else ''
        filter_sizes_li = [str(x) for x in self.filter_sizes]
        li = ['cnn_kim', str(self.num_filters)] + filter_sizes_li + [
            'do', to_fixed_digits(self.dropout_rate), conv_kwargs_str]
        return '_'.join(li)


class AlexCNN(SequenceEncoderBase):
    def __init__(self, num_filters=20, filter_sizes=[3, 8], dropout_rate=[0.5, 0.8], hidden_dims=20, **conv_kwargs):
        """Alexander Rakhlin's CNN model: https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras/

        Args:
            num_filters: The number of filters to use per `filter_size`. (Default value = 64)
            filter_sizes: The filter sizes for each convolutional layer. (Default value = [3, 4, 5])
            dropout_rate: Array for one dropout layer after the embedding and one before the final dense layer (Default value = [0.5, 0.8])
        """
        super(AlexCNN, self).__init__(dropout_rate[0])
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.dropout_rate = dropout_rate[0]
        self.dropout_rate2 = dropout_rate[1]
        self.hidden_dims = hidden_dims
        self.conv_kwargs = conv_kwargs

    def build_model(self, x):
        conv_blocks = []
        for sz in self.filter_sizes:
            conv = Conv1D(filters=self.num_filters,
                          kernel_size=sz,
                          padding="valid",
                          activation="relu",
                          strides=1, **self.conv_kwargs)(x)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)

        x = Concatenate()(conv_blocks) if len(
            conv_blocks) > 1 else conv_blocks[0]

        x = Dropout(self.dropout_rate2)(x)
        x = Dense(self.hidden_dims, activation="relu")(x)
        return x

    def __str__(self):
        conv_kwargs_str = str(self.conv_kwargs) if len(
            self.conv_kwargs) > 0 else ''
        filter_sizes_li = [str(x) for x in self.filter_sizes]
        li = ['cnn_kim', str(self.num_filters)] + filter_sizes_li + [
            'do', to_fixed_digits(self.dropout_rate), conv_kwargs_str]
        return '_'.join(li)


class StackedRNN(SequenceEncoderBase):
    def __init__(self, rnn_class=LSTM, hidden_dims=[50, 50], bidirectional=True, dropout_rate=0.5, **rnn_kwargs):
        """Creates a stacked RNN.

        Args:
            rnn_class: The type of RNN to use. (Default Value = LSTM)
            encoder_dims: The number of hidden units of RNN. (Default Value: 50)
            bidirectional: Whether to use bidirectional encoding. (Default Value = True)
            **rnn_kwargs: Additional args for building the RNN.
        """
        super(StackedRNN, self).__init__(dropout_rate)
        self.rnn_class = rnn_class
        self.hidden_dims = hidden_dims
        self.bidirectional = bidirectional
        self.rnn_kwargs = rnn_kwargs

    def build_model(self, x):
        for i, n in enumerate(self.hidden_dims):
            is_last_layer = i == len(self.hidden_dims) - 1
            rnn = self.rnn_class(
                n, return_sequences=not is_last_layer, **self.rnn_kwargs)
            if self.bidirectional:
                x = Bidirectional(rnn)(x)
            else:
                x = rnn(x)
        return x

    def allows_dynamic_length(self):
        return True

    def __str__(self):
        bi = 'bi' if self.bidirectional else 'nobi'
        rnn_classs_str = self.rnn_class.__name__
        rnn_kwargs_str = str(self.rnn_kwargs) if len(
            self.rnn_kwargs) > 0 else ''
        hidden_dims_li = [str(x) for x in self.hidden_dims]
        li = ['stacked', rnn_classs_str] + hidden_dims_li + [
            bi, 'do', to_fixed_digits(self.dropout_rate), rnn_kwargs_str]
        return '_'.join(li)


class BasicRNN(StackedRNN):
    def __init__(self, rnn_class=LSTM, hidden_dims=50, bidirectional=True, dropout_rate=0.5, **rnn_kwargs):
        super(BasicRNN, self).__init__(rnn_class=rnn_class, hidden_dims=[
            hidden_dims], bidirectional=bidirectional, dropout_rate=dropout_rate, **rnn_kwargs)

    def __str__(self):
        bi = 'bi' if self.bidirectional else 'nobi'
        rnn_classs_str = self.rnn_class.__name__
        rnn_kwargs_str = str(self.rnn_kwargs) if len(
            self.rnn_kwargs) > 0 else ''
        hidden_dims_li = [str(x) for x in self.hidden_dims]
        li = ['basic', rnn_classs_str] + hidden_dims_li + [
            bi, 'do', to_fixed_digits(self.dropout_rate), rnn_kwargs_str]
        return '_'.join(li)


class AttentionRNN(SequenceEncoderBase):

    def __init__(self, rnn_class=LSTM, encoder_dims=50, bidirectional=True, dropout_rate=0.5, **rnn_kwargs):
        """Creates an RNN model with attention. The attention mechanism is implemented as described
        in https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf, but without
        sentence level attention.

        Args:
            rnn_class: The type of RNN to use. (Default Value = LSTM)
            encoder_dims: The number of hidden units of RNN. (Default Value: 50)
            bidirectional: Whether to use bidirectional encoding. (Default Value = True)
            **rnn_kwargs: Additional args for building the RNN.
        """
        super(AttentionRNN, self).__init__(dropout_rate)
        self.rnn_class = rnn_class
        self.encoder_dims = encoder_dims
        self.bidirectional = bidirectional
        self.rnn_kwargs = rnn_kwargs

    def build_model(self, x):
        rnn = self.rnn_class(
            self.encoder_dims, return_sequences=True, **self.rnn_kwargs)
        if self.bidirectional:
            word_activations = Bidirectional(rnn)(x)
        else:
            word_activations = rnn(x)

        attention_layer = AttentionLayer()
        doc_vector = attention_layer(word_activations)
        self.attention_tensor = attention_layer.get_attention_tensor()
        return doc_vector

    def get_attention_tensor(self):
        if not hasattr(self, 'attention_tensor'):
            raise ValueError('You need to build the model first')
        return self.attention_tensor

    def allows_dynamic_length(self):
        return True

    def __str__(self):
        bi = 'bi' if self.bidirectional else 'nobi'
        rnn_kwargs_str = str(self.rnn_kwargs) if len(
            self.rnn_kwargs) > 0 else ''
        li = ['stacked', str(self.rnn_class), str(self.encoder_dims),
              bi, 'do', to_fixed_digits(self.dropout_rate), rnn_kwargs_str]

        return '_'.join(li)


class AveragingEncoder(SequenceEncoderBase):

    def __init__(self, dropout_rate=0):
        """An encoder that averages sequence inputs.
        """
        super(AveragingEncoder, self).__init__(dropout_rate)

    def __str__(self):
        return 'avg_encoder'

    def build_model(self, x):
        x = GlobalAveragePooling1D()(x)
        return x
