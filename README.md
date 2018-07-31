# Text Classification Keras [![Build Status](https://travis-ci.com/jfilter/text-classification-keras.svg?branch=master)](https://travis-ci.com/jfilter/text-classification-keras)

A high-level text classification library implementing various well-established models. With a clean and extendable interface to implement custom architectures. (Eventually, this is an alpha release so expect many breaking changes to get to that goal.(So better pin that version number 😉))

## Quick start

### Install

```bash
pip install text-classification-keras[full]==0.1.0
```

The `[full]` will additionally install the [TensorFlow](https://github.com/tensorflow/tensorflow), [Spacy](https://github.com/explosion/spaCy), and [Deep Plots](https://github.com/jfilter/deep-plots).

### Usage

```python
from texcla import experiment, data
from texcla.models import TokenModelFactory, YoonKimCNN
from texcla.preprocessing import FastTextWikiTokenizer

# input text
X = ['some random text', 'another random text', 'peter', ...]

# input labels
y = ['a', 'b', 'a', ...]

# use the special tokenizer used for constructing the embeddings
tokenizer = FastTextWikiTokenizer()

# preprocess data (once)
experiment.setup_data(X, y, tokenizer, 'data.bin', max_len=100)

# load data
ds = data.Dataset.load('data.bin')

# construct base
factory = TokenModelFactory(
    ds.num_classes, ds.tokenizer.token_index, max_tokens=100, embedding_type='fasttext.wiki.simple', embedding_dims=300)

# choose a model
word_encoder_model = YoonKimCNN()

# build a model
model = factory.build_model(
    token_encoder_model=word_encoder_model, trainable_embeddings=False)

# use experiment.train as wrapper for Keras.fit()
experiment.train(x=ds.X, y=ds.y, validation_split=0.1, model=model,
    word_encoder_model=word_encoder_model)
```

Checkout more [examples](./examples).

## API Documenation

<https://github.io/jfilter/text-classification-keras/>

## Advanced

### Embeddings

#### FastText

Several pre-trained FastText embeddings are included. For now, we only have the word embeddings and not the n-gram features.

-   [English Vectors](https://fasttext.cc/docs/en/english-vectors.html):
-   [Multilang Vectors](https://fasttext.cc/docs/en/crawl-vectors.html): `fasttext.cc.LANG_CODE`
-   [Wikipedia Vectors](https://fasttext.cc/docs/en/pretrained-vectors.html): `fasttext.wiki.LANG_CODE`

#### GloVe

Predecessor to FastText

[GloVe](https://nlp.stanford.edu/projects/glove/):

### Tokenzation

-   To represent you dataset as `(docs, words)` use `WordTokenizer`
-   To represent you dataset as `(docs, sentences, words)` use `SentenceWordTokenizer`
-   To create arbitrary hierarchies, extend `Tokenizer` and implement the `token_generator` method.

#### Spacy

You may use Spacy for the tokenization. See instructions on how to
[download model](https://spacy.io/docs/usage/models#download) for your target language.

```bash
python -m spacy download en
```

### Models

#### Token-based Models (Words)

#### Word based models

When dataset represented as `(docs, words)` word based models can be created using `TokenModelFactory`.

```python
from keras_text.models import TokenModelFactory
from keras_text.models import YoonKimCNN, AttentionRNN, StackedRNN


# RNN models can use `max_tokens=None` to indicate variable length words per mini-batch.
factory = TokenModelFactory(tokenizer.num_classes, tokenizer.token_index, max_tokens=100, embedding_type='glove.6B.100d')
word_encoder_model = YoonKimCNN()
model = factory.build_model(token_encoder_model=word_encoder_model)
```

Currently supported models include:

-   [Yoon Kim CNN](https://arxiv.org/abs/1408.5882)
-   Stacked RNNs
-   Attention (with/without context) based RNN encoders.

`TokenModelFactory.build_model` uses the provided word encoder which is then classified via `Dense` block.

#### Sentence-basded Models

-   [Hierarchical attention networks](http://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)
    (HANs) can be build by composing two attention based RNN models. This is useful when a document is very large.
-   For smaller document a reasonable way to encode sentences is to average words within it. This can be done by using
    `token_encoder_model=AveragingEncoder()`
-   Mix and match encoders as you see fit for your problem.

```python
# Pad max sentences per doc to 500 and max words per sentence to 200.

# Can also use `max_sents=None` to allow variable sized max_sents per mini-batch.

factory = SentenceModelFactory(10, tokenizer.token_index, max_sents=500, max_tokens=200, embedding_type='glove.6B.100d')
word_encoder_model = AttentionRNN()
sentence_encoder_model = AttentionRNN()

# Allows you to compose arbitrary word encoders followed by sentence encoder.

model = factory.build_model(word_encoder_model, sentence_encoder_model)
```

Currently supported models include:

-   [Yoon Kim CNN](https://arxiv.org/abs/1408.5882)
-   Stacked RNNs
-   Attention (with/without context) based RNN encoders.

`SentenceModelFactory.build_model` created a tiered model where words within a sentence is first encoded using
`word_encoder_model`. All such encodings per sentence is then encoded using `sentence_encoder_model`.

## Contributing

If you have a **question**, found a **bug** or want to propose a new **feature**, have a look at the [issues page](https://github.com/jfilter/text-classificaiton-keras/issues).

**Pull requests** are especially welcomed when they fix bugs or improve the code quality.

## Acknowledgements

Built upon the work by Raghavendra Kotikalapudi: [keras-text](https://github.com/raghakot/keras-text).

## Citation

Please cite Text Classification Keras in your publications if it helped your research. Here is an example BibTeX entry:

```tex
@misc{raghakotkerastext
    title={Text Classification Keras},
    author={Raghavendra Kotikalapudi, and Johannes Filter, and contributors},
    year={2018},
    publisher={GitHub},
    howpublished={\url{https://github.com/jfilter/text-classification-keras}},
}
```

## License

MIT.
