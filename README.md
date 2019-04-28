# Text Classification Keras [![Build Status](https://travis-ci.com/jfilter/text-classification-keras.svg?branch=master)](https://travis-ci.com/jfilter/text-classification-keras) [![PyPI](https://img.shields.io/pypi/v/text-classification-keras.svg)](https://pypi.org/project/text-classification-keras/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/text-classification-keras.svg)](https://pypi.org/project/text-classification-keras/) [![Gitter](https://img.shields.io/gitter/room/text-classification-keras/Lobby.svg)](https://gitter.im/text-classification-keras/Lobby)

A high-level text classification library implementing various well-established models. With a clean and extendable interface to implement custom architectures.

## Quick start

### Install

```bash
pip install text-classification-keras[full]
```

The `[full]` will additionally install [TensorFlow](https://github.com/tensorflow/tensorflow), [Spacy](https://github.com/explosion/spaCy), and [Deep Plots](https://github.com/jfilter/deep-plots). Choose this if you want to get started right away.

### Usage

```python
from texcla import experiment, data
from texcla.models import TokenModelFactory, YoonKimCNN
from texcla.preprocessing import FastTextWikiTokenizer

# input text
X = ['some random text', 'another random text lala', 'peter', ...]

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
    ds.num_classes, ds.tokenizer.token_index, max_tokens=100,
    embedding_type='fasttext.wiki.simple', embedding_dims=300)

# choose a model
word_encoder_model = YoonKimCNN()

# build a model
model = factory.build_model(
    token_encoder_model=word_encoder_model, trainable_embeddings=False)

# use experiment.train as wrapper for Keras.fit()
experiment.train(x=ds.X, y=ds.y, validation_split=0.1, model=model,
    word_encoder_model=word_encoder_model)
```

Check out more [examples](./examples).

## API Documenation

<https://github.io/jfilter/text-classification-keras/>

## Advanced

### Embeddings

Choose a pre-trained word embedding by setting the embedding_type and the corresponding embedding dimensions. Set `embedding_type=None` to initialize the word embeddings randomly (but make sure to set `trainable_embeddings=True` so you actually train the embeddings).

```python
factory = TokenModelFactory(embedding_type='fasttext.wiki.simple', embedding_dims=300)
```

#### FastText

Several pre-trained [FastText](https://fasttext.cc/) embeddings are included. For now, we only have the word embeddings and not the n-gram features. All embedding have 300 dimensions.

-   [English Vectors](https://fasttext.cc/docs/en/english-vectors.html): e.g. `fasttext.wn.1M.300d`, [check out all avaiable embeddings](https://github.com/jfilter/text-classification-keras/blob/master/texcla/embeddings.py#L19)
-   [Multilang Vectors](https://fasttext.cc/docs/en/crawl-vectors.html): in the format `fasttext.cc.LANG_CODE` e.g. `fasttext.cc.en`
-   [Wikipedia Vectors](https://fasttext.cc/docs/en/pretrained-vectors.html): in the format `fasttext.wiki.LANG_CODE` e.g. `fasttext.wiki.en`

#### GloVe

The [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings are some kind of predecessor to FastText. In general choose FastText embeddings over GloVe. The dimension for the pre-trained embeddings varies.

-   : e.g. `glove.6B.50d`, [check out all avaiable embeddings](https://github.com/jfilter/text-classification-keras/blob/master/texcla/embeddings.py#L19)

### Tokenzation

-   To work on token (or word) level, use a TokenTokenizer such e.g. `TwokenizeTokenizer` or `SpacyTokenizer`.
-   To work on token and sentence level, use `SpacySentenceTokenizer`.
-   To create an custom Tokenizer, extend `Tokenizer` and implement the `token_generator` method.

#### Spacy

You may use [spaCy](https://spacy.io/) for the tokenization. See instructions on how to
[download model](https://spacy.io/docs/usage/models#download) for your target language. E.g. for English:

```bash
python -m spacy download en
```

### Models

#### Token-based Models

When working on token level, use `TokenModelFactory`.

```python
from texcla.models import TokenModelFactory, YoonKimCNN

factory = TokenModelFactory(tokenizer.num_classes, tokenizer.token_index,
    max_tokens=100, embedding_type='glove.6B.100d')
word_encoder_model = YoonKimCNN()
model = factory.build_model(token_encoder_model=word_encoder_model)
```

Currently supported models include:

-   [Yoon Kim CNN](https://arxiv.org/abs/1408.5882)
-   [Stacked RNNs](https://arxiv.org/abs/1312.6026)
-   [Attention (with/without context) based RNN encoders](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)

`TokenModelFactory.build_model` uses the provided word encoder which is then classified via a [Dense](https://keras.io/layers/core/#dense) layer.

#### Sentence-based Models

When working on sentence level, use `SentenceModelFactory`.

```python
# Pad max sentences per doc to 500 and max words per sentence to 200.
# Can also use `max_sents=None` to allow variable sized max_sents per mini-batch.

factory = SentenceModelFactory(10, tokenizer.token_index, max_sents=500,
    max_tokens=200, embedding_type='glove.6B.100d')
word_encoder_model = AttentionRNN()
sentence_encoder_model = AttentionRNN()

# Allows you to compose arbitrary word encoders followed by sentence encoder.
model = factory.build_model(word_encoder_model, sentence_encoder_model)
```

-   [Hierarchical attention networks](http://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)
    (HANs) can be build by composing two attention based RNN models. This is useful when a document is very large.
-   For smaller document a reasonable way to encode sentences is to average words within it. This can be done by using
    `token_encoder_model=AveragingEncoder()`
-   Mix and match encoders as you see fit for your problem.

`SentenceModelFactory.build_model` created a tiered model where words within a sentence is first encoded using
`word_encoder_model`. All such encodings per sentence is then encoded using `sentence_encoder_model`.

## Related

-   https://github.com/brightmart/text_classification
-   https://github.com/allenai/allennlp
-   https://github.com/facebookresearch/pytext
-   https://docs.fast.ai/text.html
-   https://github.com/dkpro/dkpro-tc

## Contributing

If you have a **question**, found a **bug** or want to propose a new **feature**, have a look at the [issues page](https://github.com/jfilter/text-classification-keras/issues).

**Pull requests** are especially welcomed when they fix bugs or improve the code quality.

## Acknowledgements

Built upon the work by Raghavendra Kotikalapudi: [keras-text](https://github.com/raghakot/keras-text).

## Citation

If you find Text Classification Keras useful for an academic publication, then please use the following BibTeX to cite it:

```tex
@misc{raghakotfiltertexclakeras
    title={Text Classification Keras},
    author={Raghavendra Kotikalapudi, and Johannes Filter, and contributors},
    year={2018},
    publisher={GitHub},
    howpublished={\url{https://github.com/jfilter/text-classification-keras}},
}
```

## License

MIT.
