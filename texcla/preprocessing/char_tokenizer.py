try:
    import spacy
except ImportError:
    pass

from .tokenizer import Tokenizer
from . import utils


class CharTokenizer(Tokenizer):

    def __init__(self,
                 lang='en',
                 lower=True,
                 charset=None):
        """Encodes text into `(samples, characters)`

        Args:
            lang: The spacy language to use. (Default value: 'en')
            lower: Lower cases the tokens if True. (Default value: True)
            charset: The character set to use. For example `charset = 'abc123'`. If None, all characters will be used.
                (Default value: None)
        """
        super(CharTokenizer, self).__init__(lang, lower)
        self.charset = charset

    def token_generator(self, texts, **kwargs):
        """Yields tokens from texts as `(text_idx, character)`
        """
        for text_idx, text in enumerate(texts):
            if self.lower:
                text = text.lower()
            for char in text:
                yield text_idx, char


class SentenceCharTokenizer(CharTokenizer):

    def __init__(self,
                 lang='en',
                 lower=True,
                 charset=None):
        """Encodes text into `(samples, sentences, characters)`

        Args:
            lang: The spacy language to use. (Default value: 'en')
            lower: Lower cases the tokens if True. (Default value: True)
            charset: The character set to use. For example `charset = 'abc123'`. If None, all characters will be used.
                (Default value: None)
        """
        super(SentenceCharTokenizer, self).__init__(lang, lower, charset)

    def token_generator(self, texts, **kwargs):
        """Yields tokens from texts as `(text_idx, sent_idx, character)`

        Args:
            texts: The list of texts.
            **kwargs: Supported args include:
                n_threads/num_threads: Number of threads to use. Uses num_cpus - 1 by default.
                batch_size: The number of texts to accumulate into a common working set before processing.
                    (Default value: 1000)
        """
        # Perf optimization. Only process what is necessary.
        n_threads, batch_size = utils._parse_spacy_kwargs(**kwargs)
        nlp = spacy.load(self.lang)

        kwargs = {
            'batch_size': batch_size,
            'n_threads': n_threads,
            'disable': ['ner']
        }

        # Perf optimization: Lower the entire text instead of individual tokens.
        texts_gen = utils._apply_generator(
            texts, lambda x: x.lower()) if self.lower else texts
        for text_idx, doc in enumerate(nlp.pipe(texts_gen, **kwargs)):
            for sent_idx, sent in enumerate(doc.sents):
                for word in sent:
                    for char in word:
                        yield text_idx, sent_idx, char
