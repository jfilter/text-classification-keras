import spacy

from .tokenizer import Tokenizer

from . import utils


class WordTokenizer(Tokenizer):

    def __init__(self,
                 lang='en',
                 lower=True,
                 lemmatize=False,
                 remove_punct=True,
                 remove_digits=True,
                 remove_stop_words=False,
                 exclude_oov=False,
                 exclude_pos_tags=None,
                 exclude_entities=['PERSON']):
        """Encodes text into `(samples, words)`

        Args:
            lang: The spacy language to use. (Default value: 'en')
            lower: Lower cases the tokens if True. (Default value: True)
            lemmatize: Lemmatizes words when set to True. This also makes the word lower case
                irrespective if the `lower` setting. (Default value: False)
            remove_punct: Removes punct words if True. (Default value: True)
            remove_digits: Removes digit words if True. (Default value: True)
            remove_stop_words: Removes stop words if True. (Default value: False)
            exclude_oov: Exclude words that are out of spacy embedding's vocabulary.
                By default, GloVe 1 million, 300 dim are used. You can override spacy vocabulary with a custom
                embedding to change this. (Default value: False)
            exclude_pos_tags: A list of parts of speech tags to exclude. Can be any of spacy.parts_of_speech.IDS
                (Default value: None)
            exclude_entities: A list of entity types to be excluded.
                Supported entity types can be found here: https://spacy.io/docs/usage/entity-recognition#entity-types
                (Default value: ['PERSON'])
        """

        super(WordTokenizer, self).__init__(lang, lower)
        self.lemmatize = lemmatize
        self.remove_punct = remove_punct
        self.remove_digits = remove_digits
        self.remove_stop_words = remove_stop_words

        self.exclude_oov = exclude_oov
        self.exclude_pos_tags = set(exclude_pos_tags or [])
        self.exclude_entities = set(exclude_entities or [])

    def _apply_options(self, token):
        """Applies various filtering and processing options on token.

        Returns:
            The processed token. None if filtered.
        """
        # Apply work token filtering.
        if token.is_punct and self.remove_punct:
            return None
        if token.is_stop and self.remove_stop_words:
            return None
        if token.is_digit and self.remove_digits:
            return None
        if token.is_oov and self.exclude_oov:
            return None
        if token.pos_ in self.exclude_pos_tags:
            return None
        if token.ent_type_ in self.exclude_entities:
            return None

        # Lemmatized ones are already lowered.
        if self.lemmatize:
            return token.lemma_
        if self.lower:
            return token.lower_
        return token.orth_

    def token_generator(self, texts, **kwargs):
        """Yields tokens from texts as `(text_idx, word)`

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

        disabled = ['parser']
        if len(self.exclude_entities) > 0:
            disabled.append('ner')

        kwargs = {
            'batch_size': batch_size,
            'n_threads': n_threads,
            'disable': disabled
        }

        for text_idx, doc in enumerate(nlp.pipe(texts, **kwargs)):
            for word in doc:
                processed_word = self._apply_options(word)
                if processed_word is not None:
                    yield text_idx, processed_word


class SentenceWordTokenizer(WordTokenizer):

    def __init__(self,
                 lang='en',
                 lower=True,
                 lemmatize=False,
                 remove_punct=True,
                 remove_digits=True,
                 remove_stop_words=False,
                 exclude_oov=False,
                 exclude_pos_tags=None,
                 exclude_entities=['PERSON']):
        """Encodes text into `(samples, sentences, words)`

        Args:
            lang: The spacy language to use. (Default value: 'en')
            lower: Lower cases the tokens if True. (Default value: True)
            lemmatize: Lemmatizes words when set to True. This also makes the word lower case
                irrespective if the `lower` setting. (Default value: False)
            remove_punct: Removes punct words if True. (Default value: True)
            remove_digits: Removes digit words if True. (Default value: True)
            remove_stop_words: Removes stop words if True. (Default value: False)
            exclude_oov: Exclude words that are out of spacy embedding's vocabulary.
                By default, GloVe 1 million, 300 dim are used. You can override spacy vocabulary with a custom
                embedding to change this. (Default value: False)
            exclude_pos_tags: A list of parts of speech tags to exclude. Can be any of spacy.parts_of_speech.IDS
                (Default value: None)
            exclude_entities: A list of entity types to be excluded.
                Supported entity types can be found here: https://spacy.io/docs/usage/entity-recognition#entity-types
                (Default value: ['PERSON'])
        """
        super(SentenceWordTokenizer, self).__init__(lang,
                                                    lower,
                                                    lemmatize,
                                                    remove_punct,
                                                    remove_digits,
                                                    remove_stop_words,
                                                    exclude_oov,
                                                    exclude_pos_tags,
                                                    exclude_entities)

    def token_generator(self, texts, **kwargs):
        """Yields tokens from texts as `(text_idx, sent_idx, word)`

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

        disabled = []
        if len(self.exclude_entities) > 0:
            disabled.append('ner')

        kwargs = {
            'batch_size': batch_size,
            'n_threads': n_threads,
            'disable': disabled
        }

        for text_idx, doc in enumerate(nlp.pipe(texts, **kwargs)):
            for sent_idx, sent in enumerate(doc.sents):
                for word in sent:
                    processed_word = self._apply_options(word)
                    if processed_word is not None:
                        yield text_idx, sent_idx, processed_word
