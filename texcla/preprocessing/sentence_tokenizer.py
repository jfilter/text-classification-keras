try:
    import spacy
except ImportError:
    pass

from . import utils
from .word_tokenizer import SpacyTokenizer


class SpacySentenceTokenizer(SpacyTokenizer):
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
        super(SpacySentenceTokenizer, self).__init__(lang,
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
