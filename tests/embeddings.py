from keras_text.embeddings import get_embeddings_index


def test_build_index_glove():
    index = get_embeddings_index('glove.6B.50d')
    assert(index['a'] is not None)
    assert(index['and'] is not None)


def test_build_index_fasttext_en():
    index = get_embeddings_index('fasttext.wn.1M.300d')
    assert(index['a'] is not None)
    assert(index['and'] is not None)


def test_build_index_fasttext_wiki():
    index = get_embeddings_index('fasttext.wiki.simple')
    assert(index['a'] is not None)
    assert(index['and'] is not None)


def test_build_index_fasttext_cc():
    index = get_embeddings_index('fasttext.cc.en')
    assert(index['a'] is not None)
    assert(index['and'] is not None)
