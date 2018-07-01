from keras_text.embeddings import get_embeddings_index

def test_build_index_glove():
    index = get_embeddings_index('glove.6B.50d')
    assert(index['a'] is not None)
    assert(index['and'] is not None)
