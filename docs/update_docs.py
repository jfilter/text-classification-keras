#!/usr/bin/env python

import shutil
import subprocess
import sys

from md_autogen import MarkdownAPIGenerator
from md_autogen import to_md_file

from keras_text.models import token_model, sentence_model, sequence_encoders, layers

from keras_text import data, corpus, embeddings, experiment

from keras_text.utils import format, generators, io, sampling

from keras_text.preprocessing import char_tokenizer, sentence_tokenizer, tokenizer, utils, word_tokenizer


def generate_api_docs():
    modules = [
        token_model,
        sentence_model,
        sequence_encoders,
        layers,
        data, corpus, embeddings, experiment, format, generators, io, sampling,
        char_tokenizer, sentence_tokenizer, tokenizer, utils, word_tokenizer
    ]

    md_gen = MarkdownAPIGenerator(
        "keras_text", "https://github.com/jfilter/text-classification-keras")
    for m in modules:
        md_string = md_gen.module2md(m)
        to_md_file(md_string, m.__name__, "sources")


def update_index_md():
    shutil.copyfile('../README.md', 'sources/index.md')


def copy_templates():
    shutil.rmtree('sources', ignore_errors=True)
    shutil.copytree('templates', 'sources')


if __name__ == "__main__":
    copy_templates()
    update_index_md()
    generate_api_docs()
    if len(sys.argv) > 1 and sys.argv[1] == '--live':
        subprocess.run("mkdocs gh-deploy", shell=True, check=True)
    else:
        subprocess.run(
            "mkdocs build && cd site && python3 -m http.server", shell=True, check=True)
