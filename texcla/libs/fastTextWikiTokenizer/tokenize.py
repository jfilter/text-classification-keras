# -*- coding: utf-8 -*-
# taken from: https://gist.github.com/jfilter/28e019b5d6c62ebdc87721862ea0867c
# Taken from: https: // gist.github.com/bittlingmayer/7139a6a75ba0dbbc3a06325394ae3a13

# See https://github.com/facebookresearch/fastText/blob/master/get-wikimedia.sh
#
# From https://github.com/facebookresearch/fastText/issues/161:
# See also: https://github.com/facebookresearch/fastText/issues/401
#
# We now have a script called 'get-wikimedia.sh', that you can use to download and
# process a recent wikipedia dump of any language. This script applies the preprocessing
# we used to create the published word vectors.
#
# The parameters we used to build the word vectors are the default skip-gram settings,
# except with a dimensionality of 300 as indicated on the top of the list of word
# vectors (we now understand that this could be more visible).

# See also: known issues with the original script https://github.com/facebookresearch/fastText/issues/281, which unfortunately we must re-implement here.

'''
        sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" -e "s/'/ ' /g" -e "s/“/\"/g" -e "s/”/\"/g" \
            -e 's/"/ " /g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/, / , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
            -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' -e 's/-/ - /g' -e 's/=/ /g' -e 's/=/ /g' -e 's/*/ /g' -e 's/|/ /g' \
            -e 's/«/ /g' | tr 0-9 " "
'''
import subprocess

from . import subprocess_fix

SUBEXES = ["s/’/'/g", "s/′/'/g", "s/''/ /g", "s/'/ ' /g", 's/“/"/g', 's/”/"/g', 's/"/ /g', "s/\\./ \\. /g", "s/<br \\/>/ /g", "s/, / , /g",
           "s/(/ ( /g", "s/)/ ) /g", "s/\\!/ \\! /g", "s/\\?/ \\? /g", "s/\\;/ /g", "s/\\:/ /g", "s/-/ - /g", "s/=/ /g", "s/=/ /g", "s/*/ /g", "s/|/ /g", "s/«/ /g"]


def __normalize_text(s):
    commands = ['sed']
    for sb in SUBEXES:
        commands.append('-e')
        commands.append(sb)

    s = subprocess_fix.check_output_input(
        commands, input=s.encode()).decode("utf-8")
    return s

# Program to filter Wikipedia XML dumps to "clean" text consisting only of lowercase
# letters (a-z, converted from A-Z), and spaces (never consecutive)...
# All other characters are converted to spaces.  Only text which normally appears.
# in the web browser is displayed.  Tables are removed.  Image captions are.
# preserved.  Links are converted to normal text.  Digits are spelled out.
# *** Modified to not spell digits or throw away non-ASCII characters ***
# Written by Matt Mahoney, June 10, 2006.  This program is released to the public domain.


def __spaces(s):
    return ' '.join(s.split())


def __digits(s):
    return ''.join(filter(lambda c: not c.isdigit(), s))


def preproc(s):
    return __digits(__spaces(__normalize_text(s.lower())))


def tokenize(s):
    return(preproc(s).split())

# Example output:
#
# >>> preproc("Г. Шмидт, можно сказать «Давай давай!»?")
# 'г . шмидт , можно сказать давай давай ! » ?'
# >>> preproc('It won 1st place in the 3D film contest.')
# 'it won st place in the d film contest .'
