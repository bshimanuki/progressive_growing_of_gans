import argparse
import itertools
import os

import imageio
import nltk
import numpy as np

from make_image import decode_wordline, decode_wordblock


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='data')
    parser.add_argument('--vocab', type=argparse.FileType('r'), default='coco_captions.txt')
    parser.add_argument('outfile', type=argparse.FileType('w'))
    args = parser.parse_args()

    sentences = args.vocab.read().split('\n')[:10000]
    vocab = set()
    for sentence in sentences:
        toks = nltk.word_tokenize(sentence)
        vocab.update(toks)

    sentences = []
    for i in range(1000):
        path = os.path.join(args.dir, '{}.png'.format(i))
        img = imageio.imread(path)
        # sentence = decode_wordline(img)
        sentence = decode_wordblock(img, vocab=vocab)
        sentences.append(sentence)
    args.outfile.write('\n'.join(sentences))
