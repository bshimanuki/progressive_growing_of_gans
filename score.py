import argparse
import logging
import os
import string
import time

import nltk
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import pytesseract

from spell import SpellChecker

def method_coco(p_n, *args, **kwargs):
    small = 1e-9
    tiny = 1e-15
    return [(p_i.numerator + tiny) / (p_i.denominator + tiny) for p_i in p_n]

def get_bleu(sentences_ref, image_path, spellchecker=None):
    img = Image.open(image_path)
    if img.size[0] == 256:
        # grab only text if joint representation
        w, h = img.size
        img = img.crop((0, 0, w, h // 4))
    if np.mean(img) < np.max(img) / 2:
        # invert if black background
        img = ImageOps.invert(img)
        img = np.min(img, axis=-1)
        img = Image.fromarray(img, 'L')
        img = ImageEnhance.Brightness(img).enhance(0.8)
        img = ImageEnhance.Contrast(img).enhance(2)
        img.save('test.png')
    sentence_gen = pytesseract.image_to_string(img).lower()
    sentence_gen = nltk.word_tokenize(sentence_gen)
    if spellchecker is not None:
        sentence_gen = [spellchecker(w) for w in sentence_gen]
    smoothing_function = nltk.translate.bleu_score.SmoothingFunction().method7
    # smoothing_function = method_coco
    if len(sentence_gen) > 1:
        bleu = nltk.translate.bleu_score.sentence_bleu(sentences_ref, sentence_gen, smoothing_function=smoothing_function)
    else:
        bleu = 0.001337 # hack
    return bleu, sentence_gen

def get_average_bleu(sentences_ref, directory, spellchecker=None):
    bleu_sum = 0
    bleu_count = 0
    for path in sorted(os.listdir(directory)):
        path = os.path.join(directory, path)
        if path.endswith('.png'):
            bleu, s = get_bleu(sentences_ref, path, spellchecker=spellchecker)
            print(os.path.basename(path), bleu, ' '.join(s))
            bleu_sum += bleu
            bleu_count += 1
        # if bleu_count == 1000:
            # break
    return bleu_sum / bleu_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('reference')
    parser.add_argument('directory')
    args = parser.parse_args()

    start = time.time()
    reference = open(args.reference).read().lower().split('\n')[:10000]
    sentences_ref = [nltk.word_tokenize(s) for s in reference]

    # spellchecker = SpellChecker([w for s in sentences_ref for w in s], max_distance=2)
    spellchecker = None

    logging.info('Loaded in {}s.'.format(time.time() - start))

    bleu_score = get_average_bleu(sentences_ref, args.directory, spellchecker)
    print(bleu_score)

    # val = open('coco_captions_val.txt').read().lower().split('\n')[:1000]
    # sentences_val = [nltk.word_tokenize(s) for s in val]

    # smoothing_function = nltk.translate.bleu_score.SmoothingFunction().method7
    # # smoothing_function = method_coco
    # bleu_sum = 0
    # bleu_count = 0
    # for s in sentences_val:
        # bleu = nltk.translate.bleu_score.sentence_bleu(sentences_ref, s, smoothing_function=smoothing_function)
        # print(bleu, ' '.join(s))
        # bleu_sum += bleu
        # bleu_count += 1
    # print(bleu_sum / bleu_count)
