import argparse
import os
import string

import nltk
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import pytesseract

def get_bleu(sentences_ref, image_path):
    img = Image.open(image_path)
    if np.mean(img) < np.max(img) / 2:
        # invert if black background
        img = ImageOps.invert(img)
        img = np.min(img, axis=-1)
        img = Image.fromarray(img, 'L')
        img = ImageEnhance.Brightness(img).enhance(0.9)
        img = ImageEnhance.Contrast(img).enhance(2)
        img.save('test.png')
    sentence_gen = pytesseract.image_to_string(img).lower()
    sentence_gen = nltk.word_tokenize(sentence_gen)
    smoothing_function = nltk.translate.bleu_score.SmoothingFunction().method7
    if len(sentence_gen) > 1:
        bleu = nltk.translate.bleu_score.sentence_bleu(sentences_ref, sentence_gen, smoothing_function=smoothing_function)
    else:
        bleu = 0.001337 # hack
    return bleu, sentence_gen

def get_average_bleu(sentences_ref, directory):
    bleu_sum = 0
    bleu_count = 0
    for path in os.listdir(directory):
        path = os.path.join(directory, path)
        if path.endswith('.png'):
            bleu, s = get_bleu(sentences_ref, path)
            print(os.path.basename(path), bleu, ' '.join(s))
            bleu_sum += bleu
            bleu_count += 1
    return bleu_sum / bleu_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('reference')
    parser.add_argument('directory')
    args = parser.parse_args()

    reference = open(args.reference).read().lower().split('\n')[:10000]
    sentences_ref = [nltk.word_tokenize(s) for s in reference]

    bleu_score = get_average_bleu(sentences_ref, args.directory)
    print(bleu_score)

    # val = open('coco_captions_val.txt').read().lower().split('\n')[:1000]
    # sentences_val = [nltk.word_tokenize(s) for s in val]

    # smoothing_function = nltk.translate.bleu_score.SmoothingFunction().method4
    # bleu_sum = 0
    # bleu_count = 0
    # for s in sentences_val:
        # bleu = nltk.translate.bleu_score.sentence_bleu(sentences_ref, s, smoothing_function=smoothing_function)
        # print(bleu, ' '.join(s))
        # bleu_sum += bleu
        # bleu_count += 1
    # print(bleu_sum / bleu_count)
