import argparse
import os
import string
import textwrap

import cv2
from gensim.models import Word2Vec
import imageio
import numpy as np
import scipy.stats
from PIL import Image, ImageDraw, ImageFont
import skimage

FONT_PATH = '/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf'
# FONT_PATH = '/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf'
FONT = ImageFont.truetype(FONT_PATH, 13, encoding='unic')
word2vec = Word2Vec.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'brown.embedding'))


def convert(text, size=128, margin=4, save_path=None, color=False, word_per_line=False):
    char_width, char_height = FONT.getsize('\u2588') # full block

    linewidth = (size - margin) // char_width
    lineheight = (size - margin) // char_height
    # linewidth = 18

    if word_per_line:
        lines = text.split()
    else:
        lines = textwrap.wrap(text, width=linewidth)

    # img = Image.new('RGB', (char_width*linewidth + 2*margin, char_height*lineheight + 2*margin), 'white')
    if color:
        img = Image.new('RGB', (size, size), 'black')
    else:
        img = Image.new('RGB', (size, size), 'white')

    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        if i == lineheight:
            break
        if color:
            j = 0
            for word in line.split():
            # for j, c in enumerate(line):
                # hsv = np.uint8([[[(ord(c) % 32) << 3, 255, 255]]])
                lower = word.lower().strip('.').strip(',')
                if lower in word2vec:
                    vals = word2vec[lower][:2]
                    vals = scipy.stats.norm.cdf(vals)
                    vals = (256 * vals).astype(np.uint8)
                    h, s = vals
                    s = (3*s + 1*256) // 4
                else:
                    h = 0
                    s = 0
                v = 255
                hsv = np.uint8([[[h, s, v]]])
                fill = tuple(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0]) + (255,)
                draw.text((margin + j*char_width, margin + i*char_height), word, fill=fill, font=FONT)
                j += len(word) + 1
        else:
            draw.text((margin, margin + i*char_height), line, 'black', FONT)


    if save_path is not None:
        img.save(save_path)

    return img

def convert_all(texts, directory, **kwargs):
    os.makedirs(directory, exist_ok=True)
    for i, text in enumerate(texts):
        convert(text, save_path=os.path.join(directory, '{}.png'.format(i)), **kwargs)

def ocr(img, margin=4):
    char_width, char_height = FONT.getsize('\u2588') # full block
    linewidth = (img.shape[0] - margin) // char_width
    lineheight = (img.shape[1] - margin) // char_height

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    characters = ' ' + string.digits + string.ascii_letters + string.punctuation
    chrs = Image.new('L', (len(characters) * char_width, char_height), 'white')
    draw = ImageDraw.Draw(chrs)
    draw.text((0,0), characters, 'black', FONT)
    chrs = np.reshape(chrs, (char_height, -1, char_width))
    chrs = np.transpose(chrs, (1, 0, 2))
    img = img[margin:margin+lineheight*char_height,margin:margin+linewidth*char_width]
    img = np.reshape(img, (lineheight, char_height, linewidth, char_width))
    img = np.transpose(img, (0, 2, 1, 3))

    img = np.expand_dims(img, axis=2)

    img = skimage.img_as_float(img)
    # img -= 0.5
    img -= np.mean(img)
    chrs = skimage.img_as_float(chrs)
    # chrs -= 0.5
    chrs -= np.mean(chrs)

    activations = img * chrs
    activations = np.mean(activations, axis=(-1,-2))

    indices = np.argmax(activations, axis=-1)
    lines = [''.join(characters[c] for c in line).split() for line in indices]
    sentence = []
    for line in lines:
        sentence.extend(line)

    sentence = ' '.join(sentence)
    return sentence

if __name__ == '__main__':
    # convert('testing 1 2 3 hello how about you this is really fun let me see', save_path='img.png', color=True)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('infile', type=argparse.FileType('r'))
    # parser.add_argument('--dir', type=str, default='data')
    # args = parser.parse_args()

    # texts = args.infile.read().strip().split('\n')
    # convert_all(texts, args.dir)
    # # convert_all(texts, args.dir, word_per_line=True)
    # # convert_all(texts, args.dir, color=True)

    img = imageio.imread('img.png')
    print(ocr(img))
