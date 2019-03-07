import argparse
import glob
import os
import random
import string
import textwrap

import cv2
from gensim.models import Word2Vec
import imageio
import numpy as np
import scipy.stats
import scipy.misc
from PIL import Image, ImageDraw, ImageFont
import skimage

import coco_captions

FONT_PATH = '/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf'
# FONT_PATH = '/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf'
FONT = ImageFont.truetype(FONT_PATH, 13, encoding='unic')
word2vec = Word2Vec.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'brown.embedding'))
word2vec15 = Word2Vec.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'brown15.embedding'))


def convert_joint(text, img_path, size=256, margin=4, save_path=None):
    char_width, char_height = FONT.getsize('\u2588') # full block

    linewidth = (size - margin) // char_width
    lineheight = (size - margin) // char_height // 4

    lines = textwrap.wrap(text, width=linewidth)

    img = Image.new('RGB', (size, size), 'white')

    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        if i == lineheight:
            break
        draw.text((margin, margin + i*char_height), line, 'black', FONT)

    img_coco = Image.open(img_path)
    size_coco = 3 * size // 4
    img_coco = img_coco.resize((size_coco, size_coco), Image.BILINEAR)

    img.paste(img_coco, ((size-size_coco)//2, size-size_coco))

    if save_path is not None:
        img.save(save_path)

    return img

def convert_joint_all_flowers(all_captions=False):
    in_dir = 'datasets/flowers/jpg'
    out_dir = 'datasets/flowers_joint_png'
    classes = open('datasets/flowers/trainclasses.txt').read().split('\n')
    pairs = []
    for c in classes:
        directory = 'datasets/flowers/text_c10/{}'.format(c.strip())
        files = glob.glob(os.path.join(directory, '*.txt'))
        for f in files:
            captions = open(f).read().strip().split('\n')
            if not all_captions:
                captions = captions[:1]
            for caption in set(captions):
                base = os.path.splitext(os.path.basename(f))[0]
                img = '{}.jpg'.format(base)
                pairs.append((caption, img))
    random.seed(1)
    random.shuffle(pairs)
    convert_joint_all(None, in_dir, out_dir, dataset=pairs)

def convert_joint_shared_flowers(lod=7):
    in_dir = 'datasets/flowers/jpg'
    out_dir = 'datasets/flowers_joint_shared_npz'
    classes = open('datasets/flowers/trainclasses.txt').read().split('\n')
    pairs = []
    for c in classes:
        directory = 'datasets/flowers/text_c10/{}'.format(c.strip())
        files = glob.glob(os.path.join(directory, '*.txt'))
        for f in files:
            captions = open(f).read().strip().split('\n')
            for caption in set(captions):
                base = os.path.splitext(os.path.basename(f))[0]
                img = '{}.jpg'.format(base)
                pairs.append((caption, img))
    random.seed(1)
    random.shuffle(pairs)
    os.makedirs(out_dir, exist_ok=True)
    img_size = 2 ** lod
    cap_size = 16 * 4 ** (lod - 4)
    for i, (caption, img) in enumerate(pairs):
        in_path = os.path.join(in_dir, img)
        out_path = os.path.join(out_dir, '{}.npz'.format(i))
        imgimg = scipy.misc.imresize(imageio.imread(in_path), (img_size, img_size))
        capimg = convert(caption, color='full_mask', size=cap_size, square=False)
        np.savez(out_path, img=imgimg, cap=capimg)

def convert_joint_all(json, in_dir, out_dir, dataset=None, **kwargs):
    os.makedirs(out_dir, exist_ok=True)
    if dataset is None:
        dataset = coco_captions.CocoCaptionDataset(json, file_name=True)
    for i, (text, img) in enumerate(dataset):
        in_path = os.path.join(in_dir, img)
        out_path = os.path.join(out_dir, '{}.png'.format(i))
        convert_joint(text, in_path, save_path=out_path, **kwargs)

def convert(text, size=128, margin=4, save_path=None, color=False, word_per_line=False, square=True):
    char_width, char_height = FONT.getsize('\u2588') # full block

    linewidth = (size - margin) // char_width
    xsize = size
    if square:
        lineheight = (size - margin) // char_height
        ysize = size
    else:
        lineheight = 1
        # ysize = char_height + 2 * margin
        ysize = 16
        margin = 1
    # linewidth = 18

    if word_per_line:
        lines = text.split()
    else:
        lines = textwrap.wrap(text, width=linewidth)

    # img = Image.new('RGB', (char_width*linewidth + 2*margin, char_height*lineheight + 2*margin), 'white')
    if color == 'full' or color == 'full_mask':
        img = Image.new('RGB', (xsize, ysize), 'black')
        img2 = np.zeros((ysize, xsize, 16), dtype=np.uint8)
    elif color:
        img = Image.new('RGB', (xsize, ysize), 'black')
    else:
        img = Image.new('RGB', (xsize, ysize), 'white')

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
                    if color == 'full' or color == 'full_mask':
                        vals = word2vec15[lower] * 128 / 3 + 128
                        vals = np.clip(vals, 0, 256)
                        vals = np.round(vals).astype(np.uint8)
                        h = 0
                        s = 0
                    else:
                        vals = word2vec[lower][:2]
                        vals = scipy.stats.norm.cdf(vals)
                        vals = (256 * vals).astype(np.uint8)
                        h, s = vals
                        s = (3*s + 1*256) // 4
                else:
                    if color == 'full' or color == 'full_mask':
                        vals = 128
                    h = 0
                    s = 0
                v = 255
                hsv = np.uint8([[[h, s, v]]])
                fill = tuple(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0]) + (255,)
                draw.text((margin + j*char_width, margin + i*char_height), word, fill=fill, font=FONT)
                if color == 'full' or color == 'full_mask':
                    img2[margin+i*char_height:margin+(i+1)*char_height, margin+j*char_width:margin+(j+len(word))*char_width, 1:] = vals
                j += len(word) + 1
        else:
            draw.text((margin, margin + i*char_height), line, 'black', FONT)
    if color == 'full' or color == 'full_mask':
        arr = np.array(img.getdata())
        img2[..., 0] = arr.reshape(img.size[1], img.size[0], 3)[..., 0]
        if color == 'full_mask':
            img2 *= img2[..., :1] > 0
        img = img2

    if save_path is not None:
        if color == 'full' or color == 'full_mask':
            np.save(save_path, img)
            imageio.imwrite('{}.png'.format(save_path.replace('npy', 'png')), img[...,:3])
        else:
            img.save(save_path, format='png')

    return img

def convert_all(texts, directory, **kwargs):
    os.makedirs(directory, exist_ok=True)
    os.makedirs(directory.replace('npy', 'png'), exist_ok=True)
    for i, text in enumerate(texts):
        convert(text, save_path=os.path.join(directory, str(i)), **kwargs)

def convert_wordline(text, size=64, save_path=None, block=False):
    random.seed(text)
    words = text.split()[:size]
    if block:
        lines = sorted(random.sample(range(size // 5 * 5), len(words)))
    else:
        lines = sorted(random.sample(range(size), len(words)))

    img = np.zeros((size, size, 3), dtype=np.uint8)

    for i, word in zip(lines, words):
        lower = word.lower().strip('.').strip(',')
        if lower in word2vec:
            vals = word2vec[lower] * 128 / 3 + 128
            vals = np.clip(vals, 0, 256)
            vals = np.round(vals).astype(np.uint8)
            if block:
                line = i // 5 * 5
                col = 12 * (i % 5)
                img[line:line+5, col+1:col+11, :2] = vals.reshape((5, 10, 2))
                img[line:line+5, col, 0] = 255
            else:
                img[i, 1:51, :2] = vals.reshape(50, 2)
                img[i, 0, 0] = 255
        else:
            if block:
                line = i // 5 * 5
                col = 12 * (i % 5)
                img[line:line+5, col, 1] = 255
            else:
                img[i, 0, 1] = 255

    if save_path is not None:
        imageio.imwrite(save_path, img)

    return img

def decode_wordblock(img, vocab=None):
    words = []
    for i in range(0, img.shape[0], 5):
        for j in range(0, img.shape[1], 12):
            if img[i, j, 1] > 128:
                words.append('<UNK>')
            elif img[i, j, 0] > 128:
                vals = img[i:i+5, j+1:j+11, :2].reshape(-1).astype(np.float32)
                vals = (vals - 128) / 128 * 3
                if vocab is None:
                    [(word, _)] = word2vec.similar_by_vector(vals, topn=1)
                    # print(word2vec.similar_by_vector(vals, topn=5))
                else:
                    _word = [w for (w, _) in word2vec.similar_by_vector(vals, topn=100) if w in vocab]
                    if _word:
                        word = _word[0]
                    else:
                        word = '<UNREC>'
                words.append(word)
    return ' '.join(words)

def decode_wordline(img):
    words = []
    for line in img:
        if line[0, 1] > 128:
            words.append('<UNK>')
        elif line[0, 0] > 128:
            vals = line[1:51, :2].reshape(-1).astype(np.float32)
            vals = (vals - 128) / 128 * 3
            [(word, _)] = word2vec.similar_by_vector(vals, topn=1)
            # print(word2vec.similar_by_vector(vals, topn=5))
            words.append(word)
    return ' '.join(words)

def convert_all_wordline(texts, directory, **kwargs):
    os.makedirs(directory, exist_ok=True)
    for i, text in enumerate(texts):
        # convert(text, save_path=os.path.join(directory, '{}.png'.format(i)), **kwargs)
        convert_wordline(text, save_path=os.path.join(directory, '{}.png'.format(i)), **kwargs)

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
    # convert('testing 1 2 3 hello how about you this is really fun let me see', save_path='img.npy', color='full')
    # convert('testing 1 2 3 hello how about you this is really fun let me see', save_path='img.npy', color='full_mask')
    # convert_wordline('a quick brown fox jumps over the lazy dog', save_path='img.png')
    # print(decode_wordline(imageio.imread('img.png')))
    # convert_wordline('a quick brown fox jumps over the lazy dog', save_path='img.png', block=True)
    # print(decode_wordblock(imageio.imread('img.png')))
    # exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=argparse.FileType('r'))
    parser.add_argument('--dir', type=str, default='data')
    args = parser.parse_args()

    texts = args.infile.read().strip().split('\n')
    # convert_all(texts, args.dir)
    # convert_all(texts, args.dir, word_per_line=True)
    # convert_all(texts, args.dir, color=True)
    # convert_all(texts, args.dir, color='full')
    # convert_all(texts, args.dir, color='full_mask', size=128*8, square=False)
    # convert_all_wordline(texts, args.dir)
    # convert_all_wordline(texts, args.dir, block=True)

    # img = imageio.imread('img.png')
    # print(ocr(img))
    # convert_joint_all('/data/vision/torralba/datasets/COCO/annotations/captions_train2014.json', in_dir='/data/vision/torralba/datasets/COCO/train2014', out_dir='datasets/joint_png')
    # convert_joint_all_flowers()
    convert_joint_shared_flowers()
