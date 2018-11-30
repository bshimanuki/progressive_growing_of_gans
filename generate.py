import argparse
import os
import time

import imageio
import numpy as np
import skimage
import tensorflow as tf

import config
import tfutil
import dataset
import misc
import train


def generate(network_pkl, out_dir):
    if os.path.exists(out_dir):
        raise ValueError('{} already exists'.format(out_dir))
    misc.init_output_logging()
    np.random.seed(config.random_seed)
    tfutil.init_tf(config.tf_config)
    with tf.device('/gpu:0'):
        G, D, Gs = misc.load_pkl(network_pkl)
    training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **config.dataset)
    # grid_size, grid_reals, grid_labels, grid_latents = train.setup_snapshot_image_grid(G, training_set, **config.grid)
    number_of_images = 1000
    grid_labels = np.zeros([number_of_images, training_set.label_size], dtype=training_set.label_dtype)
    grid_latents = misc.random_latents(number_of_images, G)
    total_kimg = config.train.total_kimg
    sched = train.TrainingSchedule(total_kimg * 1000, training_set, **config.sched)
    grid_fakes = Gs.run(grid_latents, grid_labels, minibatch_size=sched.minibatch//config.num_gpus)
    os.makedirs(out_dir)
    for i, img in enumerate(grid_fakes):
        img = img.transpose((1,2,0))
        img = np.clip(img, -1, 1)
        img = skimage.img_as_ubyte(img)
        imageio.imwrite(os.path.join(out_dir, '{}.png'.format(i)), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('network_pkl')
    parser.add_argument('out_dir')
    args = parser.parse_args()
    generate(args.network_pkl, args.out_dir)
