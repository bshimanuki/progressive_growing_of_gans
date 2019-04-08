# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import contextlib

import numpy as np
import tensorflow as tf

def debug(x, data):
    with tf.device('/cpu:0'):
        print_op = tf.print(data)
    with tf.control_dependencies([print_op]):
        x = tf.identity(x)
    return x

# NOTE: Do not import any application-specific modules here!

#----------------------------------------------------------------------------

def lerp(a, b, t): return a + (b - a) * t
def lerp_clip(a, b, t): return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
def cset(cur_lambda, new_cond, new_lambda): return lambda: tf.cond(new_cond, new_lambda, cur_lambda)

#----------------------------------------------------------------------------
# Get/create weight tensor for a convolutional or fully-connected layer.

def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in) # He init
    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense(x, fmaps, gain=np.sqrt(2), use_wscale=False):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Convolutional layer.

def conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Apply bias to the given activation tensor.

def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return x + tf.reshape(b, [1, -1, 1, 1])

#----------------------------------------------------------------------------
# Leaky ReLU activation. Same as tf.nn.leaky_relu, but supports FP16.

def leaky_relu(x, alpha=0.2):
    with tf.name_scope('LeakyRelu'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        return tf.maximum(x * alpha, x)

#----------------------------------------------------------------------------
# Nearest-neighbor upscaling layer.

def upscale2d(x, factor=2, y_factor=None):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    x_factor = factor
    y_factor = factor if y_factor is None else y_factor
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, y_factor, 1, x_factor])
        x = tf.reshape(x, [-1, s[1], s[2] * y_factor, s[3] * x_factor])
        return x

#----------------------------------------------------------------------------
# Fused upscale2d + conv2d.
# Faster and uses less memory than performing the operations separately.

def upscale2d_conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False, y_factor=2, x_factor=2):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, fmaps, x.shape[1].value], gain=gain, use_wscale=use_wscale, fan_in=(kernel**2)*x.shape[1].value)
    w = tf.pad(w, [[y_factor-1, y_factor-1], [x_factor-1, x_factor-1], [0,0], [0,0]], mode='CONSTANT')
    partials = []
    # w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    for _y in range(y_factor):
        for _x in range(x_factor):
            partials.append(w[_y:kernel+_y+y_factor-1, _x:kernel+_x+x_factor-1])
    w = tf.add_n(partials)
    w = tf.cast(w, x.dtype)
    os = [tf.shape(x)[0], fmaps, x.shape[2] * y_factor, x.shape[3] * x_factor]
    return tf.nn.conv2d_transpose(x, w, os, strides=[1,1,y_factor,x_factor], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Box filter downscaling layer.

def downscale2d(x, factor=2, y_factor=None):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    x_factor = factor
    y_factor = factor if y_factor is None else y_factor
    with tf.variable_scope('Downscale2D'):
        ksize = [1, 1, y_factor, x_factor]
        return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW') # NOTE: requires tf_config['graph_options.place_pruned_graph'] = True

#----------------------------------------------------------------------------
# Fused conv2d + downscale2d.
# Faster and uses less memory than performing the operations separately.

def conv2d_downscale2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False, y_factor=2, x_factor=2):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.pad(w, [[y_factor-1,y_factor-1], [x_factor-1,x_factor-1], [0,0], [0,0]], mode='CONSTANT')
    partials = []
    # w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
    for _y in range(y_factor):
        for _x in range(x_factor):
            partials.append(w[_y:kernel+_y+y_factor-1, _x:kernel+_x+x_factor-1])
    w = tf.add_n(partials) / (y_factor * x_factor)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,y_factor,x_factor], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Pixelwise feature vector normalization.

def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)

#----------------------------------------------------------------------------
# Minibatch standard deviation.

def minibatch_stddev_layer(x, group_size=4):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             # [NCHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])   # [GMCHW] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                              # [GMCHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MCHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [N1HW]  Replicate over group and pixels.
        return tf.concat([x, y], axis=1)                        # [NCHW]  Append as new fmap.

#----------------------------------------------------------------------------
# Generator network used in the paper.

def G_paper(
    latents_in,                         # First input: Latent vectors [minibatch, latent_size].
    labels_in,                          # Second input: Labels [minibatch, label_size].
    num_channels        = 1,            # Number of output color channels. Overridden based on dataset.
    resolution          = 32,           # Output resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    latent_size         = None,         # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).
    normalize_latents   = True,         # Normalize latent vectors before feeding them to the network?
    use_wscale          = True,         # Enable equalized learning rate?
    use_pixelnorm       = True,         # Enable pixelwise feature vector normalization?
    pixelnorm_epsilon   = 1e-8,         # Constant epsilon for pixelwise feature vector normalization.
    use_leakyrelu       = True,         # True = leaky ReLU, False = ReLU.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = True,         # True = use fused upscale2d + conv2d, False = separate upscale2d layers.
    structure           = None,         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically.
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    square   = True,        # True = square images, False = rectangular images
    resolution2         = 32,           # Output resolution height. Overridden based on dataset.
    **kwargs):                          # Ignore unrecognized keyword args.
    
    y_factor = 2 if square else 1
    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def PN(x): return pixel_norm(x, epsilon=pixelnorm_epsilon) if use_pixelnorm else x
    if latent_size is None: latent_size = nf(0)
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu if use_leakyrelu else tf.nn.relu
    
    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    combo_in = tf.cast(tf.concat([latents_in, labels_in], axis=1), dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

    # Building blocks.
    def block(x, res): # res = 2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res == 2: # 4x4
                if normalize_latents: x = pixel_norm(x, epsilon=pixelnorm_epsilon)
                with tf.variable_scope('Dense'):
                    if square:
                        x = dense(x, fmaps=nf(res-1)*16, gain=np.sqrt(2)/4, use_wscale=use_wscale) # override gain to match the original Theano implementation
                        x = tf.reshape(x, [-1, nf(res-1), 4, 4])
                    else:
                        x = dense(x, fmaps=nf(res-1)*resolution2*4, gain=np.sqrt(2)/4, use_wscale=use_wscale) # override gain to match the original Theano implementation
                        x = tf.reshape(x, [-1, nf(res-1), resolution2, 4])
                    x = PN(act(apply_bias(x)))
                with tf.variable_scope('Conv'):
                    x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
            else: # 8x8 and up
                if fused_scale:
                    with tf.variable_scope('Conv0_up'):
                        x = PN(act(apply_bias(upscale2d_conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale,y_factor=y_factor))))
                else:
                    x = upscale2d(x, y_factor=y_factor)
                    with tf.variable_scope('Conv0'):
                        x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
                with tf.variable_scope('Conv1'):
                    x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
            return x
    def torgb(x, res): # res = 2..resolution_log2
        lod = resolution_log2 - res
        with tf.variable_scope('ToRGB_lod%d' % lod):
            return apply_bias(conv2d(x, fmaps=num_channels, kernel=1, gain=1, use_wscale=use_wscale))

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        x = block(combo_in, 2)
        images_out = torgb(x, 2)
        for res in range(3, resolution_log2 + 1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = torgb(x, res)
            images_out = upscale2d(images_out, y_factor=y_factor)
            with tf.variable_scope('Grow_lod%d' % lod):
                images_out = lerp_clip(img, images_out, lod_in - lod)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(x, res, lod):
            y = block(x, res)
            img = lambda: upscale2d(torgb(y, res), 2**lod, y_factor=y_factor)
            if res > 2: img = cset(img, (lod_in > lod), lambda: upscale2d(lerp(torgb(y, res), upscale2d(torgb(x, res - 1), y_factor=y_factor), lod_in - lod), 2**lod, y_factor=y_factor))
            if lod > 0: img = cset(img, (lod_in < lod), lambda: grow(y, res + 1, lod - 1))
            return img()
        images_out = grow(combo_in, 2, resolution_log2 - 2)
        
    assert images_out.dtype == tf.as_dtype(dtype)
    images_out = tf.identity(images_out, name='images_out')
    return images_out

#----------------------------------------------------------------------------
# Discriminator network used in the paper.

def D_paper(
    images_in,                          # Input: Images [minibatch, channel, height, width].
    num_channels        = 1,            # Number of input color channels. Overridden based on dataset.
    resolution          = 32,           # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    use_wscale          = True,         # Enable equalized learning rate?
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = True,         # True = use fused conv2d + downscale2d, False = separate downscale2d layers.
    structure           = None,         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    square   = True,        # True = square images, False = rectangular images
    resolution2         = 32,           # Output resolution height. Overridden based on dataset.
    **kwargs):                          # Ignore unrecognized keyword args.
    
    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu

    y_factor = 2 if square else 1
    if square:
        resolution2 = resolution
    images_in.set_shape([None, num_channels, resolution2, resolution])
    images_in = tf.cast(images_in, dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

    # Building blocks.
    def fromrgb(x, res): # res = 2..resolution_log2
        with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res)):
            return act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=1, use_wscale=use_wscale)))
    def block(x, res): # res = 2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res >= 3: # 8x8 and up
                with tf.variable_scope('Conv0'):
                    x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                if fused_scale:
                    with tf.variable_scope('Conv1_down'):
                        x = act(apply_bias(conv2d_downscale2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale, y_factor=y_factor)))
                else:
                    with tf.variable_scope('Conv1'):
                        x = act(apply_bias(conv2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                    x = downscale2d(x, y_factor=y_factor)
            else: # 4x4
                if mbstd_group_size > 1:
                    x = minibatch_stddev_layer(x, mbstd_group_size)
                with tf.variable_scope('Conv'):
                    x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                with tf.variable_scope('Dense0'):
                    x = act(apply_bias(dense(x, fmaps=nf(res-2), use_wscale=use_wscale)))
                with tf.variable_scope('Dense1'):
                    x = apply_bias(dense(x, fmaps=1+label_size, gain=1, use_wscale=use_wscale))
            return x
    
    # Linear structure: simple but inefficient.
    if structure == 'linear':
        img = images_in
        x = fromrgb(img, resolution_log2)
        for res in range(resolution_log2, 2, -1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = downscale2d(img, y_factor=y_factor)
            y = fromrgb(img, res - 1)
            with tf.variable_scope('Grow_lod%d' % lod):
                x = lerp_clip(x, y, lod_in - lod)
        combo_out = block(x, 2)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(res, lod):
            x = lambda: fromrgb(downscale2d(images_in, 2**lod, y_factor=y_factor), res)
            if lod > 0: x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
            x = block(x(), res); y = lambda: x
            if res > 2: y = cset(y, (lod_in > lod), lambda: lerp(x, fromrgb(downscale2d(images_in, 2**(lod+1), y_factor=y_factor), res - 1), lod_in - lod))
            return y()
        combo_out = grow(2, resolution_log2 - 2)

    assert combo_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(combo_out[:, :1], name='scores_out')
    labels_out = tf.identity(combo_out[:, 1:], name='labels_out')
    return scores_out, labels_out

#----------------------------------------------------------------------------
# Modified generator network used in the paper.

def G_paired(
    latents_in,                         # First input: Latent vectors [minibatch, latent_size].
    labels_in,                          # Second input: Labels [minibatch, label_size].
    num_channels        = 1,            # Number of output color channels. Overridden based on dataset.
    resolution          = 32,           # Output resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    latent_size         = None,         # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).
    normalize_latents   = True,         # Normalize latent vectors before feeding them to the network?
    use_wscale          = True,         # Enable equalized learning rate?
    use_pixelnorm       = True,         # Enable pixelwise feature vector normalization?
    pixelnorm_epsilon   = 1e-8,         # Constant epsilon for pixelwise feature vector normalization.
    use_leakyrelu       = True,         # True = leaky ReLU, False = ReLU.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = True,         # True = use fused upscale2d + conv2d, False = separate upscale2d layers.
    structure           = None,         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically.
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    shape_cap = (16, 16, 1024), # CHW
    **kwargs):                          # Ignore unrecognized keyword args.
    
    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def PN(x): return pixel_norm(x, epsilon=pixelnorm_epsilon) if use_pixelnorm else x
    if latent_size is None: latent_size = nf(0)
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu if use_leakyrelu else tf.nn.relu
    
    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    combo_in = tf.cast(tf.concat([latents_in, labels_in], axis=1), dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

    # Building blocks.
    def block(x, res, cap=False): # res = 2..resolution_log2
        with tf.variable_scope('%dx%d-%s' % (2**res, 2**res, 'caption' if cap else 'image')):
            if res > 4 and cap:
                y_factor = 1
                x_factor = 4
            else:
                y_factor = 2
                x_factor = 2
            if res == 2: # 4x4
                if normalize_latents: x = pixel_norm(x, epsilon=pixelnorm_epsilon)
                with tf.variable_scope('Dense'):
                    x = dense(x, fmaps=nf(res-1)*16, gain=np.sqrt(2)/4, use_wscale=use_wscale) # override gain to match the original Theano implementation
                    x = tf.reshape(x, [-1, nf(res-1), 4, 4])
                    x = PN(act(apply_bias(x)))
                with tf.variable_scope('Conv'):
                    x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
            else: # 8x8 and up
                if fused_scale:
                    with tf.variable_scope('Conv0_up'):
                        x = PN(act(apply_bias(upscale2d_conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale,x_factor=x_factor,y_factor=y_factor))))
                else:
                    x = upscale2d(x, factor=x_factor,y_factor=y_factor)
                    with tf.variable_scope('Conv0'):
                        x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
                with tf.variable_scope('Conv1'):
                    x = PN(act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
            return x
    def torgb(x, res, cap=False): # res = 2..resolution_log2
        nc = shape_cap[0] if cap else num_channels
        lod = resolution_log2 - res
        with tf.variable_scope('ToRGB_lod%d-%s' % (lod, 'caption' if cap else 'image')):
            return apply_bias(conv2d(x, fmaps=nc, kernel=1, gain=1, use_wscale=use_wscale))

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        x_img = block(combo_in, 2)
        x_cap = block(combo_in, 2, cap=True)
        images_out_img = torgb(x_img, 2)
        images_out_cap = torgb(x_cap, 2, cap=True)
        for res in range(3, resolution_log2 + 1):
            lod = resolution_log2 - res
            x_img = block(x_img, res)
            img_img = torgb(x_img, res)
            images_out_img = upscale2d(images_out_img)
            with tf.variable_scope('Grow_lod%d' % lod):
                images_out_img = lerp_clip(img_img, images_out_img, lod_in - lod)
            # if res < 5:
                # x_cap = x_img
            # else:
            x_cap = block(x_cap, res, cap=True)
            img_cap = torgb(x_cap, res, cap=True)
            if res >= 5:
                images_out_cap = upscale2d(images_out_cap, factor=4, y_factor=1)
            else:
                images_out_cap = upscale2d(images_out_cap)
            with tf.variable_scope('Grow_lod%d' % lod):
                images_out_cap = lerp_clip(img_cap, images_out_cap, lod_in - lod)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(x, res, lod):
            cap_log_y_factor = max(4-res, 0)
            # if res == 5:
                # x_cap = x
            # elif res > 5:
            if res == 2:
                x_cap = x
            else:
                x, x_cap = x
            y = block(x, res)
            img = lambda: upscale2d(torgb(y, res), 2**lod)
            img_cap = lambda: upscale2d(torgb(y_cap, res, cap=True), 2**(2*lod-cap_log_y_factor), y_factor=2**cap_log_y_factor)
            img_pair = lambda: (img(), img_cap())
            if res > 2:
                img_cap = cset(img_cap, (lod_in > lod), lambda: upscale2d(lerp(torgb(y, res, cap=True), upscale2d(torgb(x, res - 1, cap=True)), lod_in - lod), 2**(2*lod-cap_log_y_factor), y_factor=2**cap_log_y_factor))
                img = cset(img, (lod_in > lod), lambda: upscale2d(lerp(torgb(y, res), upscale2d(torgb(x, res - 1)), lod_in - lod), 2**lod))
                img_pair = lambda: (img(), img_cap())
            # if res >= 5:
            if True:
                x_factor = 4 if res > 4 else 2
                y_factor = 1 if res > 4 else 2
                cap_log_y_factor = max(0, 4-res)
                y_cap = block(x_cap, res, cap=True)
                img_cap = lambda: upscale2d(torgb(y_cap, res, cap=True), 2**(2*lod-cap_log_y_factor), y_factor=2**cap_log_y_factor)
                if res > 2: img_cap = cset(img_cap, (lod_in > lod), lambda: upscale2d(lerp(torgb(y_cap, res, cap=True), upscale2d(torgb(x_cap, res - 1, cap=True), factor=x_factor, y_factor=y_factor), lod_in - lod), 2**(2*lod-cap_log_y_factor), y_factor=2**cap_log_y_factor))
                img_pair = lambda: (img(), img_cap())
                if lod > 0: img_pair = cset(img_pair, (lod_in < lod), lambda: grow((y, y_cap), res + 1, lod - 1))
            else:
                if lod > 0: img_pair = cset(img_pair, (lod_in < lod), lambda: grow(y, res + 1, lod - 1))
            return img_pair()
        images_out_img, images_out_cap = grow(combo_in, 2, resolution_log2 - 2)
        
    assert images_out_img.dtype == tf.as_dtype(dtype)
    assert images_out_cap.dtype == tf.as_dtype(dtype)
    images_out_img = tf.identity(images_out_img, name='images_out_img')
    images_out_cap = tf.identity(images_out_cap, name='images_out_cap')
    return images_out_img, images_out_cap

#----------------------------------------------------------------------------
# Modified discriminator network used in the paper.

def D_paired(
    images_in_img,                      # Input: Images [minibatch, channel, height, width].
    images_in_cap,                      # Input: Images [minibatch, channel, height, width].
    num_channels        = 1,            # Number of input color channels. Overridden based on dataset.
    resolution          = 32,           # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    use_wscale          = True,         # Enable equalized learning rate?
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = True,         # True = use fused conv2d + downscale2d, False = separate downscale2d layers.
    structure           = None,         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    resolution_cap = (16, 16, 1024), # CHW
    use_caption_features=True,
    use_image_features=True,
    **kwargs):                          # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    assert resolution_cap[1] * resolution_cap[2] == resolution ** 2
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu

    images_in_img.set_shape([None, num_channels, resolution, resolution])
    images_in_img = tf.cast(images_in_img, dtype)
    images_in_cap.set_shape([None, *resolution_cap])
    images_in_cap = tf.cast(images_in_cap, dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

    # Building blocks.
    def fromrgb(x, res, cap=False): # res = 2..resolution_log2
        with tf.variable_scope('FromRGB_lod%d-%s' % (resolution_log2 - res, 'caption' if cap else 'image')):
            ret = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=1, use_wscale=use_wscale)))
            return ret
    def block(x, res): # res = 2..resolution_log2
        if res >= 5:
            x, x_cap = x
        else:
            x_cap = None

        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res >= 3: # 8x8 and up
                with tf.variable_scope('Conv0'):
                    with tf.variable_scope('image', reuse=tf.AUTO_REUSE):
                        x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                    if x_cap is not None:
                        with tf.variable_scope('caption', reuse=tf.AUTO_REUSE):
                            x_cap = act(apply_bias(conv2d(x_cap, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                if fused_scale:
                    with tf.variable_scope('Conv1_down'):
                        with tf.variable_scope('image', reuse=tf.AUTO_REUSE):
                            x = act(apply_bias(conv2d_downscale2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                        if x_cap is not None:
                            with tf.variable_scope('caption', reuse=tf.AUTO_REUSE):
                                x_cap = act(apply_bias(conv2d_downscale2d(x_cap, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale, y_factor=1, x_factor=4)))
                else:
                    with tf.variable_scope('Conv1'):
                        with tf.variable_scope('image', reuse=tf.AUTO_REUSE):
                            x = act(apply_bias(conv2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                        if x_cap is not None:
                            with tf.variable_scope('caption', reuse=tf.AUTO_REUSE):
                                x_cap = act(apply_bias(conv2d(x_cap, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                    with tf.variable_scope('image', reuse=tf.AUTO_REUSE):
                        x = downscale2d(x)
                    if x_cap is not None:
                        with tf.variable_scope('caption', reuse=tf.AUTO_REUSE):
                            x_cap = downscale2d(x_cap, factor=4, y_factor=1)
            else: # 4x4
                if mbstd_group_size > 1:
                    x = minibatch_stddev_layer(x, mbstd_group_size)
                with tf.variable_scope('Conv'):
                    x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                with tf.variable_scope('Dense0'):
                    x = act(apply_bias(dense(x, fmaps=nf(res-2), use_wscale=use_wscale)))
                with tf.variable_scope('Dense1'):
                    x = apply_bias(dense(x, fmaps=1+label_size, gain=1, use_wscale=use_wscale))
            if res == 5:
                if use_caption_features:
                    if use_image_features:
                        return x + x_cap
                    else:
                        return x_cap
                else:
                    return x
            elif x_cap is not None:
                return x, x_cap
            else:
                return x
    
    # Linear structure: simple but inefficient.
    if structure == 'linear':
        img = images_in_img
        cap = images_in_cap
        x = fromrgb(img, resolution_log2)
        x_cap = fromrgb(cap, resolution_log2, cap=True)
        x = (x, x_cap)
        for res in range(resolution_log2, 2, -1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = downscale2d(img)
            if res > 4:
                cap = downscale2d(cap, factor=4, y_factor=1)
            else:
                cap = downscale2d(cap, factor=2, y_factor=2)
            y_img = fromrgb(img, res - 1)
            y_cap = fromrgb(cap, res - 1, cap=True)
            with tf.variable_scope('Grow_lod%d' % lod):
                if isinstance(x, tuple):
                    x, x_cap = x
                    x = lerp_clip(x, y_img, lod_in - lod)
                    x_cap = lerp_clip(x_cap, y_cap, lod_in - lod)
                    x = (x, x_cap)
                else:
                    if use_caption_features:
                        if use_image_features:
                            y = y_img + y_cap
                        else:
                            y = y_cap
                    else:
                        y = y_img
                    x = lerp_clip(x, y, lod_in - lod)
        combo_out = block(x, 2)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(res, lod):
            cap_log_y_factor = max(4-res, 0)
            if res >= 5:
                x = lambda: (fromrgb(downscale2d(images_in_img, 2**lod), res), fromrgb(downscale2d(images_in_cap, 2**(2*lod-cap_log_y_factor), y_factor=2**cap_log_y_factor), res, cap=True))
            else:
                if use_caption_features:
                    if use_image_features:
                        x = lambda: fromrgb(downscale2d(images_in_img, 2**lod), res) + fromrgb(downscale2d(images_in_cap, 2**(2*lod-cap_log_y_factor), y_factor=2**cap_log_y_factor), res, cap=True)
                    else:
                        x = lambda: fromrgb(downscale2d(images_in_cap, 2**(2*lod-cap_log_y_factor), y_factor=2**cap_log_y_factor), res, cap=True)
                else:
                    x = lambda: fromrgb(downscale2d(images_in_img, 2**lod), res)
            if lod > 0: x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
            x = block(x(), res); y = lambda: x
            if res > 2:
                _cap_log_y_factor = max(4-(res-1), 0)
                if isinstance(x, tuple):
                    def sub(x):
                        x, x_cap = x
                        y = lerp(x, fromrgb(downscale2d(images_in_img, 2**(lod+1)), res - 1), lod_in - lod)
                        y_cap = lerp(x_cap, fromrgb(downscale2d(images_in_cap, 2**(2*(lod+1)-_cap_log_y_factor), y_factor=2**_cap_log_y_factor), res - 1, cap=True), lod_in - lod)
                        return y, y_cap
                    y = cset(y, (lod_in > lod), lambda x=x: sub(x))
                else:
                    if use_caption_features:
                        if use_image_features:
                            y = cset(y, (lod_in > lod), lambda: lerp(x, fromrgb(downscale2d(images_in_img, 2**(lod+1)), res - 1) + fromrgb(downscale2d(images_in_cap, 2**(2*(lod+1)-_cap_log_y_factor), y_factor=2**_cap_log_y_factor), res - 1, cap=True), lod_in - lod))
                        else:
                            y = cset(y, (lod_in > lod), lambda: lerp(x, fromrgb(downscale2d(images_in_cap, 2**(2*(lod+1)-_cap_log_y_factor), y_factor=2**_cap_log_y_factor), res - 1, cap=True), lod_in - lod))
                    else:
                        y = cset(y, (lod_in > lod), lambda: lerp(x, fromrgb(downscale2d(images_in_img, 2**(lod+1)), res - 1), lod_in - lod))
            return y()
        combo_out = grow(2, resolution_log2 - 2)

    assert combo_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(combo_out[:, :1], name='scores_out')
    labels_out = tf.identity(combo_out[:, 1:], name='labels_out')
    return scores_out, labels_out

#----------------------------------------------------------------------------
