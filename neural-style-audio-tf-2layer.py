from __future__ import print_function

import os
import sys

import tensorflow as tf
import librosa
import numpy as np
import scipy.io.wavfile
import stft

def read_audio(filename):
    fs, x = scipy.io.wavfile.read(filename)
    return x, fs


CONTENT_FILENAME = "inputs/russian_44100.wav"
STYLE_FILENAME = "inputs/english_woman_44100.wav"
N_FFT = 2048
FFT_HOP_SIZE = N_FFT / 4
N_FILTERS = 8196
KERNEL_SIZE = 11

a_content, fs = read_audio(CONTENT_FILENAME)
a_style, _ = read_audio(STYLE_FILENAME)

N_SAMPLES = min(len(a_content), len(a_style))
a_content = a_content[:N_SAMPLES]
a_style = a_style[:N_SAMPLES]

# std_kernel = 2 * np.sqrt(3.0 / (N_FFT + N_FFT))
std_kernel = 1
fft_kernel = np.random.randn(1, N_FFT, 1, N_FFT) * std_kernel

a_content_tf = np.ascontiguousarray(a_content.T[None,None,:,None])
a_style_tf = np.ascontiguousarray(a_style.T[None,None,:,None])

N_CHANNELS = N_FFT

# filter shape is "[filter_height, filter_width, in_channels, out_channels]"
std = np.sqrt(2) * np.sqrt(2.0 / ((N_CHANNELS + N_FILTERS) * KERNEL_SIZE))
kernel = np.random.randn(1, KERNEL_SIZE, N_CHANNELS, N_FILTERS)*std

g = tf.Graph()
with g.as_default(), g.device('/gpu:0'), tf.Session() as sess:
    # data shape is "[batch, in_height, in_width, in_channels]",
    x = tf.placeholder('float32', [1, 1, N_SAMPLES, 1])

    fft_kernel_tf = tf.constant(fft_kernel, name="fft_kernel", dtype='float32')
    fft_conv = tf.nn.conv2d(
        x, 
        fft_kernel_tf,
        strides=[1, 1, FFT_HOP_SIZE, 1],
        padding="VALID",
        name="fft_conv"
        )

    # https://nucl.ai/blog/extreme-style-machines/
    fft = tf.nn.elu(fft_conv)

    fft_res = fft.eval(feed_dict={x: a_content_tf})

    kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
    conv = tf.nn.conv2d(
        fft,
        kernel_tf,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")
    
    net = tf.nn.elu(conv)

    content_features = net.eval(feed_dict={x: a_content_tf})
    style_features = net.eval(feed_dict={x: a_style_tf})
    
    features = np.reshape(style_features, (-1, N_FILTERS))
    style_gram = np.matmul(features.T, features) / N_SAMPLES

CONTENT_WEIGHT = 0
STYLE_WEIGHT = 1
ITERATIONS = 300

result = None
with tf.Graph().as_default():
    x = tf.Variable(np.random.randn(1,1,N_SAMPLES,1).astype(np.float32) * 1e-2, name="x")

    fft_kernel_tf = tf.constant(fft_kernel, name="fft_kernel", dtype='float32')
    fft_conv = tf.nn.conv2d(
        x, 
        fft_kernel_tf,
        strides=[1, 1, FFT_HOP_SIZE, 1],
        padding="VALID",
        name="fft_conv"
        )

    # https://nucl.ai/blog/extreme-style-machines/
    fft = tf.nn.elu(fft_conv)

    kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
    conv = tf.nn.conv2d(
        fft,
        kernel_tf,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")
    
    net = tf.nn.elu(conv)

    content_loss = CONTENT_WEIGHT * 2 * tf.nn.l2_loss(
            net - content_features)

    style_loss = 0

    _, height, width, number = map(lambda i: i.value, net.get_shape())

    size = height * width * number
    feats = tf.reshape(net, (-1, number))
    gram = tf.matmul(tf.transpose(feats), feats) / N_SAMPLES
    style_loss = 2 * STYLE_WEIGHT * tf.nn.l2_loss(gram - style_gram)

     # Overall loss
    loss = content_loss + style_loss

    opt = tf.contrib.opt.ScipyOptimizerInterface(
          loss, method='L-BFGS-B', options={'maxiter': ITERATIONS, 'disp': True, 'gtol': 1e-9})

    # Optimization
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
       
        print('Started optimization.')

        opt.minimize(sess)
    
        print('\nFinal loss:', loss.eval())
        result = x.eval()

        print('\nStyle loss:', style_loss.eval())
        print('\nContent loss:', content_loss.eval())

x = result[0,0,:,0]


OUTPUT_FILENAME = 'outputs/out.wav'
librosa.output.write_wav(OUTPUT_FILENAME, x, fs)