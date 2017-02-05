from __future__ import print_function

import os
import sys

import tensorflow as tf
import librosa
import numpy as np
import scipy.io.wavfile
import stft

def read_audio_spectrum(filename):
    fs, x = scipy.io.wavfile.read(filename)
    S = stft.stft(x, N_FFT, N_FFT / 4, window_type='hamming', normalize_window=False).T
    S = np.log1p(np.abs(S))

    return S, fs


CONTENT_FILENAME = "inputs/russian_44100.wav"
STYLE_FILENAME = "inputs/english_woman_44100.wav"
N_FFT = 2048

a_content, fs = read_audio_spectrum(CONTENT_FILENAME)
a_style, fs = read_audio_spectrum(STYLE_FILENAME)

N_SAMPLES = min(a_content.shape[1], a_style.shape[1])
N_CHANNELS = a_content.shape[0]

a_content = a_content[:N_CHANNELS,:N_SAMPLES]
a_style = a_style[:N_CHANNELS,:N_SAMPLES]

N_FILTERS = 4096
KERNEL_SIZE = 10

a_content_tf = np.ascontiguousarray(a_content.T[None,None,:,:])
a_style_tf = np.ascontiguousarray(a_style.T[None,None,:,:])

# filter shape is "[filter_height, filter_width, in_channels, out_channels]"
std = np.sqrt(2) * np.sqrt(2.0 / ((N_CHANNELS + N_FILTERS) * KERNEL_SIZE))
kernel = np.random.randn(1, KERNEL_SIZE, N_CHANNELS, N_FILTERS)*std

g = tf.Graph()
with g.as_default(), g.device('/gpu:0'), tf.Session() as sess:
    # data shape is "[batch, in_height, in_width, in_channels]",
    x = tf.placeholder('float32', [1,1,N_SAMPLES,N_CHANNELS], name="x")

    kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
    conv = tf.nn.conv2d(
        x,
        kernel_tf,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")
    
    net = tf.nn.relu(conv)

    content_features = net.eval(feed_dict={x: a_content_tf})
    style_features = net.eval(feed_dict={x: a_style_tf})
    
    features = np.reshape(style_features, (-1, N_FILTERS))
    style_gram = np.matmul(features.T, features) / N_SAMPLES

ALPHA = 1e-4
learning_rate= 1e-3
ITERATIONS = 300

result = None
with tf.Graph().as_default():
    x = tf.Variable(np.random.randn(1,1,N_SAMPLES,N_CHANNELS).astype(np.float32) * 1e-3, name="x")

    kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
    conv = tf.nn.conv2d(
        x,
        kernel_tf,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")
    
    
    net = tf.nn.relu(conv)

    content_loss = ALPHA * 2 * tf.nn.l2_loss(
            net - content_features)

    style_loss = 0

    _, height, width, number = map(lambda i: i.value, net.get_shape())

    size = height * width * number
    feats = tf.reshape(net, (-1, number))
    gram = tf.matmul(tf.transpose(feats), feats)  / N_SAMPLES
    style_loss = 2 * tf.nn.l2_loss(gram - style_gram)

     # Overall loss
    loss = content_loss + style_loss

    opt = tf.contrib.opt.ScipyOptimizerInterface(
          loss, method='L-BFGS-B', options={'maxiter': ITERATIONS})

    step_i = [0]
    def callback(*args):
        step_i[0] += 1
        sys.stdout.write('Optimization step: %-3d\r' % step_i[0])
        sys.stdout.flush()

    # Optimization
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
       
        print('Started optimization.')

        opt.minimize(sess, loss_callback=callback)
    
        print('\nFinal loss:', loss.eval())
        result = x.eval()

a = np.zeros_like(a_content)
a[:N_CHANNELS,:] = np.exp(result[0,0].T) - 1

step_i = [0]
def callback(*args):
    step_i[0] += 1
    sys.stdout.write('Inversion step: %-3d\r' % step_i[0])
    sys.stdout.flush()

x = stft.ispectrogram_fast(a.T, N_FFT, N_FFT / 4, alpha=0.99, iters=400, callback=callback)

print()

OUTPUT_FILENAME = 'outputs/out.wav'
librosa.output.write_wav(OUTPUT_FILENAME, x, fs)

