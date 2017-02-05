import tensorflow as tf
import librosa
import os
import numpy as np
import scipy.io.wavfile
import stft

CONTENT_FILENAME = "inputs/mandarin.wav"
STYLE_FILENAME = "inputs/russian.wav"

# Reads wav file and produces spectrum
# Fourier phases are ignored
N_FFT = 2048
def read_audio_spectum(filename):
    fs, x = scipy.io.wavfile.read(filename)
    x = x[:,0]
    # x, fs = librosa.load(filename)
    S = librosa.stft(x, N_FFT)
    p = np.angle(S)
    S = np.log1p(np.abs(S[:,:430]))  

    print S.shape
    return S, fs

def read_audio_spectrum(filename):
    fs, x = scipy.io.wavfile.read(filename)
    #x = x[:,0]
    S = stft.stft(x, N_FFT, N_FFT / 4, window_type='hamming', normalize_window=False).T
    S = np.log1p(np.abs(S))

    print S.shape
    return S, fs

a_content, fs = read_audio_spectrum(CONTENT_FILENAME)
a_style, fs = read_audio_spectrum(STYLE_FILENAME)

N_SAMPLES = a_content.shape[1]
N_CHANNELS = a_content.shape[0]
a_style = a_style[:N_CHANNELS, :N_SAMPLES]

N_FILTERS = 4096

a_content_tf = np.ascontiguousarray(a_content.T[None,None,:,:])
a_style_tf = np.ascontiguousarray(a_style.T[None,None,:,:])

KERNEL_SIZE = 5

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

from sys import stderr

#ALPHA= 1e-2
ALPHA = 1e-3
learning_rate= 1e-3
iterations = 100

result = None
with tf.Graph().as_default():

    # Build graph with variable input
#     x = tf.Variable(np.zeros([1,1,N_SAMPLES,N_CHANNELS], dtype=np.float32), name="x")
    x = tf.Variable(np.random.randn(1,1,N_SAMPLES,N_CHANNELS).astype(np.float32)*1e-3, name="x")

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
          loss, method='L-BFGS-B', options={'maxiter': 300})

    step_i = [0]
    def callback(*args):
        step_i[0] += 1
        print step_i[0]

    # Optimization
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
       
        print('Started optimization.')

        opt.minimize(sess, loss_callback=callback)
    
        print 'Final loss:', loss.eval()
        result = x.eval()

a = np.zeros_like(a_content)
a[:N_CHANNELS,:] = np.exp(result[0,0].T) - 1

# This code is supposed to do phase reconstruction
#p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
#for i in range(100):
#    S = a * np.exp(1j*p)
#    x = librosa.istft(S)
#    p = np.angle(librosa.stft(x, N_FFT))
#    print i

step_i = [0]
def callback(*args):
    step_i[0] += 1
    print step_i[0]

x = stft.ispectrogram_fast(a.T, N_FFT, N_FFT / 4, alpha=0.99, iters=300, callback=callback)

OUTPUT_FILENAME = 'outputs/out.wav'
librosa.output.write_wav(OUTPUT_FILENAME, x, fs)

