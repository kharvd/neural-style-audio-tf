import numpy as np
import librosa

def __get_window(window_type, window_size, normalize_window):
    window = np.zeros(0)

    if window_type == 'hamming':
        window = np.hamming(window_size)
    elif window_type == 'hanning':
        window = np.hanning(window_size)
    elif window_type == 'rectangular':
        window = np.ones(window_size)
    elif window_type == 'truncated_hanning':
        window = np.hanning(window_size + 2)[1:-1]
    else:
        raise Exception('Unknown window: %s' % window)

    if normalize_window:
        window /= sum(window)

    return window


def __stft(signal, window_size, hop_size, window_type='hamming', normalize_window=True):
    window = __get_window(window_type, window_size, normalize_window)

    to_pad = hop_size - (len(signal) - window_size) % hop_size
    
    if to_pad < hop_size:
        signal = np.lib.pad(signal, (0, to_pad), 'constant', constant_values=0)
    
    return np.array([np.fft.rfft(signal[i:i+window_size] * window)
                     for i in range(0, len(signal) - window_size + 1, hop_size)])


def stft(signal, window_size, hop_size, window_type='hamming', normalize_window=True):
    to_pad = hop_size - (len(signal) - window_size) % hop_size
    if to_pad < hop_size:
        signal = np.lib.pad(signal, (0, to_pad), 'constant', constant_values=0)

    return librosa.stft(signal, n_fft=window_size, hop_length=hop_size, 
                        window=__get_window(window_type, window_size,
                            normalize_window), center=False).T


def __istft(data, window_size, hop_size, window_type='hamming', normalize_window=True):
    window = __get_window(window_type, window_size, normalize_window)

    res = np.zeros(window_size + hop_size * (data.shape[0] - 1))
    w_sum = np.zeros(len(res))

    for i in range(data.shape[0]):
        inverted = np.fft.irfft(data[i]).real
        res[i * hop_size : i * hop_size + window_size] += inverted * window
        w_sum[i * hop_size : i * hop_size + window_size] += window ** 2
        
    non_zero = w_sum != 0
    res[non_zero] /= w_sum[non_zero]
    
    return res


def istft(data, window_size, hop_size, window_type='hamming', normalize_window=True):
    return librosa.istft(data.T, win_length=window_size, hop_length=hop_size, 
                         window=__get_window(window_type, window_size,
                             normalize_window), center=False)


def spectrogram(signal, window_size, hop_size, 
        window_type='hamming', normalize_window=True):
    return np.abs(stft(signal, window_size, hop_size, window_type, normalize_window)) ** 2


# Griffin-Lim
def ispectrogram(data, window_size, hop_size, window_type='hamming',
        normalize_window=True, iters=10, callback=None):
    res = np.random.random(window_size + hop_size * (data.shape[0] - 1))

    init_magnitudes = np.sqrt(data)
    
    reg = np.max(data) / 1e8
    
    for i in range(iters):
        new_stft = stft(res, window_size, hop_size, window_type,
                normalize_window)
        
        normalized_stft = init_magnitudes * new_stft / np.abs(new_stft)
        res = istft(normalized_stft, window_size, hop_size, window_type,
                normalize_window)
        
        if callback != None:
            callback(res)
        
    return res


# Fast Griffin-Lim (https://lts2.epfl.ch/unlocbox/notes/unlocbox-note-007.pdf)
def __ispectrogram_fast(data, window_size, hop_size, alpha=0.99, window_type='hamming',
        normalize_window=True, iters=10, callback=None):
    magnitudes = np.sqrt(data)
    reg = np.max(data) / 1e8

    def project_c1(spec):
        inv = istft(spec, window_size, hop_size, window_type, normalize_window)
        return stft(inv, window_size, hop_size, window_type, normalize_window)
    
    def project_c2(spec):
        return magnitudes * spec / np.maximum(reg, np.abs(spec))
    
    spec_c = project_c2(np.random.random(data.shape) + 1j * np.random.random(data.shape))
    spec_t = project_c1(project_c2(spec_c))
    
    for i in range(iters):
        prev_spec_t = spec_t
        spec_t = project_c1(project_c2(spec_c))

        spec_c = spec_t + alpha * (spec_t - prev_spec_t)
        
        if callback != None:
            callback(spec_c)

    res = ispectrogram(project_c2(spec_c), window_size, hop_size, window_type, normalize_window)

    return res


# Fast Griffin-Lim (https://lts2.epfl.ch/unlocbox/notes/unlocbox-note-007.pdf)
def ispectrogram_fast(data, window_size, hop_size, alpha=0.99, window_type='hamming',
        normalize_window=True, iters=10, callback=None):
    magnitudes = data
    reg = np.max(data) / 1e8

    def project_c1(spec):
        inv = istft(spec, window_size, hop_size, window_type, normalize_window)
        return stft(inv, window_size, hop_size, window_type, normalize_window)
    
    def project_c2(spec):
        return magnitudes * spec / np.maximum(reg, np.abs(spec))
    
    res = np.random.random(window_size + hop_size * (data.shape[0] - 1))
    spec_c = stft(res, window_size, hop_size, window_type, normalize_window)
    diff_spec_c = 0.0
    
    for i in range(iters):
        prev_spec_c = spec_c
        spec_c = project_c1(project_c2(spec_c + alpha * diff_spec_c))
        diff_spec_c = spec_c - prev_spec_c
        
        if callback != None:
            callback(spec_c)

    res = istft(spec_c, window_size, hop_size, window_type, normalize_window)

    return res
