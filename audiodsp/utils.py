import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import welch


def generate_sine_sweep(
    start_freq=20, stop_freq=1000, duration=0.1, sample_rate=44100, noise_std=0
):
    """
    generate sine sweep
    :param start_freq: lowest frequency
    :param stop_freq: highest frequency
    :param duration: duration of generated signal
    :param sample_rate: sample rate
    :param noise_std: standard deviation of white noise
    :return: sine sweep signal with amplitude of 1
    """
    num_samples = int(duration * sample_rate)

    freqs = np.linspace(2 * np.pi * start_freq, 2 * np.pi * stop_freq, num_samples)

    t = np.linspace(0, duration, num_samples)

    y = np.sin(freqs * t) + np.random.normal(0, noise_std, num_samples)

    return y


def generate_ascending_sine(
    start_amp=0, stop_amp=1, duration=0.1, freq=1e3, sample_rate=44100
):
    """
    generate ascending sine
    :param start_amp: amplitude at the start
    :param stop_amp: amplitude at the end
    :param duration: duration of generated signal
    :param freq: frequency of generated signal
    :param sample_rate: sample rate
    :return:
    """
    num_samples = int(duration * sample_rate)

    amps = np.linspace(start_amp, stop_amp, num_samples)

    t = np.linspace(0, duration, num_samples)

    return amps * np.sin(2 * np.pi * freq * t)


def to_log_freq(f, pxx):
    """
    interpolate linear frequency response to log
    :param f: array of frequency bin
    :param pxx: fft result corresponds to f
    :return: interpolated power
    """
    data_point = np.linspace(0, 1, f.shape[0])
    freq_log = (np.power(81.0, data_point) - 1) * 0.0125 * f.max()
    return freq_log, interp1d(f, pxx)(freq_log)


def estimate_welch(x, fft_size=256, sample_rate=44100, log_freq=False, throw_dc=True):
    """
    spectral estimation using welch method
    :param x: input signal
    :param fft_size: fft size
    :param sample_rate: sample rate
    :param log_freq: interpolate frequencies to log scale
    :param throw_dc: throw out DC bias component
    :return: spectral estimation
    """
    f, pxx = welch(x, fs=sample_rate, nperseg=fft_size)
    pxx = 20 * np.log(pxx)

    if log_freq:
        f, pxx = to_log_freq(f, pxx)

    if throw_dc:
        throw_idx = int(fft_size / 256)
        if throw_idx < 1:
            throw_idx = 1
        return f[throw_idx:], pxx[throw_idx:]
    else:
        return f, pxx


def sigmoid(x):
    """
    get sigmoid for numpy array
    :param x: input array
    :param scale: scaler
    :return:
    """
    return 1 / (1 + np.exp(-x))
