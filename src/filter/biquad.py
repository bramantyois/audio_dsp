import numpy as np
import math

from scipy.signal import lfilter


class Biquad:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.fc = 1000
        self.q_val = 0.707

        self.buffer_out = np.zeros(2)
        self.buffer_in = np.zeros(2)

        self.b = np.zeros(3)
        self.a = np.zeros(3)

    def reset_buffer(self):
        self.buffer_out = np.zeros(2)
        self.buffer_in = np.zeros(2)

    def calc_coefficients(self, freq, gain_db, q_val):
        aa = math.pow(10, gain_db / 40)
        omega = 2 * math.pi * freq / self.sample_rate
        t_sin = math.sin(omega)
        t_cos = math.cos(omega)

        alpha = 0.5 * t_sin / q_val
        beta = math.sqrt(aa) / q_val

        self.calc_type(aa, omega, t_sin, t_cos, alpha, beta)

    def calc_type(self, aa, omega, t_sin, t_cos, alpha, beta):
        pass

    def process_block_w_param(self, x, freq=1000, gain_db=20, q_val=0.707):
        self.calc_coefficients(freq, gain_db, q_val)

        out_x = lfilter(self.a, self.b, x)

        return out_x


class BiquadHighShelf(Biquad):
    def __init__(self, sample_rate):
        super().__init__(sample_rate)

    def calc_type(self, aa, omega, t_sin, t_cos, alpha, beta):
        self.b[0] = aa * ((aa + 1) + (aa - 1) * t_cos + beta * t_sin)
        self.b[1] = -2 * aa * ((aa - 1) + (aa + 1) * t_cos)
        self.b[2] = aa * ((aa + 1) + (aa - 1) * t_cos - beta * t_sin)
        self.a[0] = (aa + 1) - (aa - 1) * t_cos + beta * t_sin
        self.a[1] = 2 * ((aa - 1) - (aa + 1) * t_cos)
        self.a[2] = (aa + 1) - (aa - 1) * t_cos - beta * t_sin

        self.b = self.b / self.a[0]
        self.a = self.a / self.a[0]


class BiquadLowShelf(Biquad):
    def __init__(self, sample_rate):
        super().__init__(sample_rate)

    def calc_type(self, aa, omega, t_sin, t_cos, alpha, beta):
        self.b[0] = aa * ((aa + 1) - (aa - 1) * t_cos + beta * t_sin)
        self.b[1] = 2 * aa * ((aa - 1) - (aa + 1) * t_cos)
        self.b[2] = aa * ((aa + 1) - (aa - 1) * t_cos - beta * t_sin)
        self.a[0] = (aa + 1) + (aa - 1) * t_cos + beta * t_sin
        self.a[1] = -2 * ((aa - 1) + (aa + 1) * t_cos)
        self.a[2] = (aa + 1) + (aa - 1) * t_cos - beta * t_sin

        self.b = self.b / self.a[0]
        self.a = self.a / self.a[0]


if __name__ == "__main__":
    from utils import generate_sine_sweep
    import matplotlib.pyplot as plt

    sr = 44100

    input_x = generate_sine_sweep(duration=0.5)

    hp = BiquadLowShelf(sr)

    filtered = hp.process_block_w_param(input_x)

    plt.plot(filtered, label="filtered")
    plt.plot(input_x, alpha=0.2, label="input")
    print(filtered.shape)
    print(input_x.shape)
    plt.legend()
    plt.show()
