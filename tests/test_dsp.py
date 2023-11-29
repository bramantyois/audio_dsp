from src.filter.onepole import OnePoleHP, OnePoleLP
from src.utils import generate_sine_sweep, estimate_welch
import numpy as np


def test_filter_onepole():
    sr = 44100
    fc = sr * 0.25

    input_x = generate_sine_sweep(duration=0.5, sample_rate=sr, noise_std=0.2)

    hp = OnePoleHP(sr, fc=fc)
    lp = OnePoleLP(sr, fc=fc)

    hp_y = hp.process_block(input_x)
    lp_y = lp.process_block(input_x)

    f, pxx = estimate_welch(input_x, sample_rate=sr, throw_dc=True)
    hp_f, hp_pxx = estimate_welch(hp_y, sample_rate=sr, throw_dc=True)
    lp_f, lp_pxx = estimate_welch(lp_y, sample_rate=sr, throw_dc=True)

    # make sure lower frequencies are attenuated
    assert hp_pxx[:10].mean() < pxx[:10].mean()
    assert hp_pxx[:10].mean() < lp_pxx[:10].mean()
    
    # now for the high frequencies
    assert lp_pxx[10:].mean() < pxx[10:].mean()
    assert lp_pxx[10:].mean() < hp_pxx[10:].mean()
