# Chapter 4: Fourier Transform

This chapter delves into the frequency domain using Discrete Fourier Transform (DFT) for spectral analysis and filtering. It covers 1D/2D DFT, shifting, and frequency-based filters for noise reduction or enhancement.

## Practical Works

- **tp4_dft_1d.py**: Computes 1D DFT on a synthetic signal, plots magnitude. Saves `outputs/output_tp4_dft_1d.png`.

- **tp4_dft_2d.py**: 2D DFT on image patch, visualizes spectrum. Uses `inputs/img1.png`, saves plot.

- **tp4_fft_shift.py**: Shifts DFT quadrants for centered spectrum. Uses DFT result, saves `outputs/output_tp4_fft_shift.png`.

- **tp4_lowpass_freq.py**: Lowpass filter in frequency domain. Saves filtered spectrum.

- **tp4_highpass_freq.py**: Highpass filter for edge enhancement. Saves filtered spectrum.

- **tp4_notch_filter.py**: Removes specific frequency peaks. Saves filtered spectrum.

- **tp4_enhance_highpass.py**: Combines highpass with original for sharpening. Saves enhanced spectrum.