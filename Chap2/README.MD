# Chapter 2: Basic Processing

This chapter explores point-wise transformations and operations on images, including histogram analysis, contrast enhancement, arithmetic and logical operations, and interpolation. It focuses on improving image quality, analyzing distributions, and basic resizing without considering spatial neighbors.

## Practical Works

- **tp2_histogram.py**: Computes and plots the histogram of grayscale intensities. Uses `inputs/img1.png`, saves `outputs/output_tp2_histogram.png`.

- **tp2_stats.py**: Calculates mean, std, min, max of pixel values. Uses `inputs/img1.png`, prints to console, saves original as `outputs/output_tp2_stats.png`.

- **tp2_linear_stretch.py**: Stretches intensity range to [0,255] for full contrast. Uses `inputs/img1.png`, saves `outputs/output_tp2_linear_stretch.png`.

- **tp2_linear_saturation.py**: Stretches with saturation at extremes. Uses `inputs/img1.png`, saves `outputs/output_tp2_linear_saturation.png`.

- **tp2_piecewise_linear.py**: Applies piecewise linear mapping for custom contrast. Uses `inputs/img1.png`, saves `outputs/output_tp2_piecewise_linear.png`.

- **tp2_gamma_correction.py**: Corrects gamma for perceptual linearity. Uses `inputs/img1.png`, saves `outputs/output_tp2_gamma_correction.png`.

- **tp2_hist_equal.py**: Equalizes histogram globally for uniform distribution. Uses `inputs/img1.png`, saves `outputs/output_tp2_hist_equal.png`.

- **tp2_local_hist_equal.py**: Equalizes histogram locally in windows. Uses `inputs/img1.png`, saves `outputs/output_tp2_local_hist_equal.png`.

- **tp2_logical_ops.py**: Performs AND/OR on thresholded images. Uses `inputs/img1.png` and `inputs/img2.png`, saves AND/OR outputs.

- **tp2_arithmetic_ops.py**: Adds, subtracts, multiplies images with clipping. Uses `inputs/img1.png` and `inputs/img2.png`, saves separate outputs.

- **tp2_interp_nearest.py**: Nearest neighbor upsampling. Uses `inputs/img1.png`, saves `outputs/output_tp2_interp_nearest.png`.

- **tp2_interp_bilinear.py**: Bilinear interpolation upsampling. Uses `inputs/img1.png`, saves `outputs/output_tp2_interp_bilinear.png`.

- **tp2_interp_bicubic.py**: Simplified bicubic upsampling. Uses `inputs/img1.png`, saves `outputs/output_tp2_interp_bicubic.png`.