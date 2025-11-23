# Image Processing Practical Works

Author: Donfack Pascal

Course: Image Processing

## Organization

The project is organized into chapters, each in its own folder (Chap1/, Chap2/, etc.). Each chapter folder contains the Python scripts for the TPs and an outputs/ folder for the results.

## Chapters

### Chapter 1: Introduction

- tp1_rgb_to_gray.py: Converts RGB image to grayscale using weighted average.

- tp1_gray_quantization.py: Quantizes grayscale image to K bits.

- tp1_spatial_sampling.py: Spatial sampling by factor N.

- tp1_intensity_profile.py: Extracts intensity profile along horizontal line.

### Chapter 2: Basic Processing

- tp2_histogram.py: Computes histogram of grayscale image.

- tp2_stats.py: Computes statistics: mean, std, min, max.

- tp2_linear_stretch.py: Linear stretch to full range.

- tp2_linear_saturation.py: Linear stretch with saturation.

- tp2_piecewise_linear.py: Piecewise linear transformation.

- tp2_gamma_correction.py: Gamma correction.

- tp2_hist_equal.py: Global histogram equalization.

- tp2_local_hist_equal.py: Local histogram equalization.

- tp2_logical_ops.py: Logical operations: AND, OR.

- tp2_arithmetic_ops.py: Arithmetic operations: add, sub, mult.

- tp2_interp_nearest.py: Nearest neighbor interpolation.

- tp2_interp_bilinear.py: Bilinear interpolation.

- tp2_interp_bicubic.py: Bicubic interpolation.

### Chapter 3: Convolution

- tp3_convolution_2d.py: 2D convolution.

- tp3_mean_filter.py: Mean filter.

- tp3_gaussian_filter.py: Gaussian filter.

- tp3_median_filter.py: Median filter.

### Chapter 4: Fourier Transform

- tp4_dft_1d.py: 1D DFT.

- tp4_dft_2d.py: 2D DFT.

- tp4_fft_shift.py: FFT shift.

- tp4_lowpass_freq.py: Lowpass frequency filter.

- tp4_highpass_freq.py: Highpass frequency filter.

- tp4_notch_filter.py: Notch filter.

- tp4_enhance_highpass.py: Highpass enhancement.

### Chapter 5: Edges

- tp5_gradient_finite_diff.py: Gradient by finite differences.

- tp5_sobel.py: Sobel operator.

- tp5_prewitt.py: Prewitt operator.

- tp5_roberts.py: Roberts operator.

- tp5_laplacian.py: Laplacian operator.

- tp5_edge_threshold.py: Edge thresholding.

- tp5_non_max_suppression.py: Non-max suppression.

- tp5_hysteresis.py: Hysteresis.

### Chapter 6: Segmentation

- tp6_simple_threshold.py: Simple thresholding.

- tp6_otsu_threshold.py: Otsu thresholding.

- tp6_region_growing.py: Region growing.

- tp6_connected_components_labeling.py: Connected components labeling.

- tp6_watershed.py: Watershed segmentation.

- tp6_kmeans_segmentation.py: K-means segmentation.

### Chapter 7: Binary Images

- tp7_connected_components_labeling.py: Connected components labeling.

- tp7_freeman_chain_code.py: Freeman chain code.

- tp7_morph_erosion.py: Morphological erosion.

- tp7_morph_dilation.py: Morphological dilation.

- tp7_morph_opening.py: Morphological opening.

- tp7_morph_closing.py: Morphological closing.

- tp7_morph_gradient.py: Morphological gradient.

Each script loads the specified image, applies the algorithm manually with loops, and saves the result in the outputs/ folder.