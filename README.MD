# Image Processing Practical Works

**Author:** Fox 
**Course:** Image Processing  
**Date:** November 2025  

## Context

This repository contains a collection of practical works (TPs) for an introductory course on image processing. The implementations are manual, using nested for loops to manipulate pixels directly, without relying on high-level functions from libraries like OpenCV or scikit-image. This approach ensures a deep understanding of the underlying algorithms. Images are represented as matrices of pixel values (grayscale 0-255 or RGB), and operations are performed on these matrices using NumPy for array handling, Matplotlib for visualization, and PIL (Pillow) solely for loading and saving images.

The TPs are organized into 7 chapters, covering fundamental concepts from basic image manipulation to advanced techniques like Fourier transforms and morphological operations. Each chapter builds on the previous one, starting with simple conversions and progressing to complex segmentation and binary image processing.

## Objectives

The main objectives of these TPs are:
- To understand the numerical representation of images as matrices.
- To implement core image processing algorithms manually to grasp their mechanics.
- To explore transformations that enhance, filter, or analyze images.
- To learn about frequency domain processing, edge detection, segmentation, and morphological operations.
- To practice Python programming with NumPy for efficient array operations and loops for pixel-wise manipulations.

All implementations handle image borders using zero padding or replication as appropriate, and results are saved as PNG files in dedicated output folders.

## Prerequisites

- **Python 3.7+**: The scripts are written in Python.
- **Libraries**:
  - NumPy: For array operations.
  - Pillow (PIL): For image loading and saving.
  - Matplotlib: For plotting histograms, profiles, and spectra.
- **Images**: Two input images are required:
  - `inputs/img1.png`: Grayscale image for most tests.
  - `inputs/img2.png`: Color or binary image depending on the TP.

## Installation

1. Clone or download the repository.
2. Install Python if not already installed.
3. Install the required libraries using pip:
   ```
   pip install numpy pillow matplotlib
   ```
   Alternatively, if using a virtual environment, activate it and run the above command.
4. Ensure the `inputs/` folder contains `img1.png` and `img2.png`.

## Usage

Each TP is a standalone Python script. To run a specific TP:
1. Navigate to the chapter folder (e.g., `cd Chap1`).
2. Execute the script: `python tpX_name.py`.
3. The output image(s) or plot(s) will be saved in the `outputs/` subfolder of the chapter.
4. For scripts that print statistics or chain codes, check the console output.

Note: Some scripts may display plots using Matplotlib; close the plot window to continue. Images are saved automatically.

## Project Structure

```
TPs de TI/
├── inputs/
│   ├── img1.png  # Grayscale image
│   └── img2.png  # Color/binary image
├── Chap1/
│   ├── README.MD  # Detailed description of Chapter 1
│   ├── tp1_*.py   # Scripts for Chapter 1
│   └── outputs/   # Output images for Chap1
├── Chap2/
│   ├── README.MD
│   ├── tp2_*.py
│   └── outputs/
├── ...
├── Chap7/
│   ├── README.MD
│   ├── tp7_*.py
│   └── outputs/
├── requirements.txt  # List of dependencies
├── README.MD         # This file (main overview)
└── notes.txt         # Additional notes
```

Each chapter folder contains a `README.MD` with a focused description of the chapter's topics, objectives, and a list of TPs with brief explanations.

## Detailed Chapters and TPs

### Chapter 1: Introduction

This chapter introduces basic image representations and simple manipulations to understand pixel-level operations.

- **tp1_rgb_to_gray.py**: Converts a color RGB image (`inputs/img2.png`) to grayscale using the weighted average formula (0.299*R + 0.587*G + 0.114*B). Loops over each pixel to compute the grayscale value. Saves `output_tp1_rgb_to_gray.png`. Purpose: Simplify processing by reducing to single-channel images.

- **tp1_gray_quantization.py**: Reduces the number of intensity levels in a grayscale image (`inputs/img1.png`) to K bits (e.g., K=4). Maps values using floor division and scaling. Saves `output_tp1_gray_quantization.png`. Purpose: Demonstrate compression and quality loss; test with K=2,4,6 by changing the variable.

- **tp1_spatial_sampling.py**: Downsamples the image (`inputs/img1.png`) by a factor N (e.g., N=2), keeping every N-th pixel. Handles non-divisible dimensions by flooring. Saves `output_tp1_spatial_sampling.png`. Purpose: Illustrate spatial resolution reduction for compression.

- **tp1_intensity_profile.py**: Extracts and plots the intensity values along a horizontal line at the middle of the image (`inputs/img1.png`). Collects values in a list and uses Matplotlib to plot. Saves `output_tp1_intensity_profile.png`. Purpose: Visualize intensity variations for edge analysis.

### Chapter 2: Basic Processing

Focuses on histograms, point-wise transformations, and basic operations to enhance or analyze images.

- **tp2_histogram.py**: Computes the histogram of a grayscale image (`inputs/img1.png`) by counting occurrences of each intensity (0-255). Plots a bar chart with Matplotlib. Saves `output_tp2_histogram.png`. Purpose: Analyze image dynamics for equalization.

- **tp2_stats.py**: Calculates mean, standard deviation, min, and max intensities (`inputs/img1.png`). Uses loops for summation and variance. Prints results to console; saves original image as `output_tp2_stats.png`. Purpose: Quantify image quality.

- **tp2_linear_stretch.py**: Stretches the intensity range to [0,255] using min-max normalization (`inputs/img1.png`). Finds global min/max, then scales each pixel. Saves `output_tp2_linear_stretch.png`. Purpose: Improve contrast in low-dynamic images.

- **tp2_linear_saturation.py**: Similar to stretch but saturates values outside [Smin, Smax] (e.g., 50-200). Clips to 0 or 255. Saves `output_tp2_linear_saturation.png`. Purpose: Control outliers while enhancing contrast.

- **tp2_piecewise_linear.py**: Applies piecewise linear transformation with defined breakpoints (e.g., [0,50,128,200,255]). Interpolates linearly between points. Saves `output_tp2_piecewise_linear.png`. Purpose: Fine-tune contrast in specific ranges.

- **tp2_gamma_correction.py**: Applies gamma correction (I' = 255 * (I/255)^(1/gamma)) with gamma=2.2 (`inputs/img1.png`). Loops over pixels. Saves `output_tp2_gamma_correction.png`. Purpose: Compensate for non-linear display perception.

- **tp2_hist_equal.py**: Performs global histogram equalization using CDF mapping (`inputs/img1.png`). Computes histogram, CDF, and remaps intensities. Saves `output_tp2_hist_equal.png`. Purpose: Uniformize distribution for optimal contrast.

- **tp2_local_hist_equal.py**: Equalizes histogram locally in NxN windows (e.g., N=7) (`inputs/img1.png`). For each pixel, computes local hist/CDF. Saves `output_tp2_local_hist_equal.png`. Purpose: Adapt contrast to local variations.

- **tp2_logical_ops.py**: Performs AND/OR on two binary images (thresholded from `inputs/img1.png` and `inputs/img2.png`). Loops for pixel-wise operations. Saves `output_tp2_logical_and.png` and `output_tp2_logical_or.png`. Purpose: Boolean combination for masking.

- **tp2_arithmetic_ops.py**: Adds, subtracts, or multiplies images (e.g., `inputs/img1.png` and `inputs/img2.png`), with clipping. Saves separate outputs. Purpose: Fusion, difference detection, or scaling.

- **tp2_interp_nearest.py**: Upsamples image by factor F (e.g., F=2) using nearest neighbor (`inputs/img1.png`). Copies pixel values to new grid. Saves `output_tp2_interp_nearest.png`. Purpose: Basic zooming.

- **tp2_interp_bilinear.py**: Bilinear interpolation for upsampling (`inputs/img1.png`). Computes weighted averages of 4 neighbors. Saves `output_tp2_interp_bilinear.png`. Purpose: Smooth zooming.

- **tp2_interp_bicubic.py**: Bicubic interpolation approximation (`inputs/img1.png`). Simplified version using bilinear. Saves `output_tp2_interp_bicubic.png`. Purpose: High-quality upsampling (full bicubic would require more complex kernels).

### Chapter 3: Convolution

Introduces spatial filtering via convolution for smoothing and noise reduction.

- **tp3_convolution_2d.py**: General 2D convolution function with kernel application (`inputs/img1.png`). Pads with zeros, sums weighted neighbors. Saves `output_tp3_convolution_2d.png`. Purpose: Base for all filters.

- **tp3_mean_filter.py**: Applies mean filter (average of NxN neighbors, e.g., N=3) (`inputs/img1.png`). Uses convolution. Saves `output_tp3_mean_filter.png`. Purpose: Blur and reduce noise.

- **tp3_gaussian_filter.py**: Gaussian smoothing with computed kernel (sigma=1.0, N=5) (`inputs/img1.png`). Exponential weights. Saves `output_tp3_gaussian_filter.png`. Purpose: Preserve edges while smoothing.

- **tp3_median_filter.py**: Median filter on color image (`inputs/img2.png`). Sorts neighbor values, takes median. Saves `output_tp3_median_filter.png`. Purpose: Remove salt-and-pepper noise.

### Chapter 4: Fourier Transform

Explores frequency domain for spectral analysis and filtering.

- **tp4_dft_1d.py**: Computes 1D DFT on a synthetic signal (sin + noise). Manual loops for complex summation. Plots magnitude. Saves `output_tp4_dft_1d.png`. Purpose: Understand frequency components.

- **tp4_dft_2d.py**: 2D DFT on a 64x64 patch (`inputs/img1.png`). Applies 1D DFT row-wise then column-wise. Saves magnitude plot. Purpose: Visualize image spectrum.

- **tp4_fft_shift.py**: Shifts DFT to center low frequencies. Rearranges quadrants. Saves shifted magnitude plot. Purpose: Standard spectrum display.

- **tp4_lowpass_freq.py**: Applies lowpass filter in frequency domain (removes high frequencies). IDFT back (simplified). Saves filtered magnitude. Purpose: Smooth image.

- **tp4_highpass_freq.py**: Highpass filter (keeps high frequencies). Saves filtered magnitude. Purpose: Enhance edges.

- **tp4_notch_filter.py**: Removes specific frequency peaks (e.g., at (10,10)). Saves filtered magnitude. Purpose: Eliminate periodic noise.

- **tp4_enhance_highpass.py**: Adds highpass to original for sharpening. Saves enhanced magnitude. Purpose: Boost details.

### Chapter 5: Edges

Detects edges using gradients and refinement techniques.

- **tp5_gradient_finite_diff.py**: Computes gradient magnitude using finite differences (`inputs/img1.png`). dx/dy from neighbors. Saves `output_tp5_gradient_finite_diff.png`. Purpose: Basic edge strength.

- **tp5_sobel.py**: Sobel operator with weighted kernels (`inputs/img1.png`). Convolution for dx/dy. Saves `output_tp5_sobel.png`. Purpose: Robust gradient.

- **tp5_prewitt.py**: Prewitt operator (simpler weights) (`inputs/img1.png`). Similar to Sobel. Saves `output_tp5_prewitt.png`. Purpose: Alternative gradient.

- **tp5_roberts.py**: Roberts cross operator (2x2 kernels) (`inputs/img1.png`). Diagonal differences. Saves `output_tp5_roberts.png`. Purpose: Detect 45° edges.

- **tp5_laplacian.py**: Laplacian for second derivatives (`inputs/img1.png`). Kernel for zero-crossings. Saves `output_tp5_laplacian.png`. Purpose: Edge detection via curvature.

- **tp5_edge_threshold.py**: Thresholds gradient magnitude (e.g., >100) (`inputs/img1.png`). Binarizes edges. Saves `output_tp5_edge_threshold.png`. Purpose: Extract strong edges.

- **tp5_non_max_suppression.py**: Thins edges to 1-pixel width using direction (`inputs/img1.png`). Suppresses non-maxima. Saves `output_tp5_non_max_suppression.png`. Purpose: Refine edges.

- **tp5_hysteresis.py**: Links edges with double threshold and connectivity (`inputs/img1.png`). Uses queue for propagation. Saves `output_tp5_hysteresis.png`. Purpose: Complete edge chains.

### Chapter 6: Segmentation

Divides images into regions or objects.

- **tp6_simple_threshold.py**: Binarizes with fixed threshold (128) (`inputs/img1.png`). Saves `output_tp6_simple_threshold.png`. Purpose: Basic segmentation.

- **tp6_otsu_threshold.py**: Automatic threshold via variance maximization (`inputs/img1.png`). Computes optimal T. Saves `output_tp6_otsu_threshold.png`. Purpose: Bimodal separation.

- **tp6_region_growing.py**: Grows region from seed based on similarity (`inputs/img2.png`). Queue-based expansion. Saves `output_tp6_region_growing.png`. Purpose: Homogeneous regions.

- **tp6_connected_components_labeling.py**: Labels connected components in binary image (`inputs/img1.png`). Two-pass with equivalence. Saves colored `output_tp6_connected_components_labeling.png`. Purpose: Count objects.

- **tp6_watershed.py**: Simplified watershed on gradient (`inputs/img1.png`). Sorts pixels, assigns labels. Saves colored `output_tp6_watershed.png`. Purpose: Oversegmentation.

- **tp6_kmeans_segmentation.py**: Clusters pixels in RGB space (K=3) (`inputs/img2.png`). Iterative assignment. Saves segmented `output_tp6_kmeans_segmentation.png`. Purpose: Color-based grouping.

### Chapter 7: Binary Images

Processes binary images with morphological operations.

- **tp7_connected_components_labeling.py**: Same as Chap6, labels binary components. Saves colored output. Purpose: Analyze binary objects.

- **tp7_freeman_chain_code.py**: Encodes contour with 8-direction codes (`inputs/img2.png`). Follows boundary. Prints chain; saves binary. Purpose: Compact shape representation.

- **tp7_morph_erosion.py**: Erodes binary image (shrinks objects) (`inputs/img1.png`). Checks all neighbors. Saves `output_tp7_morph_erosion.png`. Purpose: Remove noise.

- **tp7_morph_dilation.py**: Dilates binary image (expands objects) (`inputs/img1.png`). Checks any neighbor. Saves `output_tp7_morph_dilation.png`. Purpose: Fill gaps.

- **tp7_morph_opening.py**: Erosion followed by dilation (`inputs/img1.png`). Removes small noise. Saves `output_tp7_morph_opening.png`. Purpose: Clean without shrinking.

- **tp7_morph_closing.py**: Dilation followed by erosion (`inputs/img1.png`). Fills small holes. Saves `output_tp7_morph_closing.png`. Purpose: Close gaps.

- **tp7_morph_gradient.py**: Dilation minus erosion (`inputs/img1.png`). Extracts boundaries. Saves `output_tp7_morph_gradient.png`. Purpose: Morphological edges.

## Notes

- All scripts are minimal and functional, with brief docstrings.
- Borders are handled with zero padding unless specified.
- For complex TPs, implementations are simplified for educational purposes.
- Run in a Python environment with the required libraries.
- Outputs are in PNG format; plots are saved as images.

For questions or improvements, refer to the course materials.