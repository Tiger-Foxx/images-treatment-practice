# Chapter 3: Convolution

This chapter introduces spatial filtering through 2D convolution, applying kernels to neighborhoods for smoothing, noise reduction, and edge preservation. It implements manual convolution and specific filters like mean, Gaussian, and median.

## Practical Works

- **tp3_convolution_2d.py**: General 2D convolution with a kernel. Uses `inputs/img1.png`, saves `outputs/output_tp3_convolution_2d.png`.

- **tp3_mean_filter.py**: Applies mean (average) filter for blurring. Uses `inputs/img1.png`, saves `outputs/output_tp3_mean_filter.png`.

- **tp3_gaussian_filter.py**: Gaussian smoothing to reduce noise while preserving edges. Uses `inputs/img1.png`, saves `outputs/output_tp3_gaussian_filter.png`.

- **tp3_median_filter.py**: Median filter for impulse noise removal. Uses `inputs/img2.png`, saves `outputs/output_tp3_median_filter.png`.