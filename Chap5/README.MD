# Chapter 5: Edges

This chapter focuses on edge detection using gradient operators, thresholding, and refinement techniques like non-max suppression and hysteresis. It aims to extract contours from intensity variations.

## Practical Works

- **tp5_gradient_finite_diff.py**: Computes gradient magnitude via finite differences. Uses `inputs/img1.png`, saves `outputs/output_tp5_gradient_finite_diff.png`.

- **tp5_sobel.py**: Applies Sobel operator for weighted gradients. Uses `inputs/img1.png`, saves `outputs/output_tp5_sobel.png`.

- **tp5_prewitt.py**: Prewitt operator for simpler gradients. Uses `inputs/img1.png`, saves `outputs/output_tp5_prewitt.png`.

- **tp5_roberts.py**: Roberts cross for diagonal edges. Uses `inputs/img1.png`, saves `outputs/output_tp5_roberts.png`.

- **tp5_laplacian.py**: Laplacian for second derivatives. Uses `inputs/img1.png`, saves `outputs/output_tp5_laplacian.png`.

- **tp5_edge_threshold.py**: Thresholds gradients to binarize edges. Uses `inputs/img1.png`, saves `outputs/output_tp5_edge_threshold.png`.

- **tp5_non_max_suppression.py**: Thins edges to single pixels. Uses `inputs/img1.png`, saves `outputs/output_tp5_non_max_suppression.png`.

- **tp5_hysteresis.py**: Links edges with double thresholds. Uses `inputs/img1.png`, saves `outputs/output_tp5_hysteresis.png`.