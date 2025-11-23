# Chapter 6: Segmentation

This chapter covers image segmentation techniques to divide images into meaningful regions or objects, including thresholding, region growing, labeling, watershed, and clustering.

## Practical Works

- **tp6_simple_threshold.py**: Fixed threshold binarization. Uses `inputs/img1.png`, saves `outputs/output_tp6_simple_threshold.png`.

- **tp6_otsu_threshold.py**: Automatic threshold via variance. Uses `inputs/img1.png`, saves `outputs/output_tp6_otsu_threshold.png`.

- **tp6_region_growing.py**: Grows regions from seed based on similarity. Uses `inputs/img2.png`, saves `outputs/output_tp6_region_growing.png`.

- **tp6_connected_components_labeling.py**: Labels connected components. Uses `inputs/img1.png`, saves colored `outputs/output_tp6_connected_components_labeling.png`.

- **tp6_watershed.py**: Watershed segmentation on gradient. Uses `inputs/img1.png`, saves colored `outputs/output_tp6_watershed.png`.

- **tp6_kmeans_segmentation.py**: K-means clustering on RGB pixels. Uses `inputs/img2.png`, saves `outputs/output_tp6_kmeans_segmentation.png`.