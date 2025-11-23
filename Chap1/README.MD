# Chapter 1: Introduction

This chapter introduces the fundamental concepts of digital image representation as matrices of pixel values (grayscale 0-255 or RGB). It covers basic manipulations to understand pixel-level operations, including conversion, quantization, sampling, and profiling. The goal is to grasp how images are numerically handled and manipulated manually.

## Practical Works

- **tp1_rgb_to_gray.py**: Converts RGB image to grayscale using weighted average (0.299*R + 0.587*G + 0.114*B). Uses `inputs/img2.png`, saves `outputs/output_tp1_rgb_to_gray.png`.

- **tp1_gray_quantization.py**: Quantizes grayscale levels to K bits (e.g., K=4), reducing precision. Uses `inputs/img1.png`, saves `outputs/output_tp1_gray_quantization.png`.

- **tp1_spatial_sampling.py**: Downsamples image by factor N (e.g., N=2) for compression. Uses `inputs/img1.png`, saves `outputs/output_tp1_spatial_sampling.png`.

- **tp1_intensity_profile.py**: Plots intensity along a horizontal line at image center. Uses `inputs/img1.png`, saves `outputs/output_tp1_intensity_profile.png`.