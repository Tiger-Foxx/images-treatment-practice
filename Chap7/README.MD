# Chapter 7: Binary Images

This chapter deals with processing binary images using morphological operations like erosion, dilation, opening, closing, and gradient, as well as labeling and chain coding for shape analysis.

## Practical Works

- **tp7_connected_components_labeling.py**: Labels binary components. Uses `inputs/img1.png`, saves colored output.

- **tp7_freeman_chain_code.py**: Encodes contours with Freeman codes. Uses `inputs/img2.png`, prints chain, saves binary.

- **tp7_morph_erosion.py**: Erodes binary image to shrink objects. Uses `inputs/img1.png`, saves `outputs/output_tp7_morph_erosion.png`.

- **tp7_morph_dilation.py**: Dilates binary image to expand objects. Uses `inputs/img1.png`, saves `outputs/output_tp7_morph_dilation.png`.

- **tp7_morph_opening.py**: Erosion then dilation for noise removal. Uses `inputs/img1.png`, saves `outputs/output_tp7_morph_opening.png`.

- **tp7_morph_closing.py**: Dilation then erosion for hole filling. Uses `inputs/img1.png`, saves `outputs/output_tp7_morph_closing.png`.

- **tp7_morph_gradient.py**: Morphological gradient (dilation - erosion). Uses `inputs/img1.png`, saves `outputs/output_tp7_morph_gradient.png`.