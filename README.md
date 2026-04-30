# CS83 Assignment 2: Feature Detection, Matching, and Deep Learning Methods

Mithun Rameshkumar

**[Read the full report (PDF)](report.pdf)**

## Overview

Two-part computer vision project comparing classical and deep-learning approaches to image matching:

- **Part 1 — Classical pipeline (from scratch in NumPy).** Harris corner detector → SIFT descriptors at the detected corners → Lowe's ratio test → RANSAC homography estimation. OpenCV is used only for SIFT descriptor computation and image I/O.
- **Part 2 — Siamese network.** Two-tower ResNet18 (ImageNet-pretrained, shared weights) with a similarity head, trained on Oxford5k with Binary Cross-Entropy to predict whether two images depict the same landmark.

## Results

**Part 1 — Match-quality ranking across 5 image pairs:**

| Pair                  | Matches | Inliers | Inlier ratio | Quality Q |
| --------------------- | ------- | ------- | ------------ | --------- |
| test_image_test_image1| 4       | 4       | 1.00         | 1.0000    |
| panorama1_panorama2   | 420     | 411     | 0.98         | 0.9015    |
| img1_img2             | 94      | 68      | 0.72         | 0.2767    |
| img5_img6             | 30      | 6       | 0.20         | 0.0866    |
| img3_img4             | 45      | 6       | 0.13         | 0.0611    |

**Part 2 — Siamese network on 1335 held-out Oxford5k pairs (50 epochs, BCE):**

| Metric    | Value   |
| --------- | ------- |
| Accuracy  | 98.73%  |
| Precision | 0.9921  |
| Recall    | 0.9640  |
| F1        | 0.9778  |

The classical pipeline recovers pixel-accurate correspondences and degrades gracefully as viewpoint change grows. The Siamese network handles intra-class variation (lighting, time of day, viewpoint) that the handcrafted descriptors fail on.

## Layout

```
assignment2/
├── report.pdf                # full writeup
├── part1/                    # Harris + SIFT + RANSAC pipeline
├── part2/                    # Siamese network training
├── data/image_pairs/         # test image pairs
├── PA2-README.md             # original assignment spec
└── *.jpg                     # sample outputs (Harris, descriptors, matching)
```

## Running

```bash
# Part 1: classical pipeline
cd part1
python main.py
python run_experiments.py     # SIFT parameter sweep

# Part 2: Siamese network
cd part2
python main.py                # trains and evaluates
```
