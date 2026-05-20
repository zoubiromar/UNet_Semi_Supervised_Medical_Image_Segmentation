# Semi-Supervised U-Net for Cardiac MRI Segmentation

Segmenting cardiac MRI images into three anatomical regions (myocardium, left ventricle, right ventricle) using a U-Net trained with semi-supervised learning to exploit unlabeled data.

## Result

**4th place** out of 10 competing teams in the ACDC cardiac MRI segmentation challenge. Scoring was based on the Dice coefficient, Hausdorff distance (HDD), and average surface distance (ASD) on a held-out test set.

## Problem

Labeled medical imaging data is expensive. Annotating MRI slices requires clinical expertise and time. The ACDC challenge gave each team a small labeled training set alongside a much larger pool of unlabeled images. The question we were exploring: can the unlabeled images meaningfully improve segmentation quality, and if so, how?

## Approach

- **Architecture:** U-Net (the standard biomedical-segmentation backbone, encoder-decoder with skip connections).
- **Semi-supervised learning:** FixMatch algorithm. Generate pseudo-labels for the unlabeled images using weakly augmented inputs; train on those pseudo-labels with strongly augmented inputs, gated by a confidence threshold.
- **Data augmentation:** flips, rotations, translations, morphological openings, morphological closings.
- **Training control:** early stopping with patience of 10 iterations to prevent overfitting.

## Dataset

- 204 labeled training images
- 1,004 unlabeled training images
- 74 validation images
- Held-out test set graded by the instructor

## Stack

PyTorch, U-Net, FixMatch.

## Files

- `mainSegmentationChallenge.ipynb`: main training and evaluation notebook
- `UNet_Base.py`: baseline U-Net implementation
- `multi_UNet_complex.py`: extended U-Net variant
- `model_Gagant.py`: our final submitted model
- `pseudo_label.py`: FixMatch pseudo-labeling logic
- `training.py`: training loop
- `medicalDataLoader.py`: dataset loading + augmentation
- `Segmentation_Medicale_d_un_Coeur.pdf`: full project report

## Context

ETS Montréal Master of Engineering coursework. Team competition; my team ranked 4th.
