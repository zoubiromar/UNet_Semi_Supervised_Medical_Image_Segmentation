# UNet_Semi_Supervised_Medical_Image_Segmentation

## Overview
This project, part of the ACDC challenge, focuses on developing a high-performance model to segment 2D heart images into myocardium, left ventricle, and right ventricle.

## Methodology
The project employed semi-supervised learning with a Unet model. Key methodologies included early stopping with a patience of 10 iterations and the use of the FixMatch algorithm for semi-supervised learning. Data augmentation techniques like flips, rotations, translations, openings, and closings were applied.

## Dataset
The training dataset included 204 labeled and 1004 unlabeled images, with a separate validation set of 74 images. Test data was reserved for final evaluation by the instructor.

## Competition
The project was a competition among ten teams, each comprising 4-5 members. Teams were provided with a base code implementing the Unet model, and performance metrics included the Dice coefficient, Hausdorff distance (HDD), and average surface distance (ASD). Our team was ranked 4th.

## Metrics & Results
The project emphasized exploring strategies, optimizing parameters, and understanding model performance factors. Our team achieved fourth place, demonstrating the effectiveness of our chosen methodologies and model optimizations.

## Conclusion
The project highlights the significance of data augmentation, semi-supervised learning, and careful parameter optimization in medical image segmentation using deep learning.
