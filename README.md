# Image and Video Signal Processing Lab
## Eye Motion Analysis and Human Gait Tracking

## Overview
This repository presents a complete image and video signal processing lab, implemented in MATLAB, with a strong emphasis on **eye motion research** and **walking (gait) analysis**.

The work focuses on processing real experimental data under non-ideal conditions, including noise, illumination changes, motion blur, and calibration inaccuracies. The goal is to demonstrate research-level understanding of motion analysis in biological systems, not only algorithmic implementation.

---

## Experiments Summary

### 1. Image Segmentation and Feature Extraction
This experiment explores classical image processing techniques for object detection and analysis.

**Key methods**
- Grayscale histogram analysis and threshold selection
- Morphological operations for background removal
- Edge detection using Laplacian of Gaussian and morphological gradients
- Object labeling and feature extraction
- Orientation-based object filtering

**Main outcome**  
Demonstrates controlled segmentation, mask refinement, and feature-based object classification.

---

### 2. Noise Modeling and Image Denoising
This experiment studies the effect of noise and filter selection on image quality.

**Key methods**
- Artificial salt and pepper noise generation
- Median filtering with multiple kernel sizes
- Quantitative evaluation using Mean Squared Error
- Repeated trials for robustness
- Statistical comparison using one-way ANOVA

**Main outcome**  
Shows principled filter selection supported by statistical analysis rather than visual inspection alone.

---

### 3. Eye Motion Detection and Gaze Estimation
This is a core part of the repository and focuses on video-based eye motion research.

**Key methods**
- Eye and pupil detection from video frames
- Median filtering for noise suppression
- Circular object detection for pupil localization
- Calibration using multiple gaze directions
- Gaze classification using Euclidean distance in feature space

**Main outcome**  
The system identifies which target a subject is looking at in each frame, while handling blinking, head movement, and imperfect calibration. This experiment reflects realistic challenges in eye tracking and visual attention research.

---

### 4. Walking and Gait Motion Analysis
This experiment analyzes lower-limb motion during walking using video tracking.

**Key methods**
- Marker-based tracking of the right leg
- Frame-by-frame position extraction
- Velocity estimation from spatial trajectories
- Joint angle analysis for hip and knee motion
- Evaluation of camera placement and lighting conditions

**Main outcome**  
Demonstrates foundational gait analysis, highlighting both the potential and the limitations of vision-based motion tracking.

---

## Skills Demonstrated
- Image and video signal processing
- Eye motion and gaze estimation
- Human gait and walking analysis
- Feature extraction and segmentation
- Noise modeling and filtering
- Statistical evaluation of algorithms
- MATLAB-based experimental research workflows
