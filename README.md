# Explainable Emotion Detection using B-cos CNN on FER2013

This project focuses on **facial emotion recognition** using deep learning, with a specific emphasis on **model interpretability** through B-cos networks.

---

## Overview

Emotion recognition using CNNs has achieved strong performance, but most models act as **black boxes**.
In this project, we explore how **B-cos networks** can make CNN predictions more interpretable by highlighting the regions of the image responsible for decisions.

The goal is to compare:

* A **standard CNN**
* A **B-cos enhanced CNN**

in terms of both **prediction behavior** and **interpretability**.

---

## My Role in the Project

This was a **collaborative academic project**, where I primarily worked on:

* Implementing CNN and B-cos model architectures
* Setting up and experimenting with the training pipeline
* Analyzing model predictions and interpretability behavior
* Comparing how different models focus on facial features

---

## Dataset

We use the **FER2013 dataset**, consisting of 48×48 grayscale facial images categorized into 7 emotions:

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

The dataset contains:

* ~28k training images
* ~7k test images

---

## Model Architecture

### Standard CNN

* Convolution → BatchNorm → ReLU → Pooling layers
* Fully connected layers for classification
* Learns spatial features from input images

---

### B-cos CNN

* Replaces standard convolution layers with **B-cos layers**
* Uses modified linear transformations for classification
* Improves interpretability by aligning gradients with input features
* Highlights important facial regions influencing predictions

---

## Approach

* Implemented both CNN and B-cos variants using PyTorch
* Used **cross-entropy loss** for classification
* Optimized models using **Stochastic Gradient Descent (SGD)**
* Applied preprocessing such as normalization
* Compared models based on prediction behavior and interpretability

---

## Experiments

The project involved:

* Training both architectures on FER2013
* Observing prediction patterns across emotion classes
* Inspecting how models respond to facial features
* Comparing feature focus between CNN and B-cos CNN

---

## Results & Insights

* Both models learned meaningful patterns for emotion classification
* The **B-cos CNN showed improved interpretability** compared to the standard CNN
* The model focused on key facial regions such as:

  * eyes
  * mouth
* B-cos produced more structured and meaningful feature attributions

---

## Key Takeaways

* Interpretability can be improved without major architectural changes
* B-cos layers help make deep learning models more transparent
* Understanding model behavior is important for real-world applications

---

## References

1. Mollahosseini et al. (2017) – AffectNet Dataset
2. Zhang & Zhang (2017) – CNN-based Emotion Recognition
3. Montavon et al. (2017) – Deep Taylor Decomposition
4. Ribeiro et al. (2016) – LIME
5. Simonyan et al. (2014) – Saliency Maps
6. Böhle et al. (2022) – B-cos Networks
7. Khaireddin & Chen (2021) – FER2013 Benchmark
