# Rock Classification with MobileNetV2

## Overview

This project explores image classification using transfer learning in PyTorch. The goal was to build a model that can look at an image of a rock and predict which rock class it belongs to.

Instead of building and training a convolutional neural network from scratch, I used **MobileNetV2**, a pretrained CNN, as the starting point. The existing model is used to extract useful visual features from the images, while the final classification layer is adapted and trained for the rock classification task.

The project was mainly an opportunity to get hands-on experience with image classification, PyTorch, and transfer learning while applying them to a real dataset.

## Dataset

The images come from the **Rock Classification Dataset** on Kaggle. The dataset was originally collected by members of the BRAC University Mars Rover Team for research involving the identification of similar rocks on Mars-like surfaces. It contains images of different types of rocks that can be used for multiclass image classification.

[Rock Classification Dataset on Kaggle](https://www.kaggle.com/datasets/salmaneunus/rock-classification?utm_source=chatgpt.com)

## Approach

For this project, I used **transfer learning with MobileNetV2** rather than training an entire CNN from the beginning.

Transfer learning works well for this type of problem because a pretrained image model has already learned how to recognize useful visual features. Those learned features can then be reused for a new classification problem.

The general process used in the notebook is:

1. Load and preprocess the rock images.
2. Split the dataset into training and validation sets.
3. Load a pretrained MobileNetV2 model.
4. Freeze the pretrained model parameters.
5. Replace the original classifier with a new output layer for the rock classes.
6. Train the new classifier using the rock images.
7. Evaluate the model using the validation data.

The transfer-learning implementation was based in part on Ben Lambda's PyTorch MobileNetV2 transfer-learning notebook on Kaggle.

[PyTorch Transfer Learning with MobileNetV2 CNN](https://www.kaggle.com/code/benlambda/pytorch-transfer-learning-with-mobilenetv2-cnn?utm_source=chatgpt.com)

## Why MobileNetV2?

MobileNetV2 was used because it provides a practical pretrained convolutional neural network for image classification.

Rather than asking the model to learn basic visual features from the relatively small rock dataset, transfer learning allows those existing features to be reused. The final part of the network can then focus on learning the differences between the rock classes in this dataset.

This also makes training much more manageable than training a deep CNN completely from scratch.

## Training

The dataset is divided into **80% training data and 20% validation data**.

During training, images are passed through the pretrained MobileNetV2 network. Since the pretrained parameters are frozen, the main part being trained is the modified classifier at the end of the network.

The training data is used to update the classifier, while the validation data provides a way to see how well the model performs on images it was not directly trained on.

## Purpose

The main purpose of this project was to learn how a pretrained computer vision model can be adapted to a completely different image classification problem.

In particular, the project gave me experience with:

* Loading and preprocessing image datasets in PyTorch
* Working with training and validation datasets
* Using pretrained neural networks
* Applying transfer learning
* Modifying the classifier of an existing CNN
* Training an image classification model
* Evaluating its performance on unseen validation images

It also demonstrates how computer vision can be applied to geological images, where differences in visual appearance can be used to distinguish between different rock classes.

## Running the Project

The project is contained in a Jupyter Notebook.

The main Python libraries used are:

```python
torch
torchvision
numpy
matplotlib
PIL
```

EX: To be classified as **Igneous -> Granite**

<img width="387" height="290" alt="image" src="https://github.com/user-attachments/assets/abb25050-cf83-43f7-a86f-74c7042785c7" />


Open the notebook in Jupyter Notebook, JupyterLab, Google Colab, or another environment that supports `.ipynb` files and run the cells in order.

The rock dataset will also need to be downloaded and placed in the appropriate directory before training.

## References

**Rock Classification Dataset**
Salman Ibne Eunus et al., Kaggle
https://www.kaggle.com/datasets/salmaneunus/rock-classification

**PyTorch Transfer Learning with MobileNetV2 CNN**
Ben Lambda, Kaggle
https://www.kaggle.com/code/benlambda/pytorch-transfer-learning-with-mobilenetv2-cnn

## Notes

This project is primarily intended as a learning exercise in **computer vision and transfer learning**. The model's performance depends on factors such as the available training images, preprocessing choices, training parameters, and how well the dataset represents the rock classes it is expected to classify.
