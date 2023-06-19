"""
This script loads the MNIST dataset using TensorFlow Datasets (TFDS) and performs two main transformations:
1. Binary Mask Conversion: Each image in the dataset is thresholded to create a binary mask based on a given threshold value.
2. Bounding Box Generation: The binary masks are used to generate a set of bounding boxes that encapsulate the regions of interest.

Note: The bounding box generation tbd
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def man_fn(data):
    """
    Manipulation function to create a binary mask from the input image.

    Args:
        data: A dictionary containing 'image' and 'label' tensors.

    Returns:
        modified_data: A dictionary with the modified 'image' tensor and the 'label' tensor.
    """
    modified_data = {
        "image": threshold_binary_mask(data['image'], 188),
        "label": data['label']
    }
    return modified_data

def threshold_binary_mask(input_tensor, threshold):
    """
    Generates a binary mask by thresholding the input tensor.

    Args:
        input_tensor: The input image tensor.
        threshold: The threshold value for binarization.

    Returns:
        mask: The binary mask tensor.
    """
    mask = tf.cast(tf.greater(input_tensor, threshold), tf.float32)
    return mask

def map_to_binary(dataset):
    """
    Applies the manipulation function to each element in the dataset.

    Args:
        dataset: The dataset to be mapped.

    Returns:
        binary_dataset: The dataset with binary masks.
    """
    return dataset.map(lambda x: man_fn(x))

def main():
    """
    Main function to load the MNIST dataset and apply binary mask conversion.
    """
    mnist = tfds.load('mnist')
    binary_train = map_to_binary(mnist['train'])

if __name__ == "__main__":
    main()
