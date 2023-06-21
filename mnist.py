"""
This script loads the MNIST dataset using TensorFlow Datasets (TFDS) and performs two main transformations:
1. Binary Mask Conversion: Each image in the dataset is thresholded to create a binary mask based on a given threshold value.
2. Bounding Box Generation: The binary masks are used to generate a set of bounding boxes that encapsulate the regions of interest.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds

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

def map_to_binary(dataset, threshold):
    """
    Applies the manipulation function to each element in the dataset.

    Args:
        dataset: The dataset to be mapped.

    Returns:
        binary_dataset: The dataset with binary masks.
    """

    def man_fn(data):
        return {"image": threshold_binary_mask(data['image'], threshold), "label": data['label']}
    return dataset.map(lambda x: man_fn(x))

def map_to_bbox(dataset):
    def crawl(image):
        max_x, min_x, max_y, min_y = -1, float('inf'), -1, float('inf')
        y_idx = 0
        for y in image:
            for x in y:
                x_idx = 0
                if x == 1:
                    if max_x < x_idx:
                        max_x = x_idx
                    if max_y < y_idx:
                        max_y = y_idx
                    if min_x > x_idx:
                        min_x = x_idx
                    if min_y > y_idx:
                        min_y = y_idx
                x_idx += 1
            y_idx += 1
        return [max_x, min_x, max_y, min_y]
    return dataset.map(lambda x: crawl(x))

def main():
    mnist = tfds.load('mnist')
    binary_train = map_to_binary(mnist['train'], 188)
    bbox_train = map_to_bbox(binary_train)

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    main()
