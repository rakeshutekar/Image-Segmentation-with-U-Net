import os

import cv2
import numpy as np


def load_data(data_path):
    image_ids = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    images = []
    masks = []

    for image_id in image_ids:
        image_folder = os.path.join(data_path, image_id, 'images')
        mask_folder = os.path.join(data_path, image_id, 'masks')
        
        image_path = os.path.join(image_folder, os.listdir(image_folder)[0])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))

        mask = np.zeros((128, 128), dtype=np.uint8)
        for mask_file in os.listdir(mask_folder):
            mask_path = os.path.join(mask_folder, mask_file)
            mask_ = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_ = cv2.resize(mask_, (128, 128))
            mask = np.maximum(mask, mask_)

        images.append(image)
        masks.append(mask)

    images = np.array(images) / 255.0
    masks = np.array(masks) / 255.0
    masks = np.expand_dims(masks, axis=-1)

    return images, masks
