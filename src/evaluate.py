import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from load_data import load_data
from unet_model import unet_model

# Set the path to the dataset directory
data_path = '../data/stage1_train'
images, masks = load_data(data_path)

# Split the dataset
_, test_images, _, test_masks = train_test_split(images, masks, test_size=0.2, random_state=42)

# Load the saved model
model = tf.keras.models.load_model('../results/unet_model.h5')

# Evaluate the model
evaluation = model.evaluate(test_images, test_masks)
print(f"Test Loss: {evaluation[0]}")
print(f"Test Accuracy: {evaluation[1]}")

# Predict on test data
pred_masks = model.predict(test_images)

# Function to calculate IoU (Intersection over Union)
def iou_metric(y_true, y_pred):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    
    intersection = np.sum((y_true == 1) & (y_pred == 1))
    union = np.sum((y_true == 1) | (y_pred == 1))
    
    iou = intersection / union if union != 0 else 0
    return iou

# Calculate IoU for the test set
ious = [iou_metric(test_masks[i], pred_masks[i] > 0.5) for i in range(len(test_masks))]
mean_iou = np.mean(ious)
print(f"Mean IoU: {mean_iou}")

# Display some predictions
plt.figure(figsize=(10, 10))
for i in range(5):
    plt.subplot(3, 5, i + 1)
    plt.imshow(test_images[i])
    plt.title('Test Image')
    plt.axis('off')
    plt.subplot(3, 5, i + 6)
    plt.imshow(test_masks[i].squeeze(), cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')
    plt.subplot(3, 5, i + 11)
    plt.imshow(pred_masks[i].squeeze() > 0.5, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')
plt.show()
