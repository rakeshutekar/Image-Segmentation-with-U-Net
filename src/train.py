import os

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from load_data import load_data
from unet_model import unet_model

# Set the path to the dataset directory
data_path = '../data/stage1_train'
images, masks = load_data(data_path)

# Split the dataset
train_images, test_images, train_masks, test_masks = train_test_split(images, masks, test_size=0.2, random_state=42)

# Display some samples
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(train_images[i])
    plt.title('Image')
    plt.axis('off')
    plt.subplot(2, 5, i + 6)
    plt.imshow(train_masks[i].squeeze(), cmap='gray')
    plt.title('Mask')
    plt.axis('off')
plt.show()

model = unet_model()
model.summary()

# Train the model
history = model.fit(train_images, train_masks, validation_split=0.1, epochs=20, batch_size=32)

# Save the model
model.save('../results/unet_model.h5')

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('../results/training_logs.png')
plt.show()
