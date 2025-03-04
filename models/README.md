# Emotion Detection Models

This directory is where pre-trained emotion detection models should be placed.

## Expected Models

For full emotion detection capabilities, place the following model in this directory:

- `emotion_model.h5` - A TensorFlow/Keras model trained to detect emotions from facial images

## How to Get Models

If you don't have access to pre-trained emotion models, you have several options:

1. **Use a pre-trained model**: Download a pre-trained emotion recognition model like FER2013 or EmotioNet model.
   
2. **Train your own model**: Use datasets like FER2013, AffectNet, or EmotioNet to train a custom model.

3. **Use a model from Hugging Face**: The Hugging Face model hub has several emotion detection models you can use.

## Example: Download a pre-trained model

Here's a simple Python script to download and save a pre-trained emotion detection model:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

# Create a simple CNN model architecture
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  # 7 emotions

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Save the model architecture
model.save('emotion_model.h5')

print("Model saved. Note: this is just the architecture without training.")
```

## Note

If no pre-trained model is found, the system will fall back to using a simpler OpenCV-based detection approach using Haar cascades.
