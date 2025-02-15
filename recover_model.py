import tensorflow as tf
import os

print("✅ Starting model reconstruction...")

# ✅ Manually Define the Model Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (5, 5), strides=2, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (5, 5), strides=2, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (5, 5), strides=2, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Conv2D(256, (4, 4), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(9, activation='softmax')
])

# ✅ Load the Weights
MODEL_H5_PATH = "C:/Users/thanm/Downloads/CombinedDataset/CombinedDataset/tb_detection_web/model_4.h5"

if not os.path.exists(MODEL_H5_PATH):
    print(f"❌ ERROR: The file '{MODEL_H5_PATH}' does not exist.")
    exit(1)

try:
    model.load_weights(MODEL_H5_PATH)
    print("✅ Model weights loaded successfully.")

    # Save the full model so you don't need to do this again
    model.save("full_model.h5")
    print("✅ Full model saved as 'full_model.h5'.")
except Exception as e:
    print(f"❌ Error loading weights: {e}")
    exit(1)
