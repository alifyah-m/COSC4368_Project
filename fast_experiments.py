import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import class_weight

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Use MobileNetV2 which is smaller and faster than EfficientNet
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# --- Configuration ---
IMG_SIZE = (160, 160)  # Smaller image size for faster processing
BATCH_SZ = 32
EPOCHS = 10           # Reduced epochs
SEED = 42
DATA_DIR = "data/raw"
MAX_SAMPLES = 1000    # Limit the number of samples to speed up training

# --- Data Loading & Generators ---
def prepare_dataframe(csv_path, img_folder, max_samples=MAX_SAMPLES):
    df = pd.read_csv(csv_path)
    df['filepath'] = df['X_ray_image_name'].apply(lambda x: os.path.join(img_folder, x))
    
    # Limit the number of samples for faster processing
    if max_samples and len(df) > max_samples:
        # Ensure balanced sampling
        df_0 = df[df['Label'] == 0].sample(min(max_samples // 2, len(df[df['Label'] == 0])), random_state=SEED)
        df_1 = df[df['Label'] == 1].sample(min(max_samples // 2, len(df[df['Label'] == 1])), random_state=SEED)
        df = pd.concat([df_0, df_1])
    
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)

def make_generators(df_train, df_val):
    print("Setting up data generators...")
    
    # Calculate class weights for imbalanced dataset
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(df_train['Label']),
        y=df_train['Label']
    )
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"Class weights: {class_weights_dict}")
    
    # Use data augmentation for training
    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    val_aug = ImageDataGenerator(rescale=1./255)
    
    print("Creating training generator...")
    train_gen = train_aug.flow_from_dataframe(
        df_train, x_col='filepath', y_col='Label',
        target_size=IMG_SIZE, class_mode='raw',
        batch_size=BATCH_SZ, shuffle=True, seed=SEED
    )
    
    print("Creating validation generator...")
    val_gen = val_aug.flow_from_dataframe(
        df_val, x_col='filepath', y_col='Label',
        target_size=IMG_SIZE, class_mode='raw',
        batch_size=BATCH_SZ, shuffle=False
    )
    print("Generators created successfully")
    
    return train_gen, val_gen, class_weights_dict

# --- Model Definition ---
def build_classifier(unfreeze_layers=10):
    print("Building MobileNetV2 classifier...")
    base_net = MobileNetV2(
        weights='imagenet', include_top=False,
        input_shape=IMG_SIZE + (3,)
    )
    print("Base model loaded")
    
    # Initially freeze all layers
    for layer in base_net.layers:
        layer.trainable = False
    
    # Unfreeze the last n layers for fine-tuning
    if unfreeze_layers > 0:
        for layer in base_net.layers[-unfreeze_layers:]:
            layer.trainable = True
        print(f"Unfroze last {unfreeze_layers} layers of base model for fine-tuning")
    
    x = GlobalAveragePooling2D()(base_net.output)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_net.input, outputs=output)
    print("Model built successfully")
    return model

# --- Training ---
def train_model(model, train_gen, val_gen, class_weights=None):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]
    start = time.time()
    
    # Use class weights if provided
    if class_weights:
        print(f"Training with class weights: {class_weights}")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1,
            class_weight=class_weights
        )
    else:
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
    
    train_time = time.time() - start
    print(f"Training completed in {train_time:.1f} seconds")
    return model, history, train_time

# --- Evaluation ---
def evaluate_model(model, val_gen):
    # Make predictions
    y_pred = model.predict(val_gen)
    y_true = np.array(val_gen.labels).astype(int)  # Use labels instead of classes
    
    # Find optimal threshold
    thresholds = np.arange(0.3, 0.7, 0.05)  # Reduced range for faster processing
    f1_scores = [f1_score(y_true, (y_pred > t).astype(int)) for t in thresholds]
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold:.2f}")
    
    # Apply optimal threshold
    preds = (y_pred > optimal_threshold).astype(int).flatten()
    
    # Print evaluation metrics
    print("\nEvaluation Results:")
    # Define class names manually since we're using raw class mode
    class_names = {0: 'NORMAL', 1: 'COVID'}
    print(classification_report(y_true, preds, target_names=class_names.values()))
    
    # Print confusion matrix
    cm = confusion_matrix(y_true, preds)
    print("Confusion Matrix:")
    print(cm)
    
    return optimal_threshold, preds

# --- Main Execution ---
if __name__ == '__main__':
    print("Starting fast COVID X-ray classification...")
    
    # Prepare data
    csv_path = os.path.join(DATA_DIR, 'train_labels.csv')
    print(f"Loading data from {csv_path}")
    df = prepare_dataframe(csv_path, os.path.join(DATA_DIR, 'train'))
    print(f"Using {len(df)} samples for faster processing")
    
    # Print class distribution
    print("Class distribution:")
    print(df['Label'].value_counts())
    print(f"Class 0: {df['Label'].value_counts()[0] / len(df):.2%}")
    print(f"Class 1: {df['Label'].value_counts()[1] / len(df):.2%}")
    
    # Split data
    mask = np.random.RandomState(SEED).rand(len(df)) < 0.8  # 80/20 split for faster training
    df_train, df_val = df[mask], df[~mask]
    print(f"Split data into {len(df_train)} training and {len(df_val)} validation samples")
    
    # Create data generators
    train_gen, val_gen, class_weights = make_generators(df_train, df_val)
    
    # Build and train model
    model = build_classifier(unfreeze_layers=10)
    model, history, train_time = train_model(model, train_gen, val_gen, class_weights=class_weights)
    
    # Evaluate model
    optimal_threshold, preds = evaluate_model(model, val_gen)
    
    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('fast_training_results.png')
    
    print(f"\nTotal execution time: {train_time:.1f} seconds")
    print("Model training and evaluation completed!")
    print("Results saved to fast_training_results.png")