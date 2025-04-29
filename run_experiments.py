import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Import EfficientNetB0 (smaller model) from keras.applications instead of EfficientNetB7
from keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout,
    Conv2D, MaxPooling2D
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# --- Configuration ---
IMG_SIZE = (224, 224)
BATCH_SZ = 32
EPOCHS = 20        # you can raise this later
SEED = 42
DATA_DIR = "data/raw"

# --- Data Loading & Generators ---
def prepare_dataframe(csv_path, img_folder):
    df = pd.read_csv(csv_path)
    df['filepath'] = df['X_ray_image_name'].apply(lambda x: os.path.join(img_folder, x))
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)

def make_generators(df_train, df_val):
    print("Setting up data generators...")
    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    val_aug = ImageDataGenerator(rescale=1./255)

    print("Creating training generator...")
    # Use class_mode="raw" to allow numeric labels
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
    return train_gen, val_gen

# --- Model Definitions ---
def build_baseline_cnn():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=IMG_SIZE+(3,)),
        MaxPooling2D(),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(),
        GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_classifier(dense_units=64, dropout_rate=0.5):
    print(f"Building EfficientNetB0 classifier with {dense_units} dense units...")
    base_net = EfficientNetB0(
        weights='imagenet', include_top=False,
        input_shape=IMG_SIZE + (3,)
    )
    print("Base model loaded")
    for layer in base_net.layers:
        layer.trainable = False
    print("Set base model layers to non-trainable")

    x = GlobalAveragePooling2D()(base_net.output)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_net.input, outputs=output)
    print("Model built successfully")
    return model

# --- Training, Diagnosis & Complexity ---
def compile_and_train(model, train_gen, val_gen, learning_rate=1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    ]
    start = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    train_time = time.time() - start
    return history, train_time

def diagnose_fit_issues(history):
    tl, vl = history.history['loss'], history.history['val_loss']
    if vl[-1] > min(vl):
        print("⚠️ Overfitting: validation loss rising while train loss falls.")
    elif tl[-1] > vl[-1]:
        print("⚠️ Underfitting: train loss remains higher than validation.")
    else:
        print("✅ Fit looks balanced.")

def compute_complexity(model, train_time, val_gen):
    total = model.count_params()
    trainable = np.sum([np.prod(w.shape) for w in model.trainable_weights])
    size_mb = total * 4 / 1024**2  # float32
    print(f"Params: total={total:,}, trainable={trainable:,}, size~{size_mb:.2f}MB")
    print(f"Train time: {train_time:.1f}s")
    start = time.time()
    _ = model.predict(val_gen, verbose=0)
    print(f"Test time: {time.time() - start:.1f}s")

# --- Ablation Study ---
def run_ablation(df_train, df_val):
    results = []
    for frac in [0.5, 1.0]:
        df_sub = df_train.sample(frac=frac, random_state=SEED)
        for dense in [32, 64, 128]:
            for lr in [1e-4, 5e-5]:
                print(f"Running ablation: frac={frac}, dense={dense}, lr={lr}")
                train_gen, val_gen = make_generators(df_sub, df_val)
                model = build_classifier(dense_units=dense)
                history, _ = compile_and_train(model, train_gen, val_gen, learning_rate=lr)
                results.append({
                    'method': 'EffNetB0',
                    'data_frac': frac,
                    'dense_units': dense,
                    'learning_rate': lr,
                    'val_acc': max(history.history['val_accuracy'])
                })
    return pd.DataFrame(results)

# --- Method Comparison ---
def compare_models(df_train, df_val):
    print("\n=== Comparing BaselineCNN vs EfficientNetB0 ===")
    models = {
        'BaselineCNN': build_baseline_cnn(),
        'EffNetB0': build_classifier()
    }
    summary = []
    for name, mdl in models.items():
        print(f"\n-> Training {name}")
        train_gen, val_gen = make_generators(df_train, df_val)
        history, t_time = compile_and_train(mdl, train_gen, val_gen)
        preds = (mdl.predict(val_gen) > 0.5).astype(int).flatten()
        report = classification_report(val_gen.classes, preds, output_dict=True)
        summary.append({
            'model': name,
            'val_acc': report['accuracy'],
            'val_auc': max(history.history.get('auc', [])),
            'train_time': t_time
        })
    df = pd.DataFrame(summary).set_index('model')
    print("\nComparison Results:\n", df)
    best = df['val_acc'].idxmax()
    print(f"\nConclusion: '{best}' performs best due to pretrained feature leverage.")
    return df

# --- Plotting Helpers ---
def plot_loss_trajectory(history):
    plt.figure(figsize=(8, 5))
    xs = range(len(history.history['loss']))
    plt.plot(xs, history.history['loss'], label='Train Loss')
    plt.plot(xs, history.history['val_loss'], label='Val Loss')
    plt.title('Loss Trajectory')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_ablation(ab_df):
    plt.figure(figsize=(8, 5))
    for frac in sorted(ab_df['data_frac'].unique()):
        sub = ab_df[ab_df['data_frac'] == frac]
        plt.plot(sub['dense_units'], sub['val_acc'], marker='o', label=f"Frac {frac}")
    plt.title('Ablation: Dense Width vs Val Acc')
    plt.xlabel('Dense Units')
    plt.ylabel('Val Accuracy')
    plt.legend()
    plt.show()

# --- Main Execution ---
if __name__ == '__main__':
    # Prepare data
    print("Starting script execution...")
    csv_path = os.path.join(DATA_DIR, 'train_labels.csv')
    print(f"Loading data from {csv_path}")
    df = prepare_dataframe(csv_path, os.path.join(DATA_DIR, 'train'))
    print(f"Loaded {len(df)} records from CSV")
    mask = np.random.RandomState(SEED).rand(len(df)) < 0.9
    df_train, df_val = df[mask], df[~mask]
    print(f"Split data into {len(df_train)} training and {len(df_val)} validation samples")

    # Ablation study
    print("Starting ablation study...")
    ab_df = run_ablation(df_train, df_val)
    print("\nAblation Results:\n", ab_df)
    plot_ablation(ab_df)

    # Final training & complexity
    train_gen, val_gen = make_generators(df_train, df_val)
    final_model = build_classifier()
    history, train_time = compile_and_train(final_model, train_gen, val_gen)
    diagnose_fit_issues(history)
    compute_complexity(final_model, train_time, val_gen)

    # Compare methods
    compare_models(df_train, df_val)

    # Final evaluation
    preds = (final_model.predict(val_gen) > 0.5).astype(int).flatten()
    print("\nFinal Evaluation:")
    print(classification_report(val_gen.classes, preds, target_names=val_gen.class_indices.keys()))
    print(confusion_matrix(val_gen.classes, preds))
