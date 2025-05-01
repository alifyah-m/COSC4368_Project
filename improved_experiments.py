import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import class_weight

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Import EfficientNetB0 (smaller model) from keras.applications instead of EfficientNetB7
from keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout,
    Conv2D, MaxPooling2D
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

# --- Configuration ---
IMG_SIZE = (224, 224)
BATCH_SZ = 32
EPOCHS = 30        # increased epochs to allow more training time
SEED = 42
DATA_DIR = "data/raw"
UNFREEZE_LAYERS = 20  # Number of layers to unfreeze from the top of the base model

# --- Data Loading & Generators ---
def prepare_dataframe(csv_path, img_folder):
    df = pd.read_csv(csv_path)
    df['filepath'] = df['X_ray_image_name'].apply(lambda x: os.path.join(img_folder, x))
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)

def make_generators(df_train, df_val, balance_classes=True):
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
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    val_aug = ImageDataGenerator(rescale=1./255)

    # If balancing classes, use oversampling for minority class
    if balance_classes:
        # Separate majority and minority classes
        df_majority = df_train[df_train['Label'] == 1]
        df_minority = df_train[df_train['Label'] == 0]
        
        # Oversample minority class
        minority_upsampled = df_minority.sample(
            n=len(df_majority), 
            replace=True, 
            random_state=SEED
        )
        
        # Combine majority and upsampled minority
        df_train_balanced = pd.concat([df_majority, minority_upsampled])
        df_train_balanced = df_train_balanced.sample(frac=1, random_state=SEED).reset_index(drop=True)
        print(f"Balanced training data: {len(df_train_balanced)} samples")
        print(f"Class distribution: {df_train_balanced['Label'].value_counts()}")
        
        df_train_to_use = df_train_balanced
    else:
        df_train_to_use = df_train
    
    print("Creating training generator...")
    train_gen = train_aug.flow_from_dataframe(
        df_train_to_use, x_col='filepath', y_col='Label',
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

def build_classifier(dense_units=64, dropout_rate=0.5, unfreeze_layers=UNFREEZE_LAYERS):
    print(f"Building EfficientNetB0 classifier with {dense_units} dense units...")
    base_net = EfficientNetB0(
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
    else:
        print("All base model layers are frozen")

    x = GlobalAveragePooling2D()(base_net.output)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_net.input, outputs=output)
    print("Model built successfully")
    return model

# --- Training, Diagnosis & Complexity ---
def compile_and_train(model, train_gen, val_gen, class_weights=None, learning_rate=1e-4):
    # Define custom F1 score metric
    def f1_metric(y_true, y_pred):
        y_pred_binary = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)
        precision = tf.keras.metrics.Precision()(y_true, y_pred_binary)
        recall = tf.keras.metrics.Recall()(y_true, y_pred_binary)
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy', 
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            f1_metric
        ]
    )
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
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
                for unfreeze in [0, 10, 20]:  # Test different numbers of unfrozen layers
                    print(f"Running ablation: frac={frac}, dense={dense}, lr={lr}, unfreeze={unfreeze}")
                    train_gen, val_gen, class_weights = make_generators(df_sub, df_val, balance_classes=True)
                    model = build_classifier(dense_units=dense, unfreeze_layers=unfreeze)
                    history, _ = compile_and_train(model, train_gen, val_gen, class_weights=class_weights, learning_rate=lr)
                    
                    # Make predictions with optimal threshold
                    y_pred = model.predict(val_gen)
                    y_true = val_gen.classes
                    
                    # Find optimal threshold
                    thresholds = np.arange(0.1, 0.9, 0.05)
                    f1_scores = [f1_score(y_true, (y_pred > t).astype(int)) for t in thresholds]
                    optimal_idx = np.argmax(f1_scores)
                    optimal_threshold = thresholds[optimal_idx]
                    optimal_f1 = f1_scores[optimal_idx]
                    
                    # Calculate metrics with optimal threshold
                    y_pred_binary = (y_pred > optimal_threshold).astype(int)
                    accuracy = np.mean(y_true == y_pred_binary)
                    precision = precision_score(y_true, y_pred_binary)
                    recall = recall_score(y_true, y_pred_binary)
                    auc = roc_auc_score(y_true, y_pred)
                    
                    results.append({
                        'method': 'EffNetB0',
                        'data_frac': frac,
                        'dense_units': dense,
                        'learning_rate': lr,
                        'unfrozen_layers': unfreeze,
                        'optimal_threshold': optimal_threshold,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': optimal_f1,
                        'auc': auc
                    })
    return pd.DataFrame(results)

# --- Method Comparison ---
def compare_models(df_train, df_val):
    print("\n=== Comparing BaselineCNN vs EfficientNetB0 ===")
    models = {
        'BaselineCNN': build_baseline_cnn(),
        'EffNetB0': build_classifier(unfreeze_layers=20)  # Use unfrozen layers for EfficientNetB0
    }
    summary = []
    
    # Get class weights
    _, _, class_weights = make_generators(df_train, df_val, balance_classes=True)
    
    for name, mdl in models.items():
        print(f"\n-> Training {name}")
        train_gen, val_gen, _ = make_generators(df_train, df_val, balance_classes=True)
        history, t_time = compile_and_train(mdl, train_gen, val_gen, class_weights=class_weights)
        
        # Make predictions
        y_pred = mdl.predict(val_gen)
        y_true = val_gen.classes
        
        # Find optimal threshold
        thresholds = np.arange(0.1, 0.9, 0.05)
        f1_scores = [f1_score(y_true, (y_pred > t).astype(int)) for t in thresholds]
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Calculate metrics with optimal threshold
        y_pred_binary = (y_pred > optimal_threshold).astype(int)
        accuracy = np.mean(y_true == y_pred_binary)
        precision = precision_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary)
        auc = roc_auc_score(y_true, y_pred)
        
        summary.append({
            'model': name,
            'optimal_threshold': optimal_threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'train_time': t_time
        })
    
    df = pd.DataFrame(summary).set_index('model')
    print("\nComparison Results:\n", df)
    best = df['f1_score'].idxmax()
    print(f"\nConclusion: '{best}' performs best based on F1 score.")
    return df

# --- Plotting Helpers ---
def plot_loss_trajectory(history):
    plt.figure(figsize=(12, 8))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    xs = range(len(history.history['loss']))
    plt.plot(xs, history.history['loss'], label='Train Loss')
    plt.plot(xs, history.history['val_loss'], label='Val Loss')
    plt.title('Loss Trajectory')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(xs, history.history['accuracy'], label='Train Accuracy')
    plt.plot(xs, history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy Trajectory')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot AUC
    plt.subplot(2, 2, 3)
    plt.plot(xs, history.history['auc'], label='Train AUC')
    plt.plot(xs, history.history['val_auc'], label='Val AUC')
    plt.title('AUC Trajectory')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    
    # Plot F1 Score
    plt.subplot(2, 2, 4)
    plt.plot(xs, history.history['f1_metric'], label='Train F1')
    plt.plot(xs, history.history['val_f1_metric'], label='Val F1')
    plt.title('F1 Score Trajectory')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

def plot_ablation(ab_df):
    plt.figure(figsize=(15, 10))
    
    # Plot F1 score vs dense units
    plt.subplot(2, 2, 1)
    for frac in sorted(ab_df['data_frac'].unique()):
        for unfreeze in sorted(ab_df['unfrozen_layers'].unique()):
            sub = ab_df[(ab_df['data_frac'] == frac) & (ab_df['unfrozen_layers'] == unfreeze)]
            plt.plot(sub['dense_units'], sub['f1_score'], marker='o', 
                     label=f"Frac {frac}, Unfreeze {unfreeze}")
    plt.title('Ablation: Dense Width vs F1 Score')
    plt.xlabel('Dense Units')
    plt.ylabel('F1 Score')
    plt.legend()
    
    # Plot F1 score vs unfrozen layers
    plt.subplot(2, 2, 2)
    for frac in sorted(ab_df['data_frac'].unique()):
        for dense in sorted(ab_df['dense_units'].unique()):
            sub = ab_df[(ab_df['data_frac'] == frac) & (ab_df['dense_units'] == dense)]
            plt.plot(sub['unfrozen_layers'], sub['f1_score'], marker='o', 
                     label=f"Frac {frac}, Dense {dense}")
    plt.title('Ablation: Unfrozen Layers vs F1 Score')
    plt.xlabel('Unfrozen Layers')
    plt.ylabel('F1 Score')
    plt.legend()
    
    # Plot precision vs recall
    plt.subplot(2, 2, 3)
    plt.scatter(ab_df['recall'], ab_df['precision'], c=ab_df['f1_score'], cmap='viridis')
    plt.colorbar(label='F1 Score')
    plt.title('Precision vs Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    # Plot optimal threshold distribution
    plt.subplot(2, 2, 4)
    plt.hist(ab_df['optimal_threshold'], bins=10)
    plt.title('Distribution of Optimal Thresholds')
    plt.xlabel('Threshold')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('ablation_results.png')
    plt.show()

# --- Main Execution ---
if __name__ == '__main__':
    # Prepare data
    print("Starting script execution...")
    csv_path = os.path.join(DATA_DIR, 'train_labels.csv')
    print(f"Loading data from {csv_path}")
    df = prepare_dataframe(csv_path, os.path.join(DATA_DIR, 'train'))
    print(f"Loaded {len(df)} records from CSV")
    
    # Print class distribution
    print("Class distribution in full dataset:")
    print(df['Label'].value_counts())
    print(f"Class 0: {df['Label'].value_counts()[0] / len(df):.2%}")
    print(f"Class 1: {df['Label'].value_counts()[1] / len(df):.2%}")
    
    mask = np.random.RandomState(SEED).rand(len(df)) < 0.9
    df_train, df_val = df[mask], df[~mask]
    print(f"Split data into {len(df_train)} training and {len(df_val)} validation samples")
    
    # Print class distribution in validation set
    print("Class distribution in validation set:")
    print(df_val['Label'].value_counts())
    print(f"Class 0: {df_val['Label'].value_counts()[0] / len(df_val):.2%}")
    print(f"Class 1: {df_val['Label'].value_counts()[1] / len(df_val):.2%}")

    # Ablation study
    print("Starting ablation study...")
    ab_df = run_ablation(df_train, df_val)
    print("\nAblation Results:\n", ab_df)
    plot_ablation(ab_df)
    
    # Find best configuration
    best_config = ab_df.loc[ab_df['f1_score'].idxmax()]
    print(f"\nBest configuration based on F1 score: {best_config.to_dict()}")

    # Final training & complexity with best configuration
    train_gen, val_gen, class_weights = make_generators(df_train, df_val, balance_classes=True)
    final_model = build_classifier(
        dense_units=int(best_config['dense_units']), 
        unfreeze_layers=int(best_config['unfrozen_layers'])
    )
    history, train_time = compile_and_train(
        final_model, 
        train_gen, 
        val_gen, 
        class_weights=class_weights,
        learning_rate=best_config['learning_rate']
    )
    diagnose_fit_issues(history)
    compute_complexity(final_model, train_time, val_gen)
    plot_loss_trajectory(history)

    # Compare methods
    compare_models(df_train, df_val)

    # Final evaluation with optimal threshold
    y_pred = final_model.predict(val_gen)
    y_true = val_gen.classes
    
    # Find optimal threshold
    thresholds = np.arange(0.1, 0.9, 0.05)
    f1_scores = [f1_score(y_true, (y_pred > t).astype(int)) for t in thresholds]
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    print(f"\nOptimal threshold: {optimal_threshold:.2f}")
    
    # Apply optimal threshold
    preds = (y_pred > optimal_threshold).astype(int).flatten()
    
    print("\nFinal Evaluation:")
    print(classification_report(y_true, preds, target_names=val_gen.class_indices.keys()))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(val_gen.class_indices.keys()))
    plt.xticks(tick_marks, val_gen.class_indices.keys(), rotation=45)
    plt.yticks(tick_marks, val_gen.class_indices.keys())
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Save the best model
    final_model.save('best_covid_xray_model.h5')
    print("Best model saved as 'best_covid_xray_model.h5'")