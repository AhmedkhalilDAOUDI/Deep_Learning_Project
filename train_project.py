"""
Dogs vs Cats - Full training pipeline (TF 2.10 / GPU)
Run from VS Code with the tf_gpu-test interpreter selected.
"""
import os, sys, time, json, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, GlobalAveragePooling2D)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

print(f"TensorFlow {tf.__version__}")
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

# ── Paths ──────────────────────────────────────────────────────────────────
base_dir = r"C:\Users\Khalil\Desktop\Deep Learning project\final_project\final_project\data_cats_and_dogs"
train_path = os.path.join(base_dir, "train")
val_path = os.path.join(base_dir, "validation")
save_dir = os.path.join(base_dir, "saved_model")
os.makedirs(save_dir, exist_ok=True)

# ── Data ───────────────────────────────────────────────────────────────────
IMG_SIZE = (160, 160)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
    val_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False
)
print(f"Train: {train_generator.samples} images, classes: {train_generator.class_indices}")
print(f"Val: {val_generator.samples} images, classes: {val_generator.class_indices}")

# ── Build helpers ───────────────────────────────────────────────────────────
def build_cnn():
    model = Sequential([
        Input(shape=IMG_SIZE + (3,)),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.25),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.3),
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_aug_cnn():
    model = Sequential([
        Input(shape=IMG_SIZE + (3,)),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.25),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.3),
        Conv2D(256, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.35),
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ── Training helper ─────────────────────────────────────────────────────────
def train_model(model, train_gen, val_gen, epochs=40, patience=6):
    cb = EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True, verbose=1)
    t0 = time.perf_counter()
    with tf.device('/GPU:0'):
        hist = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[cb], verbose=2)
    elapsed = time.perf_counter() - t0
    print(f"  Training done in {elapsed:.1f}s ({len(hist.history['loss'])} epochs)")
    return hist

# ── Model 1 ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("MODEL 1: CNN From Scratch")
print("="*60)
model1 = build_cnn()
model1.summary()
hist1 = train_model(model1, train_generator, val_generator)
val_loss1, val_acc1 = model1.evaluate(val_generator, verbose=0)
print(f"Model 1 val_accuracy: {val_acc1:.4f}")

# ── Model 2 ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("MODEL 2: Augmented CNN")
print("="*60)
aug_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.18,
    height_shift_range=0.18,
    shear_range=0.18,
    zoom_range=0.25,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    fill_mode='nearest'
)
aug_train = aug_datagen.flow_from_directory(train_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
model2 = build_aug_cnn()
hist2 = train_model(model2, aug_train, val_generator)
val_loss2, val_acc2 = model2.evaluate(val_generator, verbose=0)
print(f"Model 2 val_accuracy: {val_acc2:.4f}")

# ── Model 3 ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("MODEL 3: Transfer Learning (MobileNetV2)")
print("="*60)
base_model = MobileNetV2(input_shape=IMG_SIZE+(3,), include_top=False, weights='imagenet')
base_model.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)
model3 = Model(inputs=base_model.input, outputs=output)
model3.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model3.summary()
hist3 = train_model(model3, train_generator, val_generator)
val_loss3, val_acc3 = model3.evaluate(val_generator, verbose=0)
print(f"Model 3 val_accuracy: {val_acc3:.4f}")

# ── Pick best ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RESULTS")
print("="*60)
all_accs = {'CNN_Scratch': val_acc1, 'Augmented_CNN': val_acc2, 'TransferLearning': val_acc3}
for k, v in all_accs.items():
    print(f"  {k}: {v:.4f}")
best_name = max(all_accs, key=all_accs.get)
print(f"\nBest: {best_name} ({all_accs[best_name]:.4f})")

best_model = {'CNN_Scratch': model1, 'Augmented_CNN': model2, 'TransferLearning': model3}[best_name]
best_model.save(save_dir)
print(f"Saved to: {save_dir}")

# ── Plots ───────────────────────────────────────────────────────────────────
def save_plots(hist, name):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(hist.history['loss'], label='train')
    ax[0].plot(hist.history['val_loss'], label='val')
    ax[0].set_title(f'{name} - Loss'); ax[0].legend()
    ax[1].plot(hist.history['accuracy'], label='train')
    ax[1].plot(hist.history['val_accuracy'], label='val')
    ax[1].set_title(f'{name} - Accuracy'); ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f'{name}_history.png'), dpi=150)
    plt.close()

save_plots(hist1, 'model1_cnn')
save_plots(hist2, 'model2_aug')
save_plots(hist3, 'model3_tl')

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(list(all_accs.keys()), list(all_accs.values()), color=['#3498db','#e74c3c','#2ecc71'])
ax.set_ylim(0, 1)
for bar, v in zip(bars, list(all_accs.values())):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{v:.4f}', ha='center')
ax.set_title('Model Comparison'); ax.set_ylabel('Val Accuracy')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'model_comparison.png'), dpi=150)
plt.close()

# ── Evaluation ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("DETAILED EVALUATION")
print("="*60)
val_generator.reset()
y_pred = (best_model.predict(val_generator, verbose=1) > 0.5).astype(int).flatten()
y_test = val_generator.classes

print(classification_report(y_test, y_pred, target_names=list(train_generator.class_indices.keys())))
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(train_generator.class_indices.keys()),
            yticklabels=list(train_generator.class_indices.keys()), ax=ax)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'confusion_matrix.png'), dpi=150)
plt.close()
print(f"Accuracy: {(y_pred == y_test).mean():.4f}")

# ── Error analysis ──────────────────────────────────────────────────────────
errors = np.where(y_pred != y_test)[0]
print(f"\nErrors: {len(errors)} / {len(y_test)} ({len(errors)/len(y_test)*100:.1f}%)")
if len(errors):
    sample = errors[:9]
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()
    fps = np.array(val_generator.filepaths)
    for i, idx in enumerate(sample):
        axes[i].imshow(Image.open(fps[idx]))
        tl = list(train_generator.class_indices.keys())[y_test[idx]]
        pl = list(train_generator.class_indices.keys())[y_pred[idx]]
        axes[i].set_title(f'True: {tl} | Pred: {pl}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'error_samples.png'), dpi=150)
    plt.close()

# ── Results JSON ────────────────────────────────────────────────────────────
results = {
    "model1_cnn": {"val_accuracy": float(val_acc1), "val_loss": float(val_loss1), "epochs": len(hist1.history['loss'])},
    "model2_aug": {"val_accuracy": float(val_acc2), "val_loss": float(val_loss2), "epochs": len(hist2.history['loss'])},
    "model3_tl": {"val_accuracy": float(val_acc3), "val_loss": float(val_loss3), "epochs": len(hist3.history['loss'])},
    "best_model": best_name,
    "best_accuracy": float(all_accs[best_name]),
    "params_m1": int(model1.count_params()),
    "params_m2": int(model2.count_params()),
    "params_m3": int(model3.count_params()),
}
with open(os.path.join(base_dir, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nAll done! results.json saved.")
