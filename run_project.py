"""
Dogs vs Cats Classification Project
Trains 3 models: CNN scratch, Augmented CNN, Transfer Learning (MobileNetV2)
Evaluates and saves the best model + Streamlit app
"""
import os, sys, time, json, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
print(f"TensorFlow {tf.__version__}")

sns.set_style("whitegrid")

# ── Paths ──────────────────────────────────────────────────────────────────
base_dir = r"C:\Users\Khalil\Desktop\Deep Learning project\final_project\final_project\data_cats_and_dogs"
train_path = os.path.join(base_dir, "train")
val_path = os.path.join(base_dir, "validation")
save_dir = os.path.join(base_dir, "saved_model")
os.makedirs(save_dir, exist_ok=True)

# ── Generators ─────────────────────────────────────────────────────────────
IMG_SIZE = (160, 160)
BATCH_SIZE = 32

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
    val_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False
)
print(f"Train: {train_generator.samples} images, classes: {train_generator.class_indices}")
print(f"Val: {val_generator.samples} images, classes: {val_generator.class_indices}")

# ── Helper: training ────────────────────────────────────────────────────────
def train_model(model, train_gen, val_gen, epochs=40, patience=6):
    cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=patience, restore_best_weights=True, verbose=1
    )
    t0 = time.perf_counter()
    hist = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[cb], verbose=2)
    elapsed = time.perf_counter() - t0
    print(f"Training done in {elapsed:.1f}s ({len(hist.history['loss'])} epochs)")
    return hist

# ── Model 1: CNN from scratch ───────────────────────────────────────────────
print("\n" + "="*60)
print("MODEL 1: CNN From Scratch")
print("="*60)

input_shape = IMG_SIZE + (3,)

def build_cnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model1 = build_cnn()
model1.summary()
hist1 = train_model(model1, train_generator, val_generator)
val_loss1, val_acc1 = model1.evaluate(val_generator, verbose=0)
print(f"Model 1 val_accuracy: {val_acc1:.4f}")

# ── Model 2: Augmented CNN ──────────────────────────────────────────────────
print("\n" + "="*60)
print("MODEL 2: Augmented CNN")
print("="*60)

aug_train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
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
aug_train_generator = aug_train_datagen.flow_from_directory(
    train_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)

def build_cnn_aug():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model2 = build_cnn_aug()
hist2 = train_model(model2, aug_train_generator, val_generator)
val_loss2, val_acc2 = model2.evaluate(val_generator, verbose=0)
print(f"Model 2 val_accuracy: {val_acc2:.4f}")

# ── Model 3: Transfer Learning (MobileNetV2) ────────────────────────────────
print("\n" + "="*60)
print("MODEL 3: Transfer Learning (MobileNetV2)")
print("="*60)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=input_shape, include_top=False, weights='imagenet'
)
base_model.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model3 = tf.keras.Model(inputs=base_model.input, outputs=output)
model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model3.summary()

hist3 = train_model(model3, train_generator, val_generator)
val_loss3, val_acc3 = model3.evaluate(val_generator, verbose=0)
print(f"Model 3 val_accuracy: {val_acc3:.4f}")

# ── Comparison ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RESULTS COMPARISON")
print("="*60)
print(f"  CNN Scratch:           {val_acc1:.4f}")
print(f"  Augmented CNN:         {val_acc2:.4f}")
print(f"  Transfer Learning:     {val_acc3:.4f}")

accs = {'CNN_Scratch': val_acc1, 'Augmented_CNN': val_acc2, 'TransferLearning': val_acc3}
best_name = max(accs, key=accs.get)
print(f"\nBest model: {best_name} ({accs[best_name]:.4f})")

best_map = {'CNN_Scratch': model1, 'Augmented_CNN': model2, 'TransferLearning': model3}
best_model = best_map[best_name]

# Save best model
best_model.save(save_dir)
print(f"Saved best model to: {save_dir}")

# ── Plots ───────────────────────────────────────────────────────────────────
def save_history_plot(hist_dict, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist_dict['loss'], label='train')
    axes[0].plot(hist_dict['val_loss'], label='val')
    axes[0].set_title(f'{name} - Loss')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].legend()
    axes[1].plot(hist_dict['accuracy'], label='train')
    axes[1].plot(hist_dict['val_accuracy'], label='val')
    axes[1].set_title(f'{name} - Accuracy')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('accuracy')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f'{name}_history.png'), dpi=150)
    plt.close()

save_history_plot(hist1.history, 'model1_cnn')
save_history_plot(hist2.history, 'model2_aug')
save_history_plot(hist3.history, 'model3_tl')

# Comparison bar chart
fig, ax = plt.subplots(figsize=(6, 4))
colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax.bar(list(accs.keys()), list(accs.values()), color=colors)
ax.set_ylim(0, 1)
ax.set_ylabel('Validation Accuracy')
ax.set_title('Model Comparison - Validation Accuracy')
for bar, val in zip(bars, list(accs.values())):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.4f}', ha='center')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'model_comparison.png'), dpi=150)
plt.close()
print("Plots saved.")

# ── Evaluation on best model ────────────────────────────────────────────────
print("\n" + "="*60)
print("DETAILED EVALUATION (Best Model)")
print("="*60)

val_generator.reset()
y_pred_proba = best_model.predict(val_generator, verbose=1)
y_pred_class = (y_pred_proba > 0.5).astype(int).flatten()
y_test = val_generator.classes

print("\nClassification Report:")
print(classification_report(y_test, y_pred_class, target_names=list(train_generator.class_indices.keys())))

cm = confusion_matrix(y_test, y_pred_class)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(train_generator.class_indices.keys()),
            yticklabels=list(train_generator.class_indices.keys()), ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title(f'Confusion Matrix - {best_name}')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'confusion_matrix.png'), dpi=150)
plt.close()
print(f"Confusion matrix saved. Accuracy: {(y_pred_class == y_test).mean():.4f}")

# ── Error Analysis ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ERROR ANALYSIS")
print("="*60)

filepaths = np.array(val_generator.filepaths)
errors = np.where(y_pred_class != y_test)[0]
print(f"Total errors: {len(errors)} / {len(y_test)}")
print(f"Error rate: {len(errors)/len(y_test)*100:.2f}%")

if len(errors):
    sample = errors[:min(9, len(errors))]
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()
    for i, idx in enumerate(sample):
        img = Image.open(filepaths[idx])
        axes[i].imshow(img)
        tl = list(train_generator.class_indices.keys())[y_test[idx]]
        pl = list(train_generator.class_indices.keys())[y_pred_class[idx]]
        axes[i].set_title(f'True: {tl} | Pred: {pl}')
        axes[i].axis('off')
    for j in range(len(sample), 9):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'error_samples.png'), dpi=150)
    plt.close()
    print("Error samples saved.")

# ── Save results JSON ───────────────────────────────────────────────────────
results = {
    "model1_cnn_scratch": {"val_accuracy": float(val_acc1), "val_loss": float(val_loss1), "epochs_trained": len(hist1.history['loss'])},
    "model2_augmented_cnn": {"val_accuracy": float(val_acc2), "val_loss": float(val_loss2), "epochs_trained": len(hist2.history['loss'])},
    "model3_transfer_learning": {"val_accuracy": float(val_acc3), "val_loss": float(val_loss3), "epochs_trained": len(hist3.history['loss'])},
    "best_model": best_name,
    "best_accuracy": float(accs[best_name]),
    "total_params_model1": int(model1.count_params()),
    "total_params_model2": int(model2.count_params()),
    "total_params_model3": int(model3.count_params()),
}
with open(os.path.join(base_dir, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to results.json")
print("\nDONE!")
