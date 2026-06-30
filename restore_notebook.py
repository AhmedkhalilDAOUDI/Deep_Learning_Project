
import re
import json
import uuid

path = r"C:\Users\Khalil\Desktop\Deep Learning project\consignes_projet_final_classification_images.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# The orphaned block that should be removed
orphaned = '''    "Utiliser des techniques de Data augmentation. L'objectif est d'enrichir le training set à partir des images initiales afin d'améliorer la performance du modèle.\\n",
    "Votre accuracy s'améliore t-elle post votre data augmentation ?\\n",
    "Vous êtes libre de structurer cette partie comme vous le jugez pertinent."
   ]
  },
'''

if orphaned not in content:
    print("Orphaned block not found exactly, searching for partial match...")
    # Try to find it with some flexibility
    idx = content.find("Utiliser des techniques de Data augmentation")
    if idx == -1:
        print("Could not find the orphaned block")
        exit(1)
    # Find surrounding context
    start = content.rfind('   ]\n  },\n    ', 0, idx)
    end = content.find('  },\n  {\n   "cell_type": "code"', idx)
    print(f"Found orphaned block at index {start} to {end}")
    content = content[:start] + content[end:]
else:
    content = content.replace(orphaned, '')

# Now we need to insert the missing cells right after code cell 25 ends
# Find where code cell 25 ends: it should be followed by markdown cell 31 (Transfer learning)
# Let's find the markdown cell 31
markdown31_start = content.find('  {\n   "cell_type": "markdown",\n   "metadata": {\n    "id": "gMD8iyN-xR3F"')
if markdown31_start == -1:
    print("Could not find markdown cell 31")
    exit(1)

# The cells to insert: markdown 30 + code 26 + code 27 + code 28 + code 29
cells_to_insert = []

# markdown cell 30
markdown30 = '''  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Zd0VgWZ9xR3F"
   },
   "source": [
    "# 10) Data augmentation (optionnel)\\n",
    "Utiliser des techniques de Data augmentation. L'objectif est d'enrichir le training set à partir des images initiales afin d'améliorer la performance du modèle.\\n",
    "Votre accuracy s'améliore t-elle post votre data augmentation ?\\n",
    "Vous êtes libre de structurer cette partie comme vous le jugez pertinent."
   ]
  },
'''

code26_source = [
    "# Rich augmentation for an improved baseline",
    "aug_train_datagen = ImageDataGenerator(",
    "    rescale=1./255,",
    "    rotation_range=30,",
    "    width_shift_range=0.18,",
    "    height_shift_range=0.18,",
    "    shear_range=0.18,",
    "    zoom_range=0.25,",
    "    horizontal_flip=True,",
    "    brightness_range=(0.8, 1.2),",
    "    fill_mode='nearest'",
    ")",
    "",
    "aug_train_generator = aug_train_datagen.flow_from_directory(",
    "    train_path,",
    "    target_size=IMG_SIZE,",
    "    batch_size=BATCH_SIZE,",
    "    class_mode='binary'",
    ")",
    "",
    "model_aug = Sequential([",
    "    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),",
    "    BatchNormalization(),",
    "    MaxPooling2D((2,2)),",
    "",
    "    Conv2D(64, (3,3), activation='relu', padding='same'),",
    "    BatchNormalization(),",
    "    MaxPooling2D((2,2)),",
    "    Dropout(0.25),",
    "",
    "    Conv2D(128, (3,3), activation='relu', padding='same'),",
    "    BatchNormalization(),",
    "    MaxPooling2D((2,2)),",
    "    Dropout(0.3),",
    "",
    "    Flatten(),",
    "    Dense(256, activation='relu'),",
    "    BatchNormalization(),",
    "    Dropout(0.4),",
    "    Dense(1, activation='sigmoid')",
    "])",
    "",
    "model_aug.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])",
]

code27_source = [
    "early_stopping_aug = EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1)",
    "",
    "history_aug = model_aug.fit(",
    "    aug_train_generator,",
    "    validation_data=val_generator,",
    "    epochs=EPOCHS,",
    "    callbacks=[early_stopping_aug]",
    ")",
]

code28_source = [
    "val_loss_aug, val_acc_aug = model_aug.evaluate(val_generator, verbose=0)",
    "print(f\"Validation accuracy after augmentation: {val_acc_aug:.4f}\")",
]

code29_source = [
    "plt.figure(figsize=(10,4))",
    "plt.plot(history_aug.history['loss'], label='train_loss')",
    "plt.plot(history_aug.history['val_loss'], label='val_loss')",
    "plt.title('Augmented model - loss')",
    "plt.legend()",
    "plt.show()",
]

def build_code_cell(source_lines, exec_count, cell_id):
    escaped_lines = [line.replace('\\', '\\\\').replace('"', '\\"') for line in source_lines]
    source_json = ",\n".join([f'     "{line}\\n"' for line in escaped_lines])
    return f'''  {{
   "cell_type": "code",
   "execution_count": {exec_count},
   "metadata": {{
    "id": "{cell_id}"
   }},
   "outputs": [],
   "source": [
{source_json}
   ]
  }},
'''

code26 = build_code_cell(code26_source, 26, "82mMcz6PxR3F")
code27 = build_code_cell(code27_source, 27, "abc123def456")
code28 = build_code_cell(code28_source, 28, "def456abc123")
code29 = build_code_cell(code29_source, 29, "ghi789jkl012")

all_cells = markdown30 + code26 + code27 + code28 + code29

# Insert before markdown31
content = content[:markdown31_start] + all_cells + content[markdown31_start:]

# Also need to fix the execution_count of the base_model cell from 29 to 30
# Let's find the base_model cell in the new content
# It should now be after markdown31 and code cells 26-29
# Find the cell with "base_model = MobileNetV2"
base_idx = content.find('base_model = MobileNetV2')
if base_idx != -1:
    # Find the execution_count line before it in the same cell
    # Look backwards for '"execution_count": 29,'
    exec_search = content.rfind('"execution_count": 29,', 0, base_idx)
    if exec_search != -1:
        content = content[:exec_search] + '"execution_count": 30,' + content[exec_search + len('"execution_count": 29,'):]
        print("Fixed execution_count to 30 for base_model cell")
    else:
        print("Could not find execution_count 29 before base_model")
else:
    print("Could not find base_model cell")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed notebook written with missing cells restored.")
