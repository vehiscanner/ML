import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt

base_dir = 'foto/'

def build_compile_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Mengompilasi model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
        metrics=['accuracy']
    )
    
    return model

# Tentukan generator data untuk augmentasi gambar
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Membagi data menjadi data latih dan data uji
classes = os.listdir(base_dir)
kfold_splits = min(5, len(classes))  # Memastikan jumlah lipatan tidak melebihi jumlah kelas
batch_size = 32
epochs = 15

# K-fold cross-validation
skf = StratifiedKFold(n_splits=kfold_splits, shuffle=True, random_state=42)

# Inisialisasi list untuk menyimpan hasil evaluasi setiap fold
all_eval_results = []
all_histories = []

for fold, (train_index, test_index) in enumerate(skf.split(classes, [187] * len(classes))):
    print(f"Fold {fold+1}/{kfold_splits}")

    train_classes = [classes[i] for i in train_index]
    test_classes = [classes[i] for i in test_index]

    # Membuat generator untuk data pelatihan
    train_generator = datagen.flow_from_directory(
        base_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        seed=42,
        classes=train_classes
    )

    # Membuat generator untuk data validasi
    validation_generator = datagen.flow_from_directory(
        base_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        seed=42,
        classes=train_classes
    )

    # Membangun dan mengompilasi model
    model = build_compile_model(num_classes=len(train_classes))

    # Melatih model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )

    all_histories.append(history)  # Menyimpan history setiap fold

    # Evaluasi model pada data uji
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        base_dir,
        target_size=(150, 150),
        batch_size=187,  # Sesuaikan dengan jumlah total gambar per kelas
        class_mode='categorical',
        shuffle=False,
        classes=test_classes
    )

    # Mendapatkan langkah-langkah yang diperlukan untuk satu epoch evaluasi
    test_steps = len(test_generator.filenames) // test_generator.batch_size

    # Reset generator untuk memastikan iterasi dimulai dari awal
    test_generator.reset()

    # Evaluasi model pada data uji
    eval_result = model.evaluate(test_generator, steps=test_steps, verbose=1)
    all_eval_results.append(eval_result)
    print(f"Evaluation Result for Fold {fold+1}/{kfold_splits} (Loss, Accuracy):", eval_result)

    # Prediksi pada data uji
    predictions = model.predict(test_generator, steps=test_steps, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes

    # Matriks Confusion
    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    print(f"\nFold {fold + 1} - Confusion Matrix:")
    print(conf_matrix)

    # Laporan Klasifikasi
    class_labels = list(test_generator.class_indices.keys())
    classification_rep = classification_report(true_classes, predicted_classes, labels=np.arange(len(class_labels)), target_names=class_labels)
    print("\nClassification Report:")
    print(classification_rep)

    # Menampilkan visualisasi hasil pelatihan (training) dan validasi (validation) untuk setiap fold
    print(f"\nVisualizations for Fold {fold + 1}/{kfold_splits}")

    # Plot akurasi pelatihan (training)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Fold {fold + 1} - Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss pelatihan (training)
    plt.subplot(1, 2, 3)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Fold {fold + 1} - Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
