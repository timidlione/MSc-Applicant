import tensorflow as tf
from tensorflow.keras import Input
import matplotlib.pyplot as plt  # For saving training curves
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # Core layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping  # Early stopping during training
from tensorflow.keras.regularizers import l2  # L2 regularization

"""
Project: Pill Image Classification (Baseline CNN with L2 regularization)
File: main.py

What this script does
- Loads images from directory structures for train/val/test.
- Builds a small CNN (with L2) and trains it across (epochs, batch_size, lr) combinations.
- Uses EarlyStopping on validation loss and restores the best weights.
- Evaluates on the held-out test set.
- Saves: (1) model graph (.png, once), (2) training curves (.png), (3) trained model (.keras).
- File names include key hyperparameters and the final test accuracy for reproducibility.

Expected data directory layout
train/
  class_1/ *.jpg|png
  class_2/ ...
val/
  class_1/ ...
  class_2/ ...
test/
  class_1/ ...
  class_2/ ...

Notes
- `plot_model` requires Graphviz/Pydot installed in the environment.
- Keep variable names as-is for compatibility with existing tooling/pipelines.
"""


# -----------------------------
# Model definition
# -----------------------------
def create_model(num_classes):

    # L2 regularization applied to all conv/dense layers
    l2_regularizer = l2(0.001)

    model = Sequential([
        Input(shape=(224, 224, 3)),                          # Input: RGB image 224x224
        Conv2D(32, (3, 3), activation='relu',
               kernel_regularizer=l2_regularizer),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu',
               kernel_regularizer=l2_regularizer),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu',
               kernel_regularizer=l2_regularizer),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2_regularizer),
        Dropout(0.3),                                        # Regularization by dropout
        Dense(num_classes, activation='softmax',
              kernel_regularizer=l2_regularizer)             # Multiclass logits
    ])
    return model


# -----------------------------
# Training loop (grid over epochs/batch/lr)
# -----------------------------
def train_and_save_model(epochs_list, batch_sizes, learning_rates, model_structure, train, val, test):
    # epochs_list : list of epoch counts to try
    # batch_sizes : list of batch sizes to try
    # learning_rates : list of learning rates to try
    # model_structure : name prefix for saved files
    # train, val, test : absolute paths to dataset directories

    model_structure_saved = False  # Save model graph (.png) only once

    for epochs in epochs_list:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                # -----------------------------
                # Data loaders (rescale only)
                # -----------------------------
                train_dir = train
                val_dir = val
                test_dir = test

                train_datagen = ImageDataGenerator(rescale=1.0 / 255)
                val_datagen = ImageDataGenerator(rescale=1.0 / 255)
                test_datagen = ImageDataGenerator(rescale=1.0 / 255)

                train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(224, 224),
                    batch_size=batch_size,
                    class_mode='categorical'
                )
                # Validation/Test: keep a fixed order for reproducibility & future error analysis
                val_generator = val_datagen.flow_from_directory(
                    val_dir,
                    target_size=(224, 224),
                    batch_size=batch_size,
                    class_mode='categorical',
                    shuffle=False
                )
                test_generator = test_datagen.flow_from_directory(
                    test_dir,
                    target_size=(224, 224),
                    batch_size=batch_size,
                    class_mode='categorical',
                    shuffle=False
                )

                # -----------------------------
                # Build & compile model
                # -----------------------------
                model = create_model(train_generator.num_classes)
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )

                # -----------------------------
                # Early stopping on validation loss
                # -----------------------------
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )

                # -----------------------------
                # Train
                # -----------------------------
                history = model.fit(
                    train_generator,
                    validation_data=val_generator,
                    epochs=epochs,
                    callbacks=[early_stopping]
                )

                # -----------------------------
                # Final evaluation on test set
                # -----------------------------
                test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)

                # -----------------------------
                # Save model graph (once)
                # -----------------------------
                if not model_structure_saved:
                    plot_model(
                        model,
                        to_file=model_structure + ".png",
                        show_shapes=True,
                        show_layer_names=True
                    )
                    model_structure_saved = True

                # -----------------------------
                # Save training curves (Loss/Accuracy)
                # -----------------------------
                plt.figure(figsize=(12, 5))

                # Loss curve
                plt.subplot(1, 2, 1)
                plt.plot(history.history['loss'], label='Training Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.title(f'Loss | epochs={epochs}, batch={batch_size}, lr={learning_rate}')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()

                # Accuracy curve
                plt.subplot(1, 2, 2)
                plt.plot(history.history['accuracy'], label='Training Accuracy')
                plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
                plt.title(f'Accuracy | epochs={epochs}, batch={batch_size}, lr={learning_rate}')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.legend()

                # Include test accuracy in file name for clear experiment tracking
                plt.savefig(
                    model_structure
                    + f'_loss_accuracy_{epochs}_epochs_{batch_size}_batch_lr_{learning_rate}_T_Acc_{test_accuracy:.4f}.png'
                )
                plt.close()

                # -----------------------------
                # Save trained model (.keras)
                # -----------------------------
                model.save(
                    model_structure
                    + f'_{epochs}_epochs_{batch_size}_batch_lr_{learning_rate}_T_Acc_{test_accuracy:.4f}.keras'
                )


# -----------------------------
# User-defined hyperparameters & data paths
# -----------------------------
# epochs_list = [1, 2, 50, 70, 200]       # alternative example
# batch_sizes = [32, 64]                   # alternative example
epochs_list = [150]                        # number of epochs to run
batch_sizes = [32]                         # batch size
initial_learning_rates = [0.0005]          # learning rate(s)

# Name prefix for saved files; keep consistent with repository naming
model_structure = "new_added_dataset_ES_L2_Attention"

# Absolute paths to dataset directories (edit for your environment)
train = 'data/train'
val = 'data/val'
test = 'data/test'

# Entry point
train_and_save_model(epochs_list, batch_sizes, initial_learning_rates, model_structure, train, val, test)