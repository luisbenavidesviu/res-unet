import os
import datetime

from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint, TensorBoard

# Hyper parameters
model_name = "./res_unet_"
input_shape = (512, 512, 1)  # Grayscale for segmentation
dataset_folder = "dataset"  # Provided structure: train/valid/test with sat/map
batch_size = 2

# Archivos de guardado
timestamp = datetime.datetime.now().strftime("_%d_%m_%y_%H_%M_%S")
model_file = model_name + timestamp + ".keras"             # modelo completo
weights_file = model_name + timestamp + ".weights.h5"      # solo pesos

model = build_res_unet(input_shape=input_shape)
optimizer = Adadelta()
model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[dice_coef])

os.makedirs("models", exist_ok=True)

# Guardar el mejor modelo completo (.keras)
checkpoint_model = ModelCheckpoint(
    filepath=os.path.join("models", model_file),
    monitor="loss",
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# Guardar solo los mejores pesos (.weights.h5)
checkpoint_weights = ModelCheckpoint(
    filepath=os.path.join("models", weights_file),
    monitor="loss",
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

tensorboard = TensorBoard()

# Generadores
try:
    train_gen = PASCALVOCIterator(
        directory=dataset_folder,
        split="train",
        target_size=(input_shape[0], input_shape[1]),
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle=True
    )
except ValueError as e:
    print(f"Error preparando train: {e}")
    raise

# Validación si existe
try:
    val_gen = PASCALVOCIterator(
        directory=dataset_folder,
        split="valid",
        target_size=(input_shape[0], input_shape[1]),
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle=False
    )
    validation_data = val_gen
except (AssertionError, ValueError) as e:
    print(f"Validación no disponible: {e}")
    val_gen = None
    validation_data = None

# Steps
steps = len(train_gen)
if steps == 0:
    raise ValueError("El generador de entrenamiento no tiene muestras.")

# Entrenamiento
model.fit(
    train_gen,
    validation_data=validation_data,
    steps_per_epoch=steps,
    epochs=50,
    callbacks=[tensorboard, checkpoint_model, checkpoint_weights]
)