import os
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import RandomFlip, RandomRotation, RandomTranslation, RandomZoom
from keras.utils import Sequence
from keras.preprocessing.image import load_img, img_to_array
from keras.config import floatx

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


class PASCALVOCIterator(Sequence):
    """
    Iterator adaptado a la estructura de `dataset/` proporcionada:
    - dataset/train/sat: imágenes (.tiff)
    - dataset/train/map: máscaras (grayscale, .tif)
    - dataset/valid/sat y dataset/valid/map para validación (opcional)

    No requiere `ImageSets` ni `train.txt`. Empareja por nombre de archivo.
    """

    IMG_EXTS = (".tiff",)  # imágenes en sat
    MASK_EXTS = (".tif",)  # máscaras en map

    def __init__(self, directory, split="train",
                 target_size=(256, 256), color_mode='grayscale',
                 batch_size=32, shuffle=True, seed=None,
                 interpolation='nearest'):

        self.directory = directory
        self.split = split  # 'train' | 'valid' | 'test'
        self.target_size = tuple(target_size)
        self.color_mode = color_mode
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.interpolation = interpolation

        # Augmentación (solo para train)
        self.augmentation = Sequential([
            RandomFlip("vertical"),
            RandomFlip("horizontal"),
            RandomRotation(0.1),
            RandomTranslation(0.1, 0.1),
            RandomZoom(0.1)
        ]) if split == "train" else None

        # Rutas según estructura entregada
        self.images_path = os.path.join(directory, split, "sat")
        self.masks_path = os.path.join(directory, split, "map")

        for item in [self.images_path, self.masks_path]:
            assert os.path.exists(item), f"Path does not exist: {item}"

        # Listamos imágenes y emparejamos con máscaras por nombre
        img_files = [f for f in os.listdir(self.images_path) if f.lower().endswith(self.IMG_EXTS)]
        img_files.sort()

        self.filenames = []
        self.masks = []
        missing = 0
        for fname in img_files:
            base = os.path.splitext(fname)[0]
            # Buscamos máscara con extensiones esperadas
            mask_candidates = [
                os.path.join(self.masks_path, base + ext) for ext in self.MASK_EXTS
            ]
            mask_path = next((p for p in mask_candidates if os.path.exists(p)), None)
            if mask_path is not None:
                self.filenames.append(os.path.join(self.images_path, fname))
                self.masks.append(mask_path)
            else:
                missing += 1

        self.samples = len(self.filenames)
        if self.samples == 0:
            raise ValueError(
                f"No se encontraron pares imagen-máscara en '{self.images_path}' (.tiff) y '{self.masks_path}' (.tif). "
                f"Verifica que los nombres de archivo coincidan por basename."
            )

        # Índices barajados
        self.indices = np.arange(self.samples)
        if self.shuffle:
            if self.seed is not None:
                np.random.seed(self.seed)
                
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(self.samples / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self._generate_batch(batch_indices)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _generate_batch(self, batch_indices):
        batch_x = []
        batch_y = []

        for i in batch_indices:
            # Imagen (cargamos como grayscale para que coincida con input_shape)
            img_path = self.filenames[i]
            img = load_img(
                img_path,
                color_mode=self.color_mode,  # 'grayscale' por defecto
                target_size=self.target_size,
                interpolation=self.interpolation
            )
            x = img_to_array(img) / 255.0

            # Augmentación (solo en train)
            if self.augmentation is not None:
                x = self.augmentation(x, training=True)

            # Máscara
            mask_path = self.masks[i]
            mask = load_img(
                mask_path,
                color_mode='grayscale',
                target_size=self.target_size,
                interpolation=self.interpolation
            )
            y = img_to_array(mask) / 255.0

            batch_x.append(x)
            batch_y.append(y)

        batch_x = np.array(batch_x, dtype=floatx())
        batch_y = np.array(batch_y, dtype=floatx())

        return batch_x, batch_y