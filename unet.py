from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    UpSampling2D,
    BatchNormalization,
    Activation,
    Add,
    Concatenate
)


def res_block(x, nb_filters, strides):
    # Residual path
    res_path = BatchNormalization()(x)
    res_path = Activation("relu")(res_path)
    res_path = Conv2D(nb_filters[0], (3, 3), padding='same', strides=strides[0])(res_path)

    res_path = BatchNormalization()(res_path)
    res_path = Activation("relu")(res_path)
    res_path = Conv2D(nb_filters[1], (3, 3), padding='same', strides=strides[1])(res_path)

    # Shortcut path
    shortcut = Conv2D(nb_filters[1], (1, 1), strides=strides[0])(x)
    shortcut = BatchNormalization()(shortcut)

    # Merge
    return Add()([shortcut, res_path])


def encoder(x):
    to_decoder = []

    # Initial residual block
    main_path = Conv2D(64, (3, 3), padding='same')(x)
    main_path = BatchNormalization()(main_path)
    main_path = Activation("relu")(main_path)

    main_path = Conv2D(64, (3, 3), padding='same')(main_path)
    shortcut = Conv2D(64, (1, 1))(x)
    shortcut = BatchNormalization()(shortcut)

    main_path = Add()([shortcut, main_path])
    to_decoder.append(main_path)

    # Downsampling residual blocks
    main_path = res_block(main_path, [128, 128], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    main_path = res_block(main_path, [256, 256], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    return to_decoder


def decoder(x, from_encoder):
    # First up + concat + residual
    main_path = UpSampling2D((2, 2))(x)
    main_path = Concatenate(axis=3)([main_path, from_encoder[2]])
    main_path = res_block(main_path, [256, 256], [(1, 1), (1, 1)])

    # Second
    main_path = UpSampling2D((2, 2))(main_path)
    main_path = Concatenate(axis=3)([main_path, from_encoder[1]])
    main_path = res_block(main_path, [128, 128], [(1, 1), (1, 1)])

    # Third
    main_path = UpSampling2D((2, 2))(main_path)
    main_path = Concatenate(axis=3)([main_path, from_encoder[0]])
    main_path = res_block(main_path, [64, 64], [(1, 1), (1, 1)])

    return main_path


def build_res_unet(input_shape):
    inputs = Input(shape=input_shape)

    # Encoder
    to_decoder = encoder(inputs)

    # Bottleneck
    x = res_block(to_decoder[2], [512, 512], [(2, 2), (1, 1)])

    # Decoder
    x = decoder(x, to_decoder)

    # Final segmentation output
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)

    return Model(inputs=inputs, outputs=outputs)