from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv3D, Conv2D, Input, MaxPool3D, MaxPool2D, Flatten, Activation, concatenate
from tensorflow.keras.backend import expand_dims
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model


def build_model(input_shape_dict):
    video_input_shape = input_shape_dict['video']
    audio_input_shape = input_shape_dict['audio']
    weight_decay = 0.005

    # Video 3D Conv layers
    video_input = Input(video_input_shape)
    x = Conv3D(8, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay))(video_input)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    video_output = Flatten()(x)

    # Audio 2D Conv layers
    audio_input = Input(audio_input_shape)
    x = expand_dims(audio_input)    # add channel dim
    x = Conv2D(4, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding='same')(x)
    audio_output = Flatten()(x)

    # Fully-connected layers
    fc_input = concatenate([video_output, audio_output])
    x = Dense(16, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay))(fc_input)
    # x = Dropout(0.2)(x)
    fc_output = Dense(1, activation='sigmoid', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay))(x)

    model = Model(inputs=[video_input, audio_input], outputs=fc_output)

    return model
