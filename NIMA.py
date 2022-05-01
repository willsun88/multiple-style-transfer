from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, Dropout
from keras.models import Model


def nima_init(input=None):
    base_model = InceptionResNetV2(input_shape=(None, None, 3), include_top=False, pooling='avg', weights=None,
                                   input_tensor=input)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights('NIMA/inception_resnet_weights.h5')
    return model


def nima_preprocess(image):
    return (image / 127.5) - 1.0