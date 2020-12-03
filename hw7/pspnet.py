from math import ceil

from keras import layers
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization, Activation, Input, Dropout, \
    ZeroPadding2D, Lambda
from keras.layers.merge import Concatenate, Add
from keras.models import Model
from keras.optimizers import SGD


def _build_pspnet(nb_classes, resnet_layers, input_shape,
                  activation='softmax'):

    assert IMAGE_ORDERING == 'channels_last'

    inp = Input((input_shape[0], input_shape[1], 3))

    res = ResNet(inp, layers=resnet_layers)

    psp = build_pyramid_pooling_module(res, input_shape)

    x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",
               use_bias=False)(psp)
    x = BN(name="conv5_4_bn")(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(nb_classes, (1, 1), strides=(1, 1), name="conv6")(x)
    # x = Lambda(Interp, arguments={'shape': (
    #    input_shape[0], input_shape[1])})(x)
    x = Interp([input_shape[0], input_shape[1]])(x)

    model = get_segmentation_model(inp, x)

    return model

def load_model():
    model_config = {
        "input_height": 713,
        "input_width": 713,
        "n_classes": 19,
        "model_class": "pspnet_101",
    }
    model = model_from_name[model_config['model_class']](
        model_config['n_classes'], input_height=model_config['input_height'],
        input_width=model_config['input_width'])