import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# ---------------------------
#   CeFE BLOCK
# ---------------------------
class CeFEBlock(layers.Layer):
    def __init__(self, filters, strides=1, activation="swish", **kwargs):
        super(CeFEBlock, self).__init__(**kwargs)
        self.activation_name = activation
        self.dwconv = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same",
                                             use_bias=False, name="cefe_dwconv")
        self.bn = layers.BatchNormalization(name="cefe_bn")
        self.conv3x3 = layers.Conv2D(filters, kernel_size=(3, 3), padding="same",
                                     use_bias=False, name="cefe_conv3x3")
        self.act = layers.Activation(activation, name="cefe_activation")
        self.conv1x1 = layers.Conv2D(filters, kernel_size=(1, 1), padding="same",
                                     use_bias=False, name="cefe_conv1x1")
        self.concat = layers.Concatenate(name="cefe_concat")
        self.out_conv = layers.Conv2D(filters, kernel_size=1, strides=strides,
                                      padding="same", use_bias=False, name="cefe_out_conv")

    def call(self, inputs):
        input_x = inputs
        x = self.dwconv(inputs)
        x = self.bn(x)
        x = self.conv3x3(x)
        x = self.act(x)
        x_pw2 = self.conv1x1(x)
        x = self.concat([input_x, x_pw2])
        x = self.out_conv(x)
        return x