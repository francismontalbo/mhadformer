import tensorflow as tf
from tensorflow.keras import layers

# ---------------------------
#   STEM BLOCK
# ---------------------------
class StemBlock(layers.Layer):
    def __init__(self, **kwargs):
        super(StemBlock, self).__init__(**kwargs)
        self.bn = layers.BatchNormalization(name="stem_bn")
        self.pad1 = layers.ZeroPadding2D(padding=(3, 3), name="stem_pad1")
        self.dwconv = layers.DepthwiseConv2D(kernel_size=(7, 7), strides=(2, 2),
                                             name="stem_dwconv")
        self.pad2 = layers.ZeroPadding2D(padding=(1, 1), name="stem_pad2")
        self.maxpool = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                            padding='valid', name="stem_maxpool")

    def call(self, inputs):
        x = self.bn(inputs)
        x = self.pad1(x)
        x = self.dwconv(x)
        x = self.pad2(x)
        return self.maxpool(x)