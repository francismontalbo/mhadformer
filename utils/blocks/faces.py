import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# ---------------------------
#   FACeS BLOCK
# ---------------------------
class FACeSBlock(layers.Layer):
    def __init__(self, kernel_size=3, **kwargs):
        super(FACeSBlock, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        
        self.concat = layers.Concatenate(name="faces_concat")
        self.bn_a = layers.BatchNormalization(name="faces_bn_a")
        self.bn_b = layers.BatchNormalization(name="faces_bn_b")

    def build(self, input_shape):
        filters = input_shape[-1]
        self.conv_a_stem = layers.Conv2D(filters, kernel_size=(self.kernel_size, self.kernel_size),
                                         padding='same', name="faces_a_stem_conv")
        self.conv_a_left = layers.Conv2D(filters, kernel_size=(self.kernel_size, self.kernel_size),
                                         padding='same', name="faces_a_left_conv")
        self.conv_a_out_sepconv = layers.SeparableConv2D(filters, kernel_size=(self.kernel_size, self.kernel_size),
                                                         padding='same', name="faces_a_out_sepconv")
        self.conv_b_stem = layers.Conv2D(filters, kernel_size=(self.kernel_size, self.kernel_size),
                                         padding='same', name="faces_b_stem_conv")
        self.conv_b_left = layers.Conv2D(filters, kernel_size=(self.kernel_size, self.kernel_size),
                                         padding='same', name="faces_b_left_conv")
        self.conv_b_out_sepconv = layers.SeparableConv2D(filters, kernel_size=(self.kernel_size, self.kernel_size),
                                                         padding='same', name="faces_b_out_sepconv")
        self.final_conv = layers.Conv2D(filters, kernel_size=(1, 1), padding='same',
                                        kernel_regularizer=regularizers.l2(0.01),
                                        name="faces_final_conv")
        super(FACeSBlock, self).build(input_shape)

    def call(self, x):
        stem = x
        
        a_stem = self.conv_a_stem(stem)
        a_left = self.conv_a_left(a_stem)
        bn_a = self.bn_a(stem)
        a_merge = layers.Add(name="faces_a_merge_add")([a_left, bn_a])
        a_out = self.conv_a_out_sepconv(a_merge)
        
        b_stem = self.conv_b_stem(stem)
        b_left = self.conv_b_left(b_stem)
        bn_b = self.bn_b(stem)
        b_merge = layers.Multiply(name="faces_b_merge_multiply")([b_left, bn_b])
        b_out = self.conv_b_out_sepconv(b_merge)

        matmul_out = tf.linalg.matmul(a_out, b_out, transpose_b=True)
        x_last = self.final_conv(matmul_out)
        return self.concat([stem, x_last])