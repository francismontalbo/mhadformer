import tensorflow as tf
from tensorflow.keras import layers

from utils.blocks.stemblock import StemBlock
from utils.blocks.cefe import CeFEBlock
from utils.blocks.emvit import EMViTBlock
from utils.blocks.faces import FACeSBlock

class MHADFormer(tf.keras.Model):
    def __init__(self, num_classes=5, image_size=224, **kwargs):
        super(MHADFormer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.image_size = image_size

        # Define blocks based on the architecture specifications:
        # 1. Stem Block: 224×224×3 → 56×56×3
        self.stem = StemBlock(name="stem_block")
        
        # 2. EMViT Block 1: 56×56×3 → 28×28×16
        self.emvit1 = EMViTBlock(num_blocks=1, projection_dim=16, strides=1,
                                 activation="swish", dropout_rate=0.1, name="emvit_block1")
        
        # 3. CeFE Blocks:
        # CeFE 1 (Stride=1): 28×28×16 → 28×28×32
        self.cefe1 = CeFEBlock(filters=32, strides=1, activation="swish", name="cefe_block1")
        # CeFE 2 (Stride=2): 28×28×32 → 14×14×64
        self.cefe2 = CeFEBlock(filters=64, strides=2, activation="swish", name="cefe_block2")
        
        # 4. EMViT Block 2: 14×14×64 → 7×7×128
        self.emvit2 = EMViTBlock(num_blocks=1, projection_dim=128, strides=1,
                                 activation="swish", dropout_rate=0.1, name="emvit_block2")
        
        # 5. FACeS Block: 7×7×128 → 7×7×256
        self.faces = FACeSBlock(name="faces_block")
        
        # 6. Final processing: PWConv + Swish + BN (7×7×256 → 7×7×256)
        self.final_conv = layers.Conv2D(filters=256, kernel_size=(1, 1), padding="same", name="final_conv")
        self.final_activation = layers.Activation("swish", name="final_activation")
        self.final_bn = layers.BatchNormalization(name="final_bn")
        
        # 7. Classification Head
        self.global_avg_pool = layers.GlobalAveragePooling2D(name="global_avg_pool")
        self.final_dropout = layers.Dropout(0.5, name="final_dropout")
        self.output_layer = layers.Dense(num_classes, activation="softmax", name="output_layer")
        
    def call(self, inputs, training=False):
        # Forward pass: sequentially apply each block
        x = self.stem(inputs)
        x = self.emvit1(x)
        x = self.cefe1(x)
        x = self.cefe2(x)
        x = self.emvit2(x)
        x = self.faces(x)
        x = self.final_conv(x)
        x = self.final_activation(x)
        x = self.final_bn(x, training=training)
        x = self.global_avg_pool(x)
        x = self.final_dropout(x, training=training)
        return self.output_layer(x)
