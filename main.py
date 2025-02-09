import tensorflow as tf

# Import your custom block classes.
from utils.blocks.stemblock import StemBlock
from utils.blocks.cefe import CeFEBlock
from utils.blocks.emvit import EMViTBlock
from utils.blocks.faces import FACeSBlock

def main():
    # Create a dummy input tensor (batch size 1, 224x224 image with 3 channels)
    input_tensor = tf.random.normal([1, 224, 224, 3])
    print("Input tensor shape:", input_tensor.shape)

    # -------------------------------
    # Test the StemBlock
    # -------------------------------
    stem = StemBlock(name="stem_block")
    stem_out = stem(input_tensor)
    print("StemBlock output shape:", stem_out.shape)

    # -------------------------------
    # Test the CeFEBlock
    # -------------------------------
    # For example, set filters to 64 and stride to 1.
    cefe = CeFEBlock(filters=64, strides=1, activation="swish", name="cefe_block")
    cefe_out = cefe(stem_out)
    print("CeFEBlock output shape:", cefe_out.shape)

    # -------------------------------
    # Test the EMViTBlock
    # -------------------------------
    # For example, use 2 FAST block layers with projection dimension 64.
    emvit = EMViTBlock(num_blocks=2, projection_dim=64, strides=1,
                       activation="swish", dropout_rate=0.1, name="emvit_block")
    emvit_out = emvit(cefe_out)
    print("EMViTBlock output shape:", emvit_out.shape)

    # -------------------------------
    # Test the FACeSBlock
    # -------------------------------
    faces = FACeSBlock(name="faces_block")
    faces_out = faces(emvit_out)
    print("FACeSBlock output shape:", faces_out.shape)

if __name__ == "__main__":
    main()
