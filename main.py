import tensorflow as tf
from mhadformer import MHADFormer

def main():
    model = MHADFormer(num_classes=5, image_size=224)
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()

if __name__ == "__main__":
    main()
