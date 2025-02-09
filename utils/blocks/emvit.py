import tensorflow as tf
from tensorflow.keras import layers

# =============================================================================
# FAST LAYER
# =============================================================================
import tensorflow as tf
from tensorflow.keras import layers

class FASTLayer(layers.Layer):
    def __init__(self, d, num_heads=1, **kwargs):
        super(FASTLayer, self).__init__(**kwargs)
        self.d = d
        self.num_heads = num_heads
        # Query and Key projections via 1x1 depthwise convolutions.
        self.q_conv = layers.DepthwiseConv2D(kernel_size=1, use_bias=False, name="fast_q_conv")
        self.k_conv = layers.DepthwiseConv2D(kernel_size=1, use_bias=False, name="fast_k_conv")
        # 1x1 convolution to compute broadcasting weights.
        self.w_alpha_conv = layers.Conv2D(filters=d // num_heads, kernel_size=1, use_bias=False, name="fast_w_alpha_conv")
        # Linear projection on the global context.
        self.global_context_linear = layers.Conv2D(filters=d, kernel_size=1, use_bias=False, name="fast_global_context_linear")
        # Multi-head self-attention.
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d, name="fast_mha")
        self.transformer_norm = layers.LayerNormalization(name="fast_transformer_norm")
        # Feed-forward network.
        self.ffn_dense = layers.Dense(d, activation='relu', name="fast_ffn_dense")
        self.ffn_norm = layers.LayerNormalization(name="fast_ffn_norm")
        # Final residual addition and normalization.
        self.final_norm = layers.LayerNormalization(name="fast_final_norm")
        self.output_conv = layers.Conv2D(filters=d, kernel_size=1, use_bias=False, name="fast_output_conv")

    def call(self, x):
        # Compute Q and K from input.
        Q = self.q_conv(x)
        K = self.k_conv(x)
        # Compute broadcasting weights.
        w_alpha = self.w_alpha_conv(K)
        alpha = tf.nn.softmax(w_alpha, axis=1)
        # Compute a global query vector by reducing over both spatial dimensions.
        global_queries = tf.reduce_sum(alpha * Q, axis=[1,2], keepdims=True)  # shape: (B, 1, 1, d)
        # Process the global context.
        global_context_processed = self.global_context_linear(global_queries)   # still (B, 1, 1, d)

        # --- FIX: Tile the global context to match the spatial dimensions of x ---
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        global_context_tiled = tf.tile(global_context_processed, [1, H, W, 1])   # Now (B, H, W, d)

        # Flatten for transformer compatibility.
        x_flat = tf.reshape(global_context_tiled, (B, H * W, self.d))

        # Apply multi-head self-attention.
        attn_out = self.mha(x_flat, x_flat)
        transformer_output = self.transformer_norm(x_flat + attn_out)
        # Feed-forward network.
        ffn_out = self.ffn_dense(transformer_output)
        ffn_out = self.ffn_norm(transformer_output + ffn_out)
        # Reshape back to spatial dimensions.
        global_context_processed = tf.reshape(ffn_out, (B, H, W, self.d))
        # Residual connection and final processing.
        x_hat = x + global_context_processed
        x_hat_norm = self.final_norm(x_hat)
        x_out = self.output_conv(x_hat_norm)
        return x_out


# =============================================================================
# CUSTOM MLP-MIXER BLOCK
# =============================================================================
class MLPMixerBlock(layers.Layer):
    def __init__(self, tokens_mlp_dim, dropout_rate=0.1, activation="swish", **kwargs):
        super(MLPMixerBlock, self).__init__(**kwargs)
        self.tokens_mlp_dim = tokens_mlp_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.layernorm = layers.LayerNormalization(name="mlp_mixer_ln")
        self.dropout = layers.Dropout(dropout_rate, name="mlp_mixer_dropout")
        self.dense1 = None
        self.dense2 = None

    def build(self, input_shape):
        # Determine if input is 4D (B, H, W, C) or 3D (B, num_patches, C)
        if len(input_shape) == 4:
            # Calculate number of patches = H * W
            num_patches = input_shape[1] * input_shape[2]
        else:
            num_patches = input_shape[1]

        self.dense1 = layers.Dense(
            self.tokens_mlp_dim, 
            activation=self.activation,
            name="mlp_mixer_dense1"
        )
        self.dense2 = layers.Dense(
            num_patches, 
            name="mlp_mixer_dense2"
        )
        super(MLPMixerBlock, self).build(input_shape)

    def call(self, x):
        reshaped = False
        orig_shape = tf.shape(x)  # dynamic shape

        # If input is 4D (B, H, W, C), reshape it to (B, H*W, C)
        if len(x.shape) == 4:
            reshaped = True
            # Assuming static spatial dimensions are available.
            H = x.shape[1]
            W = x.shape[2]
            B = orig_shape[0]
            C = x.shape[-1]
            x = tf.reshape(x, (B, H * W, C))

        residual = x
        x_norm = self.layernorm(x)
        # Permute to (B, channels, num_patches) for token mixing.
        x_perm = tf.transpose(x_norm, perm=[0, 2, 1])
        x_perm = self.dense1(x_perm)
        # Permute back to (B, num_patches, tokens_mlp_dim).
        x_perm = tf.transpose(x_perm, perm=[0, 2, 1])
        x_perm = self.dense2(x_perm)
        x_perm = self.dropout(x_perm)
        # Permute again to (B, tokens_mlp_dim, num_patches)
        x_perm = tf.transpose(x_perm, perm=[0, 2, 1])
        # If necessary, project residual to match dimensions.
        if x_perm.shape[-1] != residual.shape[-1]:
            residual = layers.Dense(x_perm.shape[-1], name="mlp_mixer_residual_proj")(residual)
        out = residual + x_perm

        # If we originally reshaped a 4D tensor, reshape the output back to 4D.
        if reshaped:
            out = tf.reshape(out, (orig_shape[0], H, W, out.shape[-1]))
        return out

# =============================================================================
# FAST BLOCK
# =============================================================================
class FASTBlock(layers.Layer):
    def __init__(self, num_layers, projection_dim, num_heads=1, dropout_rate=0.1, **kwargs):
        super(FASTBlock, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.ln1_layers = [layers.LayerNormalization(epsilon=1e-6, name=f"fastblock_ln1_{i}")
                           for i in range(num_layers)]
        self.fast_layers = [FASTLayer(d=projection_dim, num_heads=num_heads, name=f"fastlayer_{i}")
                            for i in range(num_layers)]
        self.ln2_layers = [layers.LayerNormalization(epsilon=1e-6, name=f"fastblock_ln2_{i}")
                           for i in range(num_layers)]
        self.mlp_mixer_layers = [MLPMixerBlock(tokens_mlp_dim=projection_dim,
                                               dropout_rate=dropout_rate,
                                               name=f"mlp_mixer_{i}")
                                  for i in range(num_layers)]

    def call(self, x):
        for i in range(self.num_layers):
            x1 = self.ln1_layers[i](x)
            attn_out = self.fast_layers[i](x1)
            x = x + attn_out
            x1 = self.ln2_layers[i](x)
            mlp_out = self.mlp_mixer_layers[i](x1)
            x = x + mlp_out
        return x

# =============================================================================
# EMViT BLOCK
# =============================================================================
class EMViTBlock(layers.Layer):
    def __init__(self, num_blocks, projection_dim, strides=1, activation="swish",
                 dropout_rate=0.1, **kwargs):
        super(EMViTBlock, self).__init__(**kwargs)
        self.projection_dim = projection_dim
        self.activation = activation
        self.conv3x3 = layers.Conv2D(filters=projection_dim, kernel_size=(3, 3),
                                     padding='same', activation=activation,
                                     name="emvit_conv3x3")
        self.conv1x1 = layers.Conv2D(filters=projection_dim, kernel_size=(1, 1),
                                     padding='same', activation=activation,
                                     name="emvit_conv1x1")
        self.fast_block = FASTBlock(num_layers=num_blocks, projection_dim=projection_dim,
                                    num_heads=1, dropout_rate=dropout_rate, name="emvit_fastblock")
        self.depthwise = layers.DepthwiseConv2D(kernel_size=(2, 2), padding='same',
                                                strides=(1, 1), name="emvit_depthwise")
        self.folded_conv = layers.Conv2D(filters=projection_dim, kernel_size=(1, 1),
                                         strides=2, padding='same', activation=activation,
                                         name="emvit_folded_conv")

    def call(self, x):
        x = self.conv3x3(x)
        x = self.conv1x1(x)
        x = self.fast_block(x)
        x = self.depthwise(x)
        x = self.folded_conv(x)
        return x
