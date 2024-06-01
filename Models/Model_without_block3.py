import tensorflow as tf

def ResidualBlock(x, num_filters, kernel_size):
    """
    Defines a residual block with two convolutional layers and a skip connection.
    """
    y = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, strides=1, padding='same')(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation('relu')(y)

    y = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, strides=1, padding='same')(y)
    y = tf.keras.layers.BatchNormalization()(y)

    x = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=1, strides=1, padding='same')(x)

    y = tf.keras.layers.Add()([x, y])

    return y

def SEBlock(inputs, reduction=16, if_train=True):
    """
    Defines a Squeeze-and-Excitation (SE) block.
    """
    x = tf.keras.layers.GlobalAveragePooling1D()(inputs)
    x = tf.keras.layers.Dense(int(x.shape[-1]) // reduction, use_bias=False, activation=tf.keras.activations.relu, trainable=if_train)(x)
    x = tf.keras.layers.Dense(int(inputs.shape[-1]), use_bias=False, activation=tf.keras.activations.hard_sigmoid, trainable=if_train)(x)
    return tf.keras.layers.Multiply()([inputs, x])

def Conformer_Model(data_input):
    """
    Defines the Conformer model architecture for data processing.
    """
    # Initial Conv1D Layer
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)(data_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x)

    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x)

    # Residual Blocks
    for _ in range(2):
        x = ResidualBlock(x, num_filters=128, kernel_size=3)

    x = tf.keras.layers.Conv1D(filters=128, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Multi-Head Attention Layer
    attention = tf.keras.layers.MultiHeadAttention(key_dim=64, num_heads=4)(x, x)
    x = tf.keras.layers.Add()([x, attention])
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x)

    # Additional Conv1D Layers
    x = tf.keras.layers.Conv1D(filters=512, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Fully Connected Layers
    x = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dense(units=32, activation=tf.nn.relu)(x)

    # Output Layer
    output = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)(x)

    return output
