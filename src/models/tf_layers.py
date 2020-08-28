import tensorflow as tf

def cbr(x, out_layer, kernel, stride, dilation):
        x = tf.keras.layers.Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same")(x)
        x = tf.keras.layers.LayerNormalization()(x) #BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        return x
    
def wave_block(x, filters, kernel_size, n):
    dilation_rates = [2**i for i in range(n)]
    x = tf.keras.layers.Conv1D(filters = filters,
                kernel_size = 1,
                padding = 'same')(x)
    res_x = x
    for dilation_rate in dilation_rates:
        tanh_out = tf.keras.layers.Conv1D(filters = filters,
                          kernel_size = kernel_size,
                          padding = 'same', 
                          activation = 'tanh', 
                          dilation_rate = dilation_rate)(x)
        sigm_out = tf.keras.layers.Conv1D(filters = filters,
                          kernel_size = kernel_size,
                          padding = 'same',
                          activation = 'sigmoid', 
                          dilation_rate = dilation_rate)(x)
        x = tf.keras.layers.Multiply()([tanh_out, sigm_out])
        x = tf.keras.layers.Conv1D(filters = filters,
                    kernel_size = 1,
                    padding = 'same')(x)
        res_x = tf.keras.layers.Add()([res_x, x])
    return res_x