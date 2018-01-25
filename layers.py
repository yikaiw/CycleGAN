import tensorflow as tf

## Helpers
def _weights(name, shape, mean=0.0, stddev=0.02):
    # Helper to create an initialized Variable
    var = tf.get_variable(name, shape,
        initializer=tf.random_normal_initializer(mean=mean, stddev=stddev, dtype=tf.float32)
    )
    return var

def _biases(name, shape, constant=0.0):
    # Helper to create an initialized Bias with constant
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(constant))

def _leaky_relu(input, slope):
    return tf.maximum(slope * input, input)

def _norm(input, is_training, norm='instance'):
    return _norm(input, norm, is_training)

def _norm(input, type, is_training=None):
    if type == 'batch':
        with tf.variable_scope("batch_norm"):
            return tf.contrib.layers.batch_norm(
                input, decay=0.9, scale=True, updates_collections=None, is_training=is_training
            )
    elif type == 'instance':
        with tf.variable_scope("instance_norm"):
            depth = input.get_shape()[3]
            scale = _weights("scale", [depth], mean=1.0)
            offset = _biases("offset", [depth])
            mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
            epsilon = 1e-5
            inv = tf.rsqrt(variance + epsilon)
            normalized = (input - mean) * inv
            return scale * normalized + offset
    else:
        return input


def safe_log(x, eps=1e-12):
    return tf.log(x + eps)

## Generator layers
def dk(input, k, reuse=False, norm='instance', is_training=True, name=None):
    # dk denotes a 3 ¡Á 3 Convolution-InstanceNorm-ReLU layer with k filters and stride 2.
    with tf.variable_scope(name, reuse=reuse):
        weights = _weights("w", shape=[3, 3, input.get_shape()[3], k])
        conv = tf.nn.conv2d(input, weights, strides=[1, 2, 2, 1], padding='SAME')
        normalized = _norm(conv, is_training, norm)
        output = tf.nn.relu(normalized)
        return output

def n_Rk(input, reuse, norm='instance', is_training=True, n=6):
    def Rk(input, k, reuse=False, norm='instance', is_training=True, name=None):
        # Rk denotes a residual block that contains two 3 ¡Á 3 convolutional layers with the same number of filters on both layer.
        with tf.variable_scope(name, reuse=reuse):
            with tf.variable_scope('l1', reuse=reuse):
                weights1 = _weights("w1", shape=[3, 3, input.get_shape()[3], k])
                padded1 = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
                conv1 = tf.nn.conv2d(padded1, weights1, strides=[1, 1, 1, 1], padding='VALID')
                normalized1 = _norm(conv1, is_training, norm)
                relu1 = tf.nn.relu(normalized1)
            with tf.variable_scope('l2', reuse=reuse):
                weights2 = _weights("w2", shape=[3, 3, relu1.get_shape()[3], k])
                padded2 = tf.pad(relu1, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
                conv2 = tf.nn.conv2d(padded2, weights2, strides=[1, 1, 1, 1], padding='VALID')
                normalized2 = _norm(conv2, is_training, norm)
            output = input + normalized2
            return output
    depth = input.get_shape()[3]
    for i in range(1, n + 1):
        output = Rk(input, depth, reuse, norm, is_training, 'R{}_{}'.format(depth, i))
        input = output
    return output

def c7s1_k(input, k, reuse=False, norm='instance', activation='relu', is_training=True, name=None):
    # c7s1-k denote a 7 ¡Á 7 Convolution-InstanceNorm-ReLU layer with k filters and stride 1.
    with tf.variable_scope(name, reuse=reuse):
        weights = _weights("w", shape=[7, 7, input.get_shape()[3], k])
        # Reflection padding was used to reduce artifacts.
        padded = tf.pad(input, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
        conv = tf.nn.conv2d(padded, weights, strides=[1, 1, 1, 1], padding='VALID')
        normalized = _norm(conv, is_training, norm)
        if activation == 'relu':
            output = tf.nn.relu(normalized)
        if activation == 'tanh':
            output = tf.nn.tanh(normalized)
        return output


def uk(input, k, reuse=False, norm='instance', is_training=True, name=None, output_size=None):
    # uk denotes a 3 ¡Á 3 fractional-strided-Convolution- InstanceNorm-ReLU layer with k filters, and stride 1/2.
    with tf.variable_scope(name, reuse=reuse):
        input_shape = input.get_shape().as_list()
        weights = _weights("weights", shape=[3, 3, k, input_shape[3]])
        if not output_size:
            output_size = input_shape[1] * 2
        output_shape = [input_shape[0], output_size, output_size, k]
        fsconv = tf.nn.conv2d_transpose(input, weights, output_shape=output_shape,
                                        strides=[1, 2, 2, 1], padding='SAME')
        normalized = _norm(fsconv, is_training, norm)
        output = tf.nn.relu(normalized)
        return output


## Discriminator layers: use 70 ¡Á 70 PatchGAN
def Ck(input, k, slope=0.2, stride=2, reuse=False, norm='instance', is_training=True, name=None):
    # Ck denote a 4 ¡Á 4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2.
    with tf.variable_scope(name, reuse=reuse):
        weights = _weights("w", shape=[4, 4, input.get_shape()[3], k])
        conv = tf.nn.conv2d(input, weights, strides=[1, stride, stride, 1], padding='SAME')
        normalized = _norm(conv, is_training, norm)
        output = _leaky_relu(normalized, slope)
        return output

def last_layer(input, reuse=False, use_sigmoid=False, name=None):
    # After the last layer, we apply a convolution to produce a 1 dimensional output: 1 filter with size 4x4, stride 1.
    with tf.variable_scope(name, reuse=reuse):
        weights = _weights("w", shape=[4, 4, input.get_shape()[3], 1])
        biases = _biases("b", [1])
        conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')
        output = conv + biases
        if use_sigmoid:
            output = tf.sigmoid(output)
        return output