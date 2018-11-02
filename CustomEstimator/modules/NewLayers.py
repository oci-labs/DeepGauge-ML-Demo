import tensorflow as tf


class NewLayer(object):

    @staticmethod
    def new_weights(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    @staticmethod
    def new_biases(length):
        return tf.Variable(tf.constant(0.05, shape=[length]))


    @classmethod
    def new_conv_layer(cls,
                       inp,  # The previous layer.
                       num_input_channels,  # Num. channels in prev. layer.
                       filter_size,  # Width and height of each filter.
                       num_filters,
                       strides,
                       padding,
                       use_pooling,
                       pooling_ksize,
                       pooling_strides):  # Use 2x2 max-pooling.


        # Shape of the filter-weights for the convolution.
        # This format is determined by the TensorFlow API.
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        # Create new weights aka. filters with the given shape.
        weights = cls.new_weights(shape=shape)

        # Create new biases, one for each filter.
        biases = cls.new_biases(length=num_filters)

        # Create the TensorFlow operation for convolution.
        # The first and last stride must always be 1,
        # because the first is for the image-number and
        # the last is for the inp-channel.
        # But e.g. strides=[1, 2, 2, 1] would mean that the filter
        # is moved 2 pixels across the x- and y-axis of the image.
        # The padding is set to 'SAME' which means the inp image
        # is padded with zeroes so the size of the output is the same.
        layer = tf.nn.conv2d(input=inp,
                             filter=weights,
                             strides=strides,
                             padding=padding)

        # Add the biases to the results of the convolution.
        # A bias-value is added to each filter-channel.
        layer += biases

        # Use pooling to down-sample the image resolution?
        if use_pooling:
            # This is 2x2 max-pooling, which means that we
            # consider 2x2 windows and select the largest value
            # in each window. Then we move 2 pixels to the next window.
            # e.g. ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1],
            layer = tf.nn.max_pool(value=layer,
                                   ksize=pooling_ksize,
                                   strides=pooling_strides,
                                   padding=padding)

        # Rectified Linear Unit (ReLU).
        # It calculates max(x, 0) for each inp pixel x.
        # This adds some non-linearity to the formula and allows us
        # to learn more complicated functions.
        layer = tf.nn.relu(layer)

        # Note that ReLU is normally executed before the pooling,
        # but since relu(max_pool(x)) == max_pool(relu(x)) we can
        # save 75% of the relu-operations by max-pooling first.

        # We return both the resulting layer and the filter-weights
        # because we will plot the weights later.
        return layer, weights

    @classmethod
    def flatten_conv_layer(cls, layer):

        # Get the shape of the input layer.
        layer_shape = layer.get_shape()

        # The shape of the input layer is assumed to be:
        # layer_shape == [num_images, img_height, img_width, num_channels]

        # The number of features is: img_height * img_width * num_channels
        # We can use a function from TensorFlow to calculate this.
        num_features = layer_shape[1:4].num_elements()

        # Reshape the layer to [num_images, num_features].
        # Note that we just set the size of the second dimension
        # to num_features and the size of the first dimension to -1
        # which means the size in that dimension is calculated
        # so the total size of the tensor is unchanged from the reshaping.
        layer_flat = tf.reshape(layer, [-1, num_features])

        # The shape of the flattened layer is now:
        # [num_images, img_height * img_width * num_channels]

        # Return both the flattened layer and the number of features.
        return layer_flat, num_features

    @classmethod
    def new_fc_layer(cls,
                     inp,  # The previous layer.
                     num_inputs,  # Num. inputs from prev. layer.
                     num_outputs,  # Num. outputs.
                     use_relu=True,
                     use_drop_out=True):  # Use Rectified Linear Unit (ReLU)?

        # Create new weights and biases.
        weights = cls.new_weights(shape=[num_inputs, num_outputs])
        biases = cls.new_biases(length=num_outputs)

        # Calculate the layer as the matrix multiplication of
        # the inp and weights, and then add the bias-values.
        layer = tf.matmul(inp, weights) + biases

        ## dropout
        layer = tf.layers.dropout(layer, rate=0.05, training=use_drop_out)

        # Use ReLU?
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer