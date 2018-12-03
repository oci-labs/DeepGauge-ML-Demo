from modules.primary_models_modules.train_modules.NewLayers import NewLayer


##
class Graph(object):

    @staticmethod
    def graph_three_conv_layer(x_image, num_channels, num_classes,
                               ##
                               filter_size1, num_filters1, strides_1,
                               use_pooling_1, pooling_ksize_1, pooling_strides_1,
                               ##
                               filter_size2, num_filters2, strides_2,
                               use_pooling_2, pooling_ksize_2, pooling_strides_2,
                               ##
                               filter_size3, num_filters3, strides_3,
                               use_pooling_3, pooling_ksize_3, pooling_strides_3,
                               ##
                               fc_size, padding):
        ## convolutional layer 1
        layer_conv1, weights_conv1 = NewLayer.new_conv_layer(inp=x_image,
                                                             num_input_channels=num_channels,
                                                             filter_size=filter_size1,
                                                             num_filters=num_filters1,
                                                             use_pooling=use_pooling_1,
                                                             strides=strides_1,
                                                             padding=padding,
                                                             pooling_ksize=pooling_ksize_1,
                                                             pooling_strides=pooling_strides_1)

        ## convolutional layer 2
        layer_conv2, weights_conv2 = NewLayer.new_conv_layer(inp=layer_conv1,
                                                             num_input_channels=num_filters1,
                                                             filter_size=filter_size2,
                                                             num_filters=num_filters2,
                                                             use_pooling=use_pooling_2,
                                                             strides=strides_2,
                                                             padding=padding,
                                                             pooling_ksize=pooling_ksize_2,
                                                             pooling_strides=pooling_strides_2)

        ## convolutional layer 3
        layer_conv3, weights_conv3 = NewLayer.new_conv_layer(inp=layer_conv2,
                                                             num_input_channels=num_filters2,
                                                             filter_size=filter_size3,
                                                             num_filters=num_filters3,
                                                             use_pooling=use_pooling_3,
                                                             strides=strides_3,
                                                             padding=padding,
                                                             pooling_ksize=pooling_ksize_3,
                                                             pooling_strides=pooling_strides_3)

        ## flatten last conv layer
        layer_flat, num_features = NewLayer.flatten_conv_layer(layer_conv3)

        ## fully connected layer 1
        layer_fc1 = NewLayer.new_fc_layer(inp=layer_flat,
                                          num_inputs=num_features,
                                          num_outputs=fc_size,
                                          use_relu=True,
                                          use_drop_out=False)

        ## fully connected layer 2
        layer_fc2 = NewLayer.new_fc_layer(inp=layer_fc1,
                                          num_inputs=fc_size,
                                          num_outputs=fc_size,
                                          use_relu=True,
                                          use_drop_out=False)

        ## fully connected layer 3
        layer_fc3 = NewLayer.new_fc_layer(inp=layer_fc2,
                                          num_inputs=fc_size,
                                          num_outputs=num_classes,
                                          use_relu=False,
                                          use_drop_out=False)
        return layer_fc3

    @staticmethod
    def graph_two_fc_layer(x_appended, num_features, num_classes,
                           fc_size_1, fc_size_2,
                           use_drop_out_1=False, use_drop_out_2=False):

        ## fully connected layer 1
        layer_fc1 = NewLayer.new_fc_layer(inp=x_appended,
                                          num_inputs=num_features,
                                          num_outputs=fc_size_1,
                                          use_relu=True,
                                          use_drop_out=use_drop_out_1)

        ## fully connected layer 2
        layer_fc2 = NewLayer.new_fc_layer(inp=layer_fc1,
                                          num_inputs=fc_size_1,
                                          num_outputs=fc_size_2,
                                          use_relu=True,
                                          use_drop_out=use_drop_out_2)

        ## fully connected layer 3
        layer_fc3 = NewLayer.new_fc_layer(inp=layer_fc2,
                                          num_inputs=fc_size_2,
                                          num_outputs=num_classes,
                                          use_relu=False,
                                          use_drop_out=False)
        return layer_fc3
