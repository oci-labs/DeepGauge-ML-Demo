import glob
import os
from tensorflow.python.framework import meta_graph
import tensorflow as tf
from modules import LoadImg
import time
import numpy as np


def building_funcs():
    def __build_pipeline(model_name_scope=None,
                         pipeline_params=None,
                         raw_imgs_placeholder=None):
        graph_model = tf.Graph()
        with graph_model.as_default():
            ##################################
            with tf.Session(graph=graph_model) as sess:
                saver = tf.train.import_meta_graph(pipeline_params["checkpoint_file_path"],
                                                   clear_devices=True,
                                                   import_scope='CNN_model')
                saver.restore(tf.get_default_session(),
                              tf.train.latest_checkpoint(pipeline_params["checkpoint_path"]))

            X_image_tf = graph_model.get_tensor_by_name("CNN_model/X_image_tf:0")
            logits_tf = graph_model.get_tensor_by_name("CNN_model/logits_tf:0")
            # logits_tf_sg = tf.stop_gradient(logits_tf)
            ####################################

        graph_pipeline = tf.Graph()
        with graph_pipeline.as_default():
            ##################################
            X_raw = tf.placeholder(tf.float32, shape=[None, None, None, None], name="X_raw")
            meta_graph.import_scoped_meta_graph(pipeline_params["checkpoint_file_path"],
                                                clear_devices=True,
                                                import_scope='img_size_info')

            X_image_tf = graph_pipeline.get_tensor_by_name("img_size_info/X_image_tf:0")

            resized_imgs = tf.identity(tf.image.resize_images(X_raw, (X_image_tf.get_shape().as_list()[1],
                                                                      X_image_tf.get_shape().as_list()[2])),
                                       name='resized_imgs')
            ####################################

        graph = tf.get_default_graph()

        raw_imgs = raw_imgs_placeholder

        meta_graph_1 = tf.train.export_meta_graph(graph=graph_pipeline)
        meta_graph.import_scoped_meta_graph(meta_graph_1,
                                            input_map={"X_raw": raw_imgs},
                                            import_scope=model_name_scope + '_img_pipeline')

        out_1 = graph.get_tensor_by_name(model_name_scope + '_img_pipeline' + '/resized_imgs:0')

        meta_graph_2 = tf.train.export_meta_graph(graph=graph_model)
        meta_graph.import_scoped_meta_graph(meta_graph_2,
                                            input_map={"CNN_model/X_image_tf": out_1},
                                            import_scope=model_name_scope + '_CNN')

        out_2 = graph.get_tensor_by_name(model_name_scope + '_CNN' + '/CNN_model/logits_tf:0')

        return out_2

    def _build_architecture(models_directory=None):
        graph_parent = tf.Graph()
        with graph_parent.as_default():
            raw_imgs = tf.placeholder(tf.float32, shape=[None, None, None, None], name='raw_imgs')
            for i, model in enumerate(os.listdir(models_directory)):
                checkpoint_path = glob.glob(os.path.join(models_directory, model))[0]
                checkpoint_file_path = glob.glob(os.path.join(checkpoint_path, '*.meta'))[0]
                params = {"checkpoint_path": checkpoint_path,
                          "checkpoint_file_path": checkpoint_file_path}

                logits_out = __build_pipeline(
                    model_name_scope='M_' + model,
                    pipeline_params=params,
                    raw_imgs_placeholder=raw_imgs)

                # logits_out = Integrated_Graph.__build_pipeline(
                #     model_name_scope='M_' + model,
                #     pipeline_params=params,
                #     raw_imgs_placeholder=raw_imgs)
                try:
                    final_logits = tf.concat([final_logits, logits_out], 1)
                except:
                    final_logits = logits_out

            final_logits_named = tf.identity(final_logits, name='final_logits')

        graph = tf.get_default_graph()

        meta_graph_3 = tf.train.export_meta_graph(graph=graph_parent)
        meta_graph.import_scoped_meta_graph(meta_graph_3,
                                            import_scope='graph_parent')

        raw_imgs_name = [n.name for n in tf.get_default_graph().as_graph_def().node if 'raw_imgs' in n.name][0]
        raw_imgs_tf = graph.get_tensor_by_name(raw_imgs_name + ':0')

        logits_name = [n.name for n in tf.get_default_graph().as_graph_def().node if 'final_logit' in n.name][0]
        logits_concat = graph.get_tensor_by_name(logits_name + ':0')

        return raw_imgs_tf, logits_concat

    def build_finalGraph_and_return_final_tensors(hidden_units=None,
                                                  y_data=None,
                                                  primary_models_directory=None):
        def _new_weights(shape):
            with tf.name_scope('weights'):
                weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))
            return weights

        def _new_biases(length):
            with tf.name_scope('biases'):
                biases = tf.Variable(tf.constant(0.05, shape=[length]))
            return biases

        def new_fc_layer(inp,
                         num_inputs,
                         num_outputs,
                         use_relu=True,
                         use_drop_out=True,
                         name_scope=''):
            with tf.name_scope(name_scope):
                weights = _new_weights(shape=[num_inputs, num_outputs])
                biases = _new_biases(length=num_outputs)

                layer = tf.matmul(inp, weights) + biases

                if use_drop_out:
                    layer = tf.layers.dropout(layer, rate=0.05, training=use_drop_out)

                if use_relu:
                    layer = tf.nn.relu(layer)

            return layer

        ##
        raw_imgs, concatenated_features = _build_architecture(
            models_directory=primary_models_directory)
        ##
        params_fc = {'hidden_units': hidden_units.copy(),
                     'y_data': y_data.copy()}
        ##
        graph_fc = tf.Graph()
        with graph_fc.as_default():
            X_tf = tf.placeholder(tf.float32, shape=[None, concatenated_features.get_shape().as_list()[1]], name='X_tf')

            #########################
            layer = None
            for n_layer, n_nodes in enumerate(params_fc['hidden_units']):
                if n_layer == 0:
                    layer = new_fc_layer(X_tf,
                                         num_inputs=X_tf.get_shape().as_list()[1],
                                         num_outputs=n_nodes,
                                         name_scope='layer_' + str(n_layer + 1))
                else:
                    layer = new_fc_layer(layer,
                                         num_inputs=layer.get_shape().as_list()[1],
                                         num_outputs=n_nodes,
                                         name_scope='layer_' + str(n_layer + 1))

            logits = new_fc_layer(layer,
                                  num_inputs=layer.get_shape().as_list()[1],
                                  num_outputs=params_fc['y_data'].shape[1],
                                  use_relu=False,
                                  use_drop_out=False,
                                  name_scope='output_layer')

            logits_fc = tf.identity(logits, name='logits_fc')

        ##
        graph = tf.get_default_graph()

        meta_graph_3 = tf.train.export_meta_graph(graph=graph_fc)
        meta_graph.import_scoped_meta_graph(meta_graph_3,
                                            input_map={"X_tf": concatenated_features},
                                            import_scope='DNN_fc')

        logits_fc = graph.get_tensor_by_name('DNN_fc/logits_fc:0')
        return raw_imgs, logits_fc

    return

# raw_imgs, logits_fc = build_finalGraph_and_return_final_tensors(hidden_units=[500, 100],
#                                                                 y_data=y_train,
#                                                                 primary_models_directory='./logs/primary_models/')



###########################
##### prediction
import tensorflow as tf
from modules import LoadImg

X_train, X_test, y_train, y_test, cls_indices = LoadImg.Dataset.prep_datasets(
    ver_ratio=0.2, container_path='data/ImageEveryUnit',
    final_img_width=224, final_img_height=224,
    color_mode="grayscale", random_state=1911, is_trial=True)

with tf.Session() as session:
    writer = tf.summary.FileWriter('./trial/writer_prediction', session.graph)
    saver_3 = tf.train.import_meta_graph('./trial/best_model_main.meta',
                                         clear_devices=True)
    saver_3.restore(tf.get_default_session(),
                    tf.train.latest_checkpoint('./trial'))
    graph_pred = tf.get_default_graph()
    raw_imgs = graph_pred.get_tensor_by_name("raw_imgs:0")
    final_logits = graph_pred.get_tensor_by_name("logits_tf:0")

    ##
    feed_dict_pred = {raw_imgs: X_train}

    ##
    logits = session.run(final_logits, feed_dict=feed_dict_pred)
    writer.close()