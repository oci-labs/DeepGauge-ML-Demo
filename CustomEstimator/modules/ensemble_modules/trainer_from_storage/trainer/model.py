import tensorflow as tf
import os
from tensorflow.python.framework import meta_graph


def create_ensemble_architecture(hidden_units=None,
                                 n_output=None,
                                 primary_models_directory=None,
                                 images_shape=None,
                                 save_path=None):
    class Ensemble():
        @staticmethod
        def __build_pipeline(model_name_scope=None,
                             pipeline_params=None,
                             raw_imgs_placeholder=None):
            graph_model = tf.Graph()
            with graph_model.as_default():
                ##
                with tf.Session(graph=graph_model) as sess:
                    saver = tf.train.import_meta_graph(pipeline_params["checkpoint_file_path"],
                                                       clear_devices=True,
                                                       import_scope='CNN_model')
                    saver.restore(tf.get_default_session(),
                                  tf.train.latest_checkpoint(pipeline_params["checkpoint_path"]))

                X_image_tf = graph_model.get_tensor_by_name("CNN_model/X_image_tf:0")
                logits_tf = graph_model.get_tensor_by_name("CNN_model/logits_tf:0")
                # logits_tf_sg = tf.stop_gradient(logits_tf)
                ##

            graph_pipeline = tf.Graph()
            with graph_pipeline.as_default():
                ##
                X_raw = tf.placeholder(tf.float32, shape=[None, None, None, None], name="X_raw")
                meta_graph.import_scoped_meta_graph(pipeline_params["checkpoint_file_path"],
                                                    clear_devices=True,
                                                    import_scope='img_size_info')

                X_image_tf = graph_pipeline.get_tensor_by_name("img_size_info/X_image_tf:0")

                resized_imgs = tf.identity(tf.image.resize_images(X_raw, (X_image_tf.get_shape().as_list()[1],
                                                                          X_image_tf.get_shape().as_list()[2])),
                                           name='resized_imgs')
                ##

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

        @staticmethod
        def _combine_all_channel(models_directory=None, images_shape=None):
            graph_parent = tf.Graph()
            with graph_parent.as_default():
                raw_imgs = tf.placeholder(tf.float32, shape=[None, None, None, None], name='raw_imgs')
                for i, model in enumerate(tf.gfile.ListDirectory(models_directory)):
                    checkpoint_path = os.path.join(models_directory, model)
                    checkpoint_file_path = tf.gfile.Glob(os.path.join(checkpoint_path, '*.meta'))[0]
                    params = {"checkpoint_path": checkpoint_path,
                              "checkpoint_file_path": checkpoint_file_path}

                    logits_out = Ensemble.__build_pipeline(
                        model_name_scope='M_' + model,
                        pipeline_params=params,
                        raw_imgs_placeholder=raw_imgs)

                    try:
                        final_logits = tf.concat([final_logits, logits_out], 1)
                    except:
                        final_logits = logits_out

                final_logits_named = tf.identity(final_logits, name='final_logits')

            graph = tf.get_default_graph()

            raw_imgs_in_main_graph = tf.placeholder(tf.float32, shape=images_shape, name='raw_imgs')

            meta_graph_3 = tf.train.export_meta_graph(graph=graph_parent)
            meta_graph.import_scoped_meta_graph(meta_graph_3,
                                                input_map={"raw_imgs": raw_imgs_in_main_graph},
                                                import_scope='graph_parent')

            logits_name = [n.name for n in tf.get_default_graph().as_graph_def().node if 'final_logit' in n.name][0]
            logits_concat = graph.get_tensor_by_name(logits_name + ':0')

            with tf.Session(graph=tf.Graph()) as sess:
                tf.graph_util.convert_variables_to_constants(
                    sess,
                    tf.get_default_graph().as_graph_def(),
                    [v for v in tf.trainable_variables()]
                )

            return raw_imgs_in_main_graph, logits_concat

        @classmethod
        def build_finalGraph_and_return_final_tensors(cls,
                                                      hidden_units=None,
                                                      n_output=None,
                                                      primary_models_directory=None,
                                                      images_shape=None):
            def _new_weights(shape):
                with tf.name_scope('weights_ensemble'):
                    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))
                return weights

            def _new_biases(length):
                with tf.name_scope('biases_ensemble'):
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
            raw_imgs, concatenated_features = cls._combine_all_channel(
                models_directory=primary_models_directory, images_shape=images_shape)
            ##
            params_fc = {'hidden_units': hidden_units,
                         'n_output': n_output}
            ##
            graph_fc = tf.Graph()
            with graph_fc.as_default():
                X_tf = tf.placeholder(tf.float32, shape=[None, concatenated_features.get_shape().as_list()[1]],
                                      name='X_tf')

                ##
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
                                      num_outputs=params_fc['n_output'],
                                      use_relu=False,
                                      use_drop_out=False,
                                      name_scope='output_layer')

                logits_fc = tf.identity(logits, name='logits_tf')

            ##
            graph = tf.get_default_graph()

            meta_graph_3 = tf.train.export_meta_graph(graph=graph_fc)
            meta_graph.import_scoped_meta_graph(meta_graph_3,
                                                input_map={"X_tf": concatenated_features},
                                                import_scope='')

            logits_fc = graph.get_tensor_by_name('logits_tf:0')
            return raw_imgs, logits_fc

    ## to create the ensemble architecture from the primary models
    tf.reset_default_graph()
    ##
    _, _ = Ensemble.build_finalGraph_and_return_final_tensors(
        hidden_units=hidden_units,
        n_output=n_output,
        primary_models_directory=primary_models_directory,
        images_shape=images_shape)
    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, os.path.join(save_path, 'ensemble_architecture_' + 'main'))
    tf.reset_default_graph()
    return


def model_fn(features, labels, mode, params):

    graph_ensemble = tf.Graph()
    with tf.Session(graph=graph_ensemble) as sess:
        meta_graph_path = tf.gfile.Glob(os.path.join(params["ensemble_architecture_path"], '*.meta'))[0]
        loader = tf.train.import_meta_graph(meta_graph_path, clear_devices=True)
        loader.restore(sess, tf.train.latest_checkpoint(params["ensemble_architecture_path"]))

    graph = tf.get_default_graph()

    meta_graph_1 = tf.train.export_meta_graph(graph=graph_ensemble)
    meta_graph.import_scoped_meta_graph(meta_graph_1,
                                        input_map={"raw_imgs": features['img']},
                                        import_scope='main_graph')

    logits = graph.get_tensor_by_name('main_graph/logits_tf:0')

    predicted_classes = tf.argmax(logits, 1)

    category_map = tf.convert_to_tensor(params["category_map"])

    ##
    class_label = tf.gather_nd(category_map, predicted_classes)
    class_label = tf.convert_to_tensor([class_label], dtype=tf.string)

    ##
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
            'class_label': class_label[:, tf.newaxis],
            # 'category_map': tf.convert_to_tensor([str(params["category_map"])])[:, tf.newaxis],
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,
                                                               logits=logits,
                                                               name='cross_entropy')
    loss = tf.reduce_mean(cross_entropy, name='cost_fc')

    accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, 1),
                                   predictions=predicted_classes,
                                   name='acc_op')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN

    if params['retrain_primary_models'] != True:
        trainable_variables = [v for v in tf.trainable_variables() if 'ensemble' in v.name]
    else:
        trainable_variables = [v for v in tf.trainable_variables()]

    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'], name='adam_fc')
    train_op = optimizer.minimize(loss, var_list=trainable_variables,
                                  global_step=tf.train.get_or_create_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
