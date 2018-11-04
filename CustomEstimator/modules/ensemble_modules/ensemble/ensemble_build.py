import glob
import numpy as np
import os
import tensorflow as tf
import time
from tensorflow.python.framework import meta_graph

from CustomEstimator.modules.primary_models_modules import LoadImg

X_train, X_test, y_train, y_test, cls_indices = LoadImg.Dataset.prep_datasets(
    ver_ratio=0.2, container_path='./CustomEstimator/data/ImageEveryUnit',
    final_img_width=224, final_img_height=224,
    color_mode="grayscale", random_state=1911,
    is_trial=True,
    bin_path='./CustomEstimator/modules/ensemble_modules/ensemble/bin/')


class Ensemble():
    @staticmethod
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

    @staticmethod
    def _combine_all_channel(models_directory=None, images_shape=None):
        graph_parent = tf.Graph()
        with graph_parent.as_default():
            raw_imgs = tf.placeholder(tf.float32, shape=[None, None, None, None], name='raw_imgs')
            for i, model in enumerate(os.listdir(models_directory)):
                checkpoint_path = glob.glob(os.path.join(models_directory, model))[0]
                checkpoint_file_path = glob.glob(os.path.join(checkpoint_path, '*.meta'))[0]
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

        # raw_imgs_name = [n.name for n in tf.get_default_graph().as_graph_def().node if 'raw_imgs' in n.name][0]
        # raw_imgs_tf = graph.get_tensor_by_name(raw_imgs_name + ':0')

        logits_name = [n.name for n in tf.get_default_graph().as_graph_def().node if 'final_logit' in n.name][0]
        logits_concat = graph.get_tensor_by_name(logits_name + ':0')

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
        params_fc = {'hidden_units': hidden_units.copy(),
                     'n_output': n_output}
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


def build_model_and_train(primary_models_directory='./CustomEstimator/logs/primary_models/',
                          writer_path='./CustomEstimator/modules/ensemble_modules/ensemble/trial/writer',
                          save_model_path='./CustomEstimator/modules/ensemble_modules/ensemble/trial/best_model_main',
                          images_shape=[None, 224, 224, 3],
                          hidden_units=[500, 100],
                          X_train=X_train, X_test=X_test,
                          y_train=y_train, y_test=y_test,
                          batch_size=4,
                          epochs=10):
    raw_imgs, logits_fc = Ensemble.build_finalGraph_and_return_final_tensors(
        hidden_units=hidden_units,
        n_output=y_train.shape[1],
        primary_models_directory=primary_models_directory,
        images_shape=images_shape)
    # tf.reset_default_graph()
    # graph = tf.get_default_graph()
    #
    # ## y
    # y_true_tf = tf.placeholder(tf.float64, shape=[None, y_train.shape[1]],
    #                            name='y_true_tf')
    #
    # y_pred = tf.nn.softmax(logits_fc)
    # y_pred_cls = tf.argmax(y_pred, axis=1)
    # y_true_cls = tf.argmax(y_true_tf, axis=1)
    # correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #
    # ## dataset
    # dataset = tf.data.Dataset.from_tensor_slices((raw_imgs, y_true_tf))
    # dataset = dataset.shuffle(buffer_size=100)
    # dataset = dataset.batch(batch_size)
    # dataset = dataset.repeat(epochs)
    # iter = dataset.make_initializable_iterator()
    # # iter = dataset.make_one_shot_iterator()
    # get_batch = iter.get_next()
    #
    # ## cost
    # cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_fc,
    #                                                   labels=y_true_tf,
    #                                                   name='cost_fc')
    #
    # trainable_variables = [v for v in tf.trainable_variables() if 'ensemble' in v.name]
    # optimizer = tf.train.AdamOptimizer(learning_rate=3e-4, name='adam_fc'). \
    #     minimize(cost, var_list=trainable_variables)
    # # variables_list = [n.name for n in tf.get_default_graph().as_graph_def().node if 'raw_imgs' in n.name]
    #
    # feed_dict_train = {raw_imgs: X_train,
    #                    y_true_tf: y_train}
    #
    # feed_dict_test = {raw_imgs: X_test,
    #                   y_true_tf: y_test}

    # Start-time used for printing time-usage below.
    # start_time = time.time()

    # with tf.Session() as session:
    #     # initialize weights and biases variables
    #     session.run(tf.global_variables_initializer())
        # session.run(iter.initializer, feed_dict=feed_dict_train)
        # writer = tf.summary.FileWriter(writer_path, session.graph)

        # best_validation_accuracy=0
    #
    #     for i in range(200):
    #         try:
    #             print('batch {} ... '.format(i))
    #             Xydata = session.run(get_batch)
    #             session.run(optimizer, feed_dict={raw_imgs: Xydata[0],
    #                                               y_true_tf: Xydata[1]})
    #             accuracy_train = session.run(accuracy, feed_dict=feed_dict_train)
    #             print('train_acc = {} ... '.format(accuracy_train))
    #             accuracy_validation = session.run(accuracy, feed_dict=feed_dict_test)
    #             print('validation_acc = {} ... '.format(accuracy_validation))
    #             if accuracy_validation > best_validation_accuracy:
    #                 saver = tf.train.Saver(max_to_keep=1)
    #                 saver.save(session, save_model_path)
    #         except:
    #             break
    #
    #     writer.close()
    #
    # end_time = time.time()
    #
    # print("Training took {}.".format(end_time-start_time))

    return

build_model_and_train()

graph_pred = tf.get_default_graph()
a=tf.trainable_variables()
trainable_variables = [v for v in tf.trainable_variables() if 'logits_tf' in v.name]



with tf.Session() as session:
    saver_3 = tf.train.import_meta_graph('./CustomEstimator/modules/ensemble_modules/ensemble/logs/primary_models/1st/best_model_main.meta',
                                         clear_devices=True)
    saver_3.restore(tf.get_default_session(),
                    tf.train.latest_checkpoint('./CustomEstimator/modules/ensemble_modules/ensemble/logs/primary_models/1st/'))
    graph_pred = tf.get_default_graph()
    raw_imgs = graph_pred.get_tensor_by_name("raw_imgs:0")
    logits_tf = graph_pred.get_tensor_by_name("logits_tf:0")
    b = tf.trainable_variables()

    c=b[0].eval()

    ##
    feed_dict_pred = {raw_imgs: X_train}

    ##
    # logits = session.run(b[0])
