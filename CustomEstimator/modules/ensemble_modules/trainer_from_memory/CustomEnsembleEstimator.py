from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
from modules import LoadImg
import glob
import os
from tensorflow.python.framework import meta_graph
import numpy as np

from sklearn.datasets import load_files
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
import pandas as pd
from MultiColProcessor import MultiColProcessor as mcp
import pickle
import json
from collections import OrderedDict


########################
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_images', default='data/ImageEveryUnit',
                        type=str,
                        help='path to images (e.g. gs://...)')
    parser.add_argument('--primary_models_directory',
                        default='./logs/primary_models/',
                        type=str)
    parser.add_argument('--images_shape',
                        default='[None, 224, 224, 3]',
                        type=str)
    parser.add_argument('--hidden_units',
                        default='[500, 100]',
                        type=str)
    parser.add_argument('--learning_rate',
                        default=3e4,
                        type=float)
    parser.add_argument('--retrain_primary_models',
                        default='False',
                        type=str)
    parser.add_argument('--batch_size',
                        default=6,
                        type=int)
    parser.add_argument('--train_epochs',
                        default=1,
                        type=int)
    parser.add_argument('--epochs_between_evals',
                        default=1,
                        type=int)
    parser.add_argument('--export_dir',
                        default='./logs/exported_model',
                        type=str)
    parser.add_argument('--ensemble_architecture_path',
                        default='./logs/ensemble_graph/',
                        type=str)
    parser.add_argument('--metric',
                        default='accuracy',
                        type=str)
    parser.add_argument('--bin_path',
                        default='dumps/',
                        type=str)
    parser.add_argument('--dev',
                        default='False',
                        type=str)
    parser.add_argument('--color_mode',
                        default='grayscale',
                        type=str)
    parser.add_argument('--random_state',
                        default=1911,
                        type=int)
    return parser


parser = create_parser()


class Dataset():
    def __init__(self):
        self.category_mapper = {}

    @staticmethod
    def dump_pkl_MultiColProcessor(gauge_categories, bin_path):
        y_true_cat = pd.DataFrame(data=gauge_categories, columns=['y_true'], dtype='category')
        MultiColumnOneHotEncoder = mcp.MultiColomnOneHotEncoder()
        MultiColumnOneHotEncoder.fit(data=y_true_cat)
        with open(bin_path + 'MultiColProcessor.pkl', 'wb') as pklFile:
            pickle.dump(MultiColumnOneHotEncoder, pklFile)
        return

    @staticmethod
    def dump_json_category_mapper(data, bin_path):
        ##
        category_mapper = OrderedDict()
        category_mapper_num_to_cat = OrderedDict()
        category_mapper_cat_to_num = OrderedDict()
        for i in range(len(data['target_names'])):
            category_mapper_num_to_cat[i] = data['target_names'][i]
        for i in range(len(data['target_names'])):
            category_mapper_cat_to_num[data['target_names'][i]] = i

        ##
        category_mapper['num_to_cat'] = category_mapper_num_to_cat
        category_mapper['cat_to_num'] = category_mapper_cat_to_num
        category_mapper['num_classes'] = len(data['target_names'])

        Dataset.category_mapper = category_mapper

        with open(bin_path + 'category_mapper.json', 'w') as outfile:
            json.dump(category_mapper, outfile)
        return

    @staticmethod
    def load_dataset(path, bin_path):
        data = load_files(path, load_content=False)

        ## to save categories encoder
        Dataset.dump_json_category_mapper(data=data, bin_path=bin_path)

        ##
        guage_files = np.array(data['filenames'])
        gauge_categories = np.array(data['target'])

        ##
        Dataset.dump_pkl_MultiColProcessor(gauge_categories=gauge_categories, bin_path=bin_path)

        return guage_files, gauge_categories

    @staticmethod
    def split_data_files(ver_ratio, path, random_state, is_trial, bin_path):
        ##
        guage_files, gauge_categories = Dataset.load_dataset(path=path, bin_path=bin_path)

        if is_trial == True:
            guage_files = guage_files[:20]
            gauge_categories = gauge_categories[:20]

        #
        X_train_path_names, X_test_path_names, y_train, y_test = \
            train_test_split(guage_files,
                             gauge_categories,
                             test_size=ver_ratio,
                             random_state=random_state)
        ##
        return X_train_path_names, X_test_path_names, y_train, y_test

    @staticmethod
    def path_to_tensor(img_path, img_size_x, img_size_y, color_mode):
        if img_size_x is None:
            img_size_x = 224
        if img_size_y is None:
            img_size_y = 224
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(img_size_y, img_size_x, color_mode))
        # convert PIL.Image.Image type to 3D tensor with shape (?, ?, ?)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, ?, ?, ?) and return 4D tensor
        return np.expand_dims(x, axis=0)

    @staticmethod
    def paths_to_tensor(imgs_path, img_size_x, img_size_y, color_mode):
        list_of_tensors = [Dataset.path_to_tensor(img_path, img_size_x, img_size_y, color_mode)
                           for img_path in list(imgs_path)]
        return np.vstack(list_of_tensors)

    @staticmethod
    def find_cls_indices(train_y_data, test_y_data):
        ##
        num_classes = Dataset.category_mapper['num_classes']

        ##
        train_indices = OrderedDict()
        test_indices = OrderedDict()
        for cls in range(num_classes):
            label = Dataset.category_mapper['num_to_cat'][cls]
            train_indices[label] = train_y_data[:, cls] == True
            test_indices[label] = test_y_data[:, cls] == True

        ##
        cls_indices = OrderedDict()
        cls_indices['train_indices'] = train_indices
        cls_indices['test_indices'] = test_indices

        return cls_indices

    @classmethod
    def prep_datasets(cls, ver_ratio, container_path, final_img_width, final_img_height,
                      color_mode='grayscale',
                      random_state=19, is_trial=None, bin_path='dumps/'):
        ##
        X_train_path_names, X_test_path_names, y_train, y_test = \
            cls.split_data_files(ver_ratio=ver_ratio,
                                 path=container_path,
                                 random_state=random_state,
                                 is_trial=is_trial,
                                 bin_path=bin_path)

        X_train = cls.paths_to_tensor(imgs_path=X_train_path_names,
                                      img_size_x=final_img_width,
                                      img_size_y=final_img_height,
                                      color_mode=color_mode)

        X_test = cls.paths_to_tensor(imgs_path=X_test_path_names,
                                     img_size_x=final_img_width,
                                     img_size_y=final_img_height,
                                     color_mode=color_mode)

        ## OneHot encoding of y_train and y_test
        with open('dumps/MultiColProcessor.pkl', 'rb') as handle:
            MultiColumnOneHotEncoder = pickle.load(handle)

        y_train_cat = pd.DataFrame(data=y_train, columns=['y_true'], dtype='category')
        y_train = MultiColumnOneHotEncoder.transform(data=y_train_cat)
        y_test_cat = pd.DataFrame(data=y_test, columns=['y_true'], dtype='category')
        y_test = MultiColumnOneHotEncoder.transform(data=y_test_cat)

        ##
        cls_indices = cls.find_cls_indices(train_y_data=y_train.values, test_y_data=y_test.values)

        print("The image pre-processing is complete.")

        return X_train, X_test, y_train.values, y_test.values, cls_indices


def input_fn_train(raw_imgs, labels, batch_size, epochs_between_evals):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(({'X': raw_imgs}, labels))
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.cache().shuffle(100000).repeat(epochs_between_evals).batch(batch_size)
    # Return the read end of the pipeline.
    return dataset  # dataset #dataset.make_one_shot_iterator().get_next()


def input_fn_eval(raw_imgs, labels):
    dataset = tf.data.Dataset.from_tensor_slices(({'X': raw_imgs}, labels))
    dataset = dataset.batch(raw_imgs.shape[0])
    return dataset.make_one_shot_iterator().get_next()


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
                X_tf = tf.placeholder(tf.float32, shape=[None, concatenated_features.get_shape().as_list()[1]],
                                      name='X_tf')

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


def ensemble_model(features, labels, mode, params):
    ######### dev
    # params = {'primary_models_directory': './logs/primary_models/',
    #           'writer_path': './trial/writer',
    #           'save_model_path': './trial/best_model_main',
    #           'number_of_categories': labels.shape[1],
    #           'images_shape': [None, 224, 224, 3],
    #           'hidden_units': [500, 100],
    #           'n_output': labels.shape[1],
    #           'epochs': 10,
    #           'learning_rate': 3e-4}

    #########

    graph_ensemble = tf.Graph()
    with tf.Session(graph=graph_ensemble) as sess:
        meta_graph_path = glob.glob(os.path.join(params["ensemble_architecture_path"], '*.meta'))[0]
        loader = tf.train.import_meta_graph(meta_graph_path, clear_devices=True)
        loader.restore(sess, tf.train.latest_checkpoint(params["ensemble_architecture_path"]))

    graph = tf.get_default_graph()

    meta_graph_0 = tf.train.export_meta_graph(graph=graph_ensemble)
    meta_graph.import_scoped_meta_graph(meta_graph_0,
                                        input_map={"raw_imgs": features['X']},
                                        import_scope='main_graph')

    logits = graph.get_tensor_by_name('main_graph/logits_tf:0')

    predicted_classes = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
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

    ###### trial
    print('there are {} DDDDDDDDDDDD'.format(len(trainable_variables)))
    ######

    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'], name='adam_fc')
    train_op = optimizer.minimize(loss, var_list=trainable_variables,
                                  global_step=tf.train.get_or_create_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):
    args = parser.parse_args(argv[1:])

    ##
    images_shape = eval(args.images_shape)
    path_to_images = args.path_to_images
    color_mode = args.color_mode
    bin_path = args.bin_path
    is_trial = args.dev == 'True'
    primary_models_directory = args.primary_models_directory
    hidden_units = eval(args.hidden_units)
    ensemble_architecture_path = args.ensemble_architecture_path
    learning_rate = args.learning_rate
    retrain_primary_models = args.retrain_primary_models == 'True'
    train_epochs = args.train_epochs
    epochs_between_evals = args.epochs_between_evals
    batch_size = args.batch_size
    export_dir = args.export_dir
    metric = args.metric
    ##

    # Fetch the data
    print('loading image data ...')
    X_train, X_test, y_train, y_test, _ = Dataset.prep_datasets(
        ver_ratio=0.2, container_path=path_to_images,
        final_img_width=images_shape[2], final_img_height=images_shape[1],
        color_mode=color_mode, random_state=1911, bin_path=bin_path, is_trial=is_trial)

    create_ensemble_architecture(hidden_units=hidden_units,
                                 n_output=y_train.shape[1],
                                 primary_models_directory=primary_models_directory,
                                 images_shape=images_shape,
                                 save_path=ensemble_architecture_path)

    classifier = tf.estimator.Estimator(
        model_fn=ensemble_model,
        params={
            'primary_models_directory': primary_models_directory,
            'images_shape': images_shape,
            'hidden_units': hidden_units,
            'learning_rate': learning_rate,
            'ensemble_architecture_path': ensemble_architecture_path,
            'retrain_primary_models': retrain_primary_models
        })

    # Train and evaluate model.
    model_criteria = 0.0
    image = tf.placeholder(tf.float32, shape=images_shape, name='export_input_image')
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({'X': image})
    ##
    for i in range(train_epochs // epochs_between_evals):
        print('epoch {} .........'.format(i * epochs_between_evals + 1))
        classifier.train(
            input_fn=lambda: input_fn_train(
                raw_imgs=X_train, labels=y_train,
                batch_size=batch_size,
                epochs_between_evals=epochs_between_evals))

        eval_result = classifier.evaluate(
            input_fn=lambda: input_fn_eval(raw_imgs=X_test, labels=y_test))
        print('')
        print('')
        print('current validation dataset accuracy is: {}'.format(eval_result['accuracy']))
        print('current highest validation dataset accuracy is: {}'.format(model_criteria))
        if eval_result[metric] > model_criteria:
            model_criteria = eval_result[metric].copy()
            print('current highest validation dataset accuracy updated to: {}'.format(model_criteria))
            print('')
            print('')
            classifier.export_savedmodel(export_dir, input_fn, strip_default_attrs=True)
            print('model updated')
            continue
        print('')
        print('')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
