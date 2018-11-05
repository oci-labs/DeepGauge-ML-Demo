import tensorflow as tf
from modules.NeuralNet import Graph
from modules.PerfMeasures import Measures
import time
from datetime import timedelta
import pickle
import json
from collections import OrderedDict
import os


class OptimizerLogger(object):
    def __init__(self):
        self.modelsInfo = dict()
        self.category_mapper = dict()

    @staticmethod
    def initialize():

        ##
        if not os.path.exists('./logs/TensorBoard'):
            os.makedirs('./logs/TensorBoard')

        ##
        try:
            #
            with open('dumps/category_mapper.json') as handle:
                category_mapper = json.load(handle)
            #
            OptimizerLogger.category_mapper = category_mapper

        except:
            print('Error: the category_mapper json file does not exist.')

        ##
        try:
            with open('dumps/best_models_info.json') as handle:
                best_models_info = json.load(handle)
            OptimizerLogger.modelsInfo = best_models_info

        except:
            # logFolders = {cat: './logs/models/' + str(cat)
            #               for _, cat in category_mapper['num_to_cat'].items()}
            # logFolders['main'] = './logs/models/main'
            #
            # #
            # for _, values in logFolders.items():
            #     if not os.path.exists(values):
            #         os.makedirs(values)

            ##
            modelsInfo = OrderedDict()

            for _, value in category_mapper['num_to_cat'].items():
                modelsInfo[value] = {#'folder': logFolders[value],
                                     'best_acc': 0,
                                     'best_median': 0,
                                     'class_logits': [],
                                     'hyper_params': OrderedDict()
                                     }

            modelsInfo['main'] = {#'folder': logFolders['main'],
                                  'best_acc': 0,
                                  'best_median': 0,
                                  'class_logits': [],
                                  'hyper_params': OrderedDict()
                                  }

            OptimizerLogger.modelsInfo = modelsInfo

        return

    @staticmethod
    def optimize_and_save_logs(optimizer, saver, logits_tf, num_iterations, meta_dict,
                               device_name, models_log_path, keep_best_model):
        ##
        writer = tf.summary.FileWriter('./logs/TensorBoard')
        writer.add_graph(tf.get_default_graph())
        ##
        # Put the data into a dict with the proper names
        # for placeholder variables in the tensorflow graph.
        X_tf = meta_dict['X_tf']
        y_true_tf = meta_dict['y_true_tf']
        feed_dict_train = {X_tf: meta_dict['X_train'],
                           y_true_tf: meta_dict['y_train']}

        feed_dict_test = {X_tf: meta_dict['X_test'],
                          y_true_tf: meta_dict['y_test']}

        ##
        y_train = meta_dict['y_train']
        y_test = meta_dict['y_test']

        ## initial variables
        max_test_accuracy = 0
        total_iterations = 0

        # Start-time used for printing time-usage below.
        start_time = time.time()

        with tf.device(device_name):
            with tf.Session() as session:
                ## initialize weights and biases variables
                session.run(tf.global_variables_initializer())

                for i in range(total_iterations,
                               total_iterations + num_iterations):
                    print('epoch {} ... '.format(i))
                    # TensorFlow assigns the variables in feed_dict_train
                    # to the placeholder variables and then runs the optimizer.
                    session.run(optimizer, feed_dict=feed_dict_train)

                    logits_train = session.run(logits_tf, feed_dict=feed_dict_train)
                    logits_test = session.run(logits_tf, feed_dict=feed_dict_test)

                    #########################################
                    train_acc, test_acc, classes_info = Measures.compute_measures(logits_train=logits_train,
                                                                                  y_train=y_train,
                                                                                  logits_test=logits_test,
                                                                                  y_test=y_test,
                                                                                  meta_dict=meta_dict)

                    #
                    modelsInfo = Measures.log_best_models(classes_info=classes_info,
                                                          best_models_info=OptimizerLogger.modelsInfo,
                                                          session=session,
                                                          saver=saver,
                                                          meta_dict=meta_dict,
                                                          models_log_path=models_log_path,
                                                          keep_best_model=keep_best_model)

                    #
                    OptimizerLogger.modelsInfo = modelsInfo

                    ##
                    # modelsInfo['main']['best_median']
                    print('train_acc: {:.4f}   test_acc: {:.4f}  best_logits_median: {:.4f}'.
                          format(train_acc, test_acc, modelsInfo['main']['best_median']))
                    print('')

                ## to save modelsInfo
                with open('./dumps/best_models_info.pkl', 'wb') as pklFile:
                    pickle.dump(modelsInfo, pklFile)

        tf.reset_default_graph()
        ###########################################

        # Update the total number of iterations performed.
        total_iterations += num_iterations

        # Ending time.
        end_time = time.time()

        # Difference between start and end-times.
        time_dif = end_time - start_time

        # Print the time-usage.
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
        return

    @staticmethod
    def find_num_of_classes(y_data_shape):
        ## find num_classes from y_train or y_test shape
        num_classes = y_data_shape[1]
        return num_classes

    @staticmethod
    def find_num_of_channels(X_data_shape):
        ## find num_channels from X_train or X_test shape
        num_channels = X_data_shape[3]
        return num_channels

    @staticmethod
    def find_img_size(X_data_shape):
        ## find img_size from X_train or X_test shape
        img_size_x = X_data_shape[1]
        img_size_y = X_data_shape[2]
        return img_size_x, img_size_y

    @classmethod
    def train_and_save_logs_three_CNN(cls,
                                      filter_size1, num_filters1, strides_1,
                                      use_pooling_1, pooling_ksize_1, pooling_strides_1,
                                      ##
                                      filter_size2, num_filters2, strides_2,
                                      use_pooling_2, pooling_ksize_2, pooling_strides_2,
                                      ##
                                      filter_size3, num_filters3, strides_3,
                                      use_pooling_3, pooling_ksize_3, pooling_strides_3,
                                      ##
                                      fc_size, num_iterations,
                                      learning_rate, momentum,
                                      X_train, y_train,
                                      X_test, y_test,
                                      cls_indices, padding,
                                      models_log_path,
                                      device_name, keep_best_model):

        ## find num_classes
        num_classes = cls.find_num_of_classes(y_data_shape=y_train.shape)

        ## find num_channels
        num_channels = cls.find_num_of_channels(X_data_shape=X_train.shape)

        ## find img_size
        img_size_x, img_size_y = cls.find_img_size(X_data_shape=X_train.shape)

        ## placeholders
        X_tf = tf.placeholder(tf.float32, shape=[None, img_size_x, img_size_y, num_channels],
                              name='X_image_tf')
        y_true_tf = tf.placeholder(tf.float64, shape=[None, num_classes],
                                   name='y_true_tf')

        ## get fully connected last layer
        last_layer_fc = Graph.graph_three_conv_layer(x_image=X_tf,
                                                     num_classes=num_classes,
                                                     filter_size1=filter_size1, num_filters1=num_filters1,
                                                     strides_1=strides_1, use_pooling_1=use_pooling_1,
                                                     pooling_ksize_1=pooling_ksize_1,
                                                     pooling_strides_1=pooling_strides_1,
                                                     filter_size2=filter_size2, num_filters2=num_filters2,
                                                     strides_2=strides_2, use_pooling_2=use_pooling_2,
                                                     pooling_ksize_2=pooling_ksize_2,
                                                     pooling_strides_2=pooling_strides_2,
                                                     filter_size3=filter_size3, num_filters3=num_filters3,
                                                     strides_3=strides_3, use_pooling_3=use_pooling_3,
                                                     pooling_ksize_3=pooling_ksize_3,
                                                     pooling_strides_3=pooling_strides_3,
                                                     fc_size=fc_size, padding=padding,
                                                     num_channels=num_channels)

        ##
        logits_tf = tf.nn.softmax(last_layer_fc, name='logits_tf')

        ## create cost function
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=last_layer_fc, labels=y_true_tf)
        cost = tf.reduce_mean(cross_entropy)

        ## initiate optimizer
        # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
        #                                        momentum=momentum).minimize(cost)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        ##
        saver = tf.train.Saver(max_to_keep=1000000)

        ##
        hyper_params = OrderedDict({'filter_size1': filter_size1, 'num_filters1': num_filters1,
                                    'strides_1': strides_1,
                                    'filter_size2': filter_size2, 'num_filters2': num_filters2,
                                    'strides_2': strides_2,
                                    'filter_size3': filter_size3, 'num_filters3': num_filters3,
                                    'strides_3': strides_3,
                                    'fc_size': fc_size,
                                    'num_iterations': num_iterations,
                                    'learning_rate': learning_rate,
                                    'momentum': momentum,
                                    'use_pooling_1': use_pooling_1,
                                    'pooling_ksize_1': pooling_ksize_1,
                                    'pooling_strides_1': pooling_strides_1,
                                    'use_pooling_2': use_pooling_2,
                                    'pooling_ksize_2': pooling_ksize_2,
                                    'pooling_strides_2': pooling_strides_2,
                                    'use_pooling_3': use_pooling_3,
                                    'pooling_ksize_3': pooling_ksize_3,
                                    'pooling_strides_3': pooling_strides_3,
                                    'padding': padding,
                                    'img_size_x': img_size_x,
                                    'img_size_y': img_size_y,
                                    'num_channels': num_channels})

        ##
        meta_dict = dict()
        meta_dict['hyper_params'] = hyper_params
        meta_dict['X_train'] = X_train
        meta_dict['y_train'] = y_train
        meta_dict['X_test'] = X_test
        meta_dict['y_test'] = y_test
        meta_dict['cls_indices'] = cls_indices
        meta_dict['X_tf'] = X_tf
        meta_dict['y_true_tf'] = y_true_tf
        meta_dict['num_classes'] = num_classes

        ##
        cls.initialize()  # num_classes=meta_dict['num_classes'])

        ##
        cls.optimize_and_save_logs(optimizer=optimizer,
                                   saver=saver,
                                   logits_tf=logits_tf,
                                   num_iterations=num_iterations,
                                   meta_dict=meta_dict,
                                   models_log_path=models_log_path,
                                   device_name=device_name,
                                   keep_best_model=keep_best_model)

        return

    @classmethod
    def train_and_save_two_fc_ensemble(cls,
                                       ##
                                       fc_size_1, fc_size_2,
                                       use_drop_out_1, use_drop_out_2,
                                       ##
                                       num_iterations,
                                       learning_rate, momentum,
                                       X_train, y_train,
                                       X_test, y_test,
                                       cls_indices,
                                       models_log_path,
                                       device_name):

        ## find num_classes
        num_classes = cls.find_num_of_classes(y_data_shape=y_train.shape)
        x_length = X_train.shape[1]

        ## placeholders
        X_tf = tf.placeholder(tf.float32, shape=[None, x_length], name='X_tf')
        y_true_tf = tf.placeholder(tf.float64, shape=[None, num_classes], name='y_true_tf')

        ## get fully connected last layer
        last_layer_fc = Graph.graph_two_fc_layer(x_appended=X_tf, num_features=x_length,
                                                 num_classes=num_classes,
                                                 fc_size_1=fc_size_1, fc_size_2=fc_size_2,
                                                 use_drop_out_1=use_drop_out_1,
                                                 use_drop_out_2=use_drop_out_2)

        ##
        logits_tf = tf.nn.softmax(last_layer_fc, name='logits_tf')

        ## create cost function
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=last_layer_fc, labels=y_true_tf)
        cost = tf.reduce_mean(cross_entropy)

        ## initiate optimizer
        # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
        #                                        momentum=momentum).minimize(cost)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        ##
        saver = tf.train.Saver(max_to_keep=1000000)

        ##
        hyper_params = OrderedDict({'fc_size_1': fc_size_1,
                                    'fc_size_2': fc_size_2,
                                    'use_drop_out_1': use_drop_out_1,
                                    'use_drop_out_2': use_drop_out_2,
                                    'num_iterations': num_iterations,
                                    'learning_rate': learning_rate,
                                    'momentum': momentum})

        ##
        meta_dict = dict()
        meta_dict['hyper_params'] = hyper_params
        meta_dict['X_train'] = X_train
        meta_dict['y_train'] = y_train
        meta_dict['X_test'] = X_test
        meta_dict['y_test'] = y_test
        meta_dict['cls_indices'] = cls_indices
        meta_dict['X_tf'] = X_tf
        meta_dict['y_true_tf'] = y_true_tf
        meta_dict['num_classes'] = num_classes

        ##
        cls.initialize()  # num_classes=meta_dict['num_classes'])

        ##
        cls.optimize_and_save_logs(optimizer=optimizer,
                                   saver=saver,
                                   logits_tf=logits_tf,
                                   num_iterations=num_iterations,
                                   meta_dict=meta_dict,
                                   models_log_path=models_log_path,
                                   device_name=device_name)

        return
