from sklearn.datasets import load_files
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from MultiColProcessor import MultiColProcessor as mcp
import pickle
import json
from collections import OrderedDict
import tensorflow as tf
import multiprocessing


class Dataset():
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

        if is_trial:
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
    def parse_function(filename, label):
        image_string = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [224, 224])
        image = tf.squeeze(image, 0)
        image = tf.image.convert_image_dtype(image, tf.float32)
        return {'X': image}, label

    @classmethod
    def prep_train_input_function(cls, ver_ratio, container_path,
                                  prefetch_buffer_size,
                                  color_mode='grayscale',
                                  epochs_between_evals=None,
                                  random_state=19,
                                  is_trial=False,
                                  bin_path=None,
                                  multi_threading=True,
                                  train_batch_size=None):
        X_train_path_names, _, y_train, _ = \
            cls.split_data_files(ver_ratio=ver_ratio,
                                 path=container_path,
                                 random_state=random_state,
                                 is_trial=is_trial,
                                 bin_path=bin_path)
        ##
        ## OneHot encoding of y_train and y_test
        with open(bin_path + '/MultiColProcessor.pkl', 'rb') as handle:
            MultiColumnOneHotEncoder = pickle.load(handle)

        y_train_cat = pd.DataFrame(data=y_train, columns=['y_true'], dtype='category')
        y_train = MultiColumnOneHotEncoder.transform(data=y_train_cat)
        y_train = y_train.values
        ##
        num_threads = multiprocessing.cpu_count() if multi_threading else 1

        X_train_path_names_tf = tf.constant(X_train_path_names)
        y_train_tf = tf.constant(y_train)

        dataset_train = tf.data.Dataset.from_tensor_slices((X_train_path_names_tf, y_train_tf))

        ##
        dataset_train = dataset_train.apply(
            tf.contrib.data.map_and_batch(
                map_func=cls.parse_function,
                batch_size=train_batch_size,
                num_parallel_batches=num_threads))

        dataset_train = dataset_train.\
            cache().shuffle(len(X_train_path_names) + 10).repeat(epochs_between_evals)#.batch(batch_size)
        #
        dataset_train = dataset_train.prefetch(buffer_size=prefetch_buffer_size)

        return dataset_train

    @classmethod
    def prep_eval_input_function(cls, ver_ratio, container_path,
                                 random_state=19,
                                 is_trial=False,
                                 bin_path=None,
                                 multi_threading=True):
        _, X_test_path_names, _, y_test = \
            cls.split_data_files(ver_ratio=ver_ratio,
                                 path=container_path,
                                 random_state=random_state,
                                 is_trial=is_trial,
                                 bin_path=bin_path)
        ##
        with open(bin_path + '/MultiColProcessor.pkl', 'rb') as handle:
            MultiColumnOneHotEncoder = pickle.load(handle)

        y_test_cat = pd.DataFrame(data=y_test, columns=['y_true'], dtype='category')
        y_test = MultiColumnOneHotEncoder.transform(data=y_test_cat)
        y_test = y_test.values
        ##
        X_test_path_names_tf = tf.constant(X_test_path_names)
        y_test_tf = tf.constant(y_test)

        num_threads = multiprocessing.cpu_count() if multi_threading else 1
        dataset_test = tf.data.Dataset.from_tensor_slices((X_test_path_names_tf, y_test_tf))

        dataset_test = dataset_test.map(map_func=cls.parse_function,
                                        num_parallel_calls=num_threads)

        dataset_test = dataset_test.batch(y_test.shape[0])

        return dataset_test
