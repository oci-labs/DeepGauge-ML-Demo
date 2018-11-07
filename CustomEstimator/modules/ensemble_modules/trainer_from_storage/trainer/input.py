from sklearn.datasets import load_files
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
# from MultiColProcessor import MultiColProcessor as mcp
import pickle
import json
from collections import OrderedDict
import tensorflow as tf
import multiprocessing
import os
from sklearn import preprocessing


##
class MultiColomnOneHotEncoder:
    ##
    def __init__(self):
        self.__catColumns = []
        self.__MultiOHE = {}

    ##
    def __getCategoryColomns(self, data=pd.DataFrame()):
        catColumns = []
        for i, j in enumerate(data):
            if (data.dtypes[i].name == 'category'):
                catColumns.append(j)
            else:
                continue
        ##
        self.__catColumns = catColumns
        ##
        return

    ##
    def fit(self, data):
        ##
        self.__getCategoryColomns(data)
        ##
        for col in self.__catColumns:
            OneHotEncoder = preprocessing.OneHotEncoder(sparse=False)
            OneHotEncoder.fit(np.array(data.loc[:, col]).reshape(-1, 1))
            self.__MultiOHE[col] = OneHotEncoder
        ##
        return self

    def transform(self, data):

        ##
        catData = data[self.__catColumns]
        data = data.drop(self.__catColumns, axis=1)

        ##
        def Transform_Rec(dta=catData):
            ##
            nCol = dta.shape[1]
            ##
            if nCol == 1:
                ##
                col = dta.columns[0]
                OneHotEncoder = self.__MultiOHE[col]
                transformed = OneHotEncoder.transform(np.array(dta.loc[:, col]).reshape(-1, 1))
                transformed = pd.DataFrame(transformed)
                transformed.columns = [str(col) + '_' + str(c) for c in transformed.columns]
                ##
                return transformed

            else:
                ##
                if (nCol % 2 == 0):
                    middle_index = int(nCol / 2)
                else:
                    middle_index = int(nCol / 2 - 0.5)
                ##
                left = dta.iloc[:, :middle_index]
                right = dta.iloc[:, middle_index:]
                ##
                return pd.concat([Transform_Rec(dta=left), Transform_Rec(dta=right)], axis=1)

        ##
        transformedCatData = Transform_Rec(dta=catData)
        transformedCatData.set_index(data.index, inplace=True)

        ##
        return pd.concat([data, transformedCatData], axis=1)


###################################
class Dataset():
    @staticmethod
    def dump_pkl_MultiColProcessor(gauge_categories, bin_path):
        y_true_cat = pd.DataFrame(data=gauge_categories, columns=['y_true'], dtype='category')
        MultiColumnOneHotEncoder = MultiColomnOneHotEncoder()
        MultiColumnOneHotEncoder.fit(data=y_true_cat)
        with open(os.path.join(bin_path, 'MultiColProcessor.pkl'), 'wb') as pklFile:
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

        with open(os.path.join(bin_path, 'category_mapper.json'), 'w') as outfile:
            json.dump(category_mapper, outfile)
        return

    @staticmethod
    def load_dataset(path, bin_path):
        ####################
        directory = path
        images_folders = tf.train.match_filenames_once(directory + '/*')

        init = (tf.global_variables_initializer(), tf.local_variables_initializer())

        with tf.Session() as sess:
            sess.run(init)
            folders = sess.run(images_folders)

        folders = [str(path) for path in folders]

        # #####################################
        # for path in folders:
        #     print(path)
        #######################################

        imgs_paths_labels = pd.DataFrame(columns=['filenames', 'target'])
        for f in folders:
            with tf.Session() as sess:
                images = tf.train.match_filenames_once(f + '/*.*')
                sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
                image_paths = sess.run(images)
                ##
                imgs_paths_labels_current = pd.DataFrame(columns=['filenames', 'target'])
                image_paths = [str(p) for p in image_paths]
                imgs_paths_labels_current.loc[:, 'filenames'] = image_paths
                # target_list = [str(f).split('/')[-1] for range(len(image_paths))]
                imgs_paths_labels_current.loc[:, 'target'] = str(f).split('/')[-1]
                # for img in image_paths:
                #     print(img)
                imgs_paths_labels = imgs_paths_labels.append(imgs_paths_labels_current)

        imgs_paths_labels.reset_index(drop=True, inplace=True)

        print(imgs_paths_labels['target'])
        print(imgs_paths_labels['filenames'])

        # image_paths = [path.split("'")[1] for path in image_paths_b]

        # target = [str(path).split('/')[-1] for path in folders]
        ########################################
        # for cat in target:
        #     print(cat)
        ##########################
        # target = [path.split("'")[2] for path in targets_b]

        ##########################
        # data = load_files(path, load_content=False)
        data = {'filenames': imgs_paths_labels.filenames.values, 'target': imgs_paths_labels.target.values}

        ## to save categories encoder
        Dataset.dump_json_category_mapper(data=data, bin_path=bin_path)

        ##
        # guage_files = imgs_paths_labels.filenames.values
        # gauge_categories = imgs_paths_labels.target.values
        guage_files = data['filenames']
        gauge_categories = data['target']

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
        filename = filename['file']
        image_string = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize_images(image, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = tf.image.convert_image_dtype(image, tf.float32)
        return {'X': image}, label

    @classmethod
    def prep_input_function(cls, ver_ratio, container_path,
                            prefetch_buffer_size=None,
                            color_mode='grayscale',
                            epochs_between_evals=None,
                            random_state=19,
                            is_trial=False,
                            bin_path=None,
                            multi_threading=True,
                            train_batch_size=None,
                            mode=None):

        X_train_path_names, X_test_path_names, y_train, y_test = \
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
        y_test_cat = pd.DataFrame(data=y_test, columns=['y_true'], dtype='category')
        y_test = MultiColumnOneHotEncoder.transform(data=y_test_cat)
        y_test = y_test.values
        ##
        num_threads = multiprocessing.cpu_count() if multi_threading else 1

        if mode == tf.estimator.ModeKeys.TRAIN:

            dataset_train = tf.data.Dataset.from_tensor_slices(({'file': X_train_path_names}, y_train))

            ##
            dataset_train = dataset_train.apply(
                tf.contrib.data.map_and_batch(
                    map_func=cls.parse_function,
                    batch_size=train_batch_size,
                    num_parallel_batches=num_threads))

            dataset_train = dataset_train. \
                cache().shuffle(len(X_train_path_names) + 10).repeat(epochs_between_evals)  # .batch(batch_size)
            #
            dataset_final = dataset_train.prefetch(buffer_size=prefetch_buffer_size)

        elif mode == tf.estimator.ModeKeys.EVAL:

            dataset_test = tf.data.Dataset.from_tensor_slices(({'file': X_test_path_names}, y_test))

            dataset_test = dataset_test.map(map_func=cls.parse_function,
                                            num_parallel_calls=num_threads)

            dataset_final = dataset_test.batch(y_test.shape[0])

        return dataset_final
