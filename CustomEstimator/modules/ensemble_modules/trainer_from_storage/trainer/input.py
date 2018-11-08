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
    def load_dataset(path):
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

        labels_unique = np.unique(imgs_paths_labels.target.values)

        category_mapper = {labels_unique[i]: i for i in range(len(labels_unique))}
        category_mapper_ext = {y: x for x, y in category_mapper.items()}

        for i in range(20):
            print('here0')

        category_mapper.update(category_mapper_ext)

        for i in range(20):
            print('here1')

        # with open(os.path.join(bin_path, 'category_mapper.json'), 'w') as outfile:
        #     json.dump(category_mapper, outfile)
        #
        # for i in range(20):
        #     print('here2')

        target = imgs_paths_labels['target'].map(category_mapper)

        for i in range(20):
            print('here3')

        # target = target.values

        print(target)

        # imgs_paths_labels.loc[:, 'target'] = target

        y_true_zeros = np.zeros((target.shape[0], labels_unique.shape[0]))
        # y_true = [y_true_zeros[i] for i in range(len(y_true_zeros))]
        for i in range(len(y_true_zeros)):
            y_true_zeros[i, target[i]] = 1

        y_true = y_true_zeros.copy()

        print(y_true)

        # for i in range(20):
        #     print('here1')
        #
        # y_true_cat = pd.DataFrame(data=imgs_paths_labels.loc[:, 'target'].values,
        #                           columns=['y_true'],
        #                           dtype='category')
        #
        # print(y_true_cat)
        #
        # # for i in range(20):
        # #     print('here2')
        # #
        # # MultiColumnOneHotEncoder = MultiColomnOneHotEncoder()
        #
        # for i in range(20):
        #     print('here3')
        #
        # onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
        #
        # # le = preprocessing.LabelEncoder()
        #
        # for i in range(20):
        #     print('here4')
        #
        # y_train = onehot_encoder.fit_transform(y_true_cat.y_true.reshape(-1, 1))
        #
        # # MultiColumnOneHotEncoder.fit(data=y_true_cat.loc[:, 'y_true'])
        #
        # for i in range(20):
        #     print('here5')
        #
        # # y_train = MultiColumnOneHotEncoder.transform(data=y_true_cat)
        #
        # print(y_train)

        # print(imgs_paths_labels['target'])
        # print(imgs_paths_labels['filenames'])

        # image_paths = [path.split("'")[1] for path in image_paths_b]

        # target = [str(path).split('/')[-1] for path in folders]
        ########################################
        # for cat in target:
        #     print(cat)
        ##########################
        # target = [path.split("'")[2] for path in targets_b]

        ##########################
        # data = load_files(path, load_content=False)
        # data = {'filenames': imgs_paths_labels.filenames.values, 'target': imgs_paths_labels.target.values}
        #
        # ## to save categories encoder
        # Dataset.dump_json_category_mapper(data=data, bin_path=bin_path)

        ##
        # guage_files = imgs_paths_labels.filenames.values
        # gauge_categories = imgs_paths_labels.target.values
        guage_files = imgs_paths_labels.filenames.values
        gauge_categories = y_true

        for i in range(20):
            print('here4')

        print(guage_files)
        print(gauge_categories)

        ##
        # Dataset.dump_pkl_MultiColProcessor(gauge_categories=gauge_categories, bin_path=bin_path)

        return guage_files, gauge_categories

    @classmethod
    def split_data_files(cls, ver_ratio, path, random_state, is_trial):
        ##
        guage_files, gauge_categories = cls.load_dataset(path=path)

        if is_trial:
            guage_files = guage_files[:20]
            gauge_categories = gauge_categories[:20]

            for i in range(20):
                print('here5')

        #
        X_train_path_names, X_test_path_names, y_train, y_test = \
            train_test_split(guage_files,
                             gauge_categories,
                             test_size=ver_ratio,
                             random_state=random_state)

        for i in range(20):
            print('here6')
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
    def prep_input_function(cls,
                            prefetch_buffer_size=None,
                            epochs_between_evals=None,
                            multi_threading=True,
                            train_batch_size=None,
                            mode=None,
                            X_train_path_names=None,
                            X_test_path_names=None,
                            y_train=None,
                            y_test=None):

        # X_train_path_names, X_test_path_names, y_train, y_test = \
        #     cls.split_data_files(ver_ratio=ver_ratio,
        #                          path=container_path,
        #                          random_state=random_state,
        #                          is_trial=is_trial,
        #                          bin_path=bin_path)
        ##
        ## OneHot encoding of y_train and y_test
        # with open(bin_path + '/MultiColProcessor.pkl', 'rb') as handle:
        #     MultiColumnOneHotEncoder = pickle.load(handle)

        # y_train_cat = pd.DataFrame(data=y_train, columns=['y_true'], dtype='category')
        # y_train = MultiColumnOneHotEncoder.transform(data=y_train_cat)
        # y_train = y_train.values
        # ##
        # y_test_cat = pd.DataFrame(data=y_test, columns=['y_true'], dtype='category')
        # y_test = MultiColumnOneHotEncoder.transform(data=y_test_cat)
        # y_test = y_test.values
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
