from sklearn.datasets import load_files
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
import pandas as pd
from MultiColProcessor import MultiColProcessor as mcp
import pickle
import json
from collections import OrderedDict
import os


class Dataset(object):
    def __init__(self):
        self.category_mapper = {}

    @staticmethod
    def dump_pkl_MultiColProcessor(gauge_categories, bin_path):
        y_true_cat = pd.DataFrame(data=gauge_categories, columns=['y_true'], dtype='category')
        MultiColumnOneHotEncoder = mcp.MultiColomnOneHotEncoder()
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
                      random_state=19,
                      is_trial=False,
                      bin_path='modules/primary_models_modules/dumps/'):
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
        with open(os.path.join(bin_path, 'MultiColProcessor.pkl'), 'rb') as handle:
            MultiColumnOneHotEncoder = pickle.load(handle)

        y_train_cat = pd.DataFrame(data=y_train, columns=['y_true'], dtype='category')
        y_train = MultiColumnOneHotEncoder.transform(data=y_train_cat)
        y_test_cat = pd.DataFrame(data=y_test, columns=['y_true'], dtype='category')
        y_test = MultiColumnOneHotEncoder.transform(data=y_test_cat)

        ##
        cls_indices = cls.find_cls_indices(train_y_data=y_train.values, test_y_data=y_test.values)

        print("The image pre-processing is complete.")

        return X_train, X_test, y_train.values, y_test.values, cls_indices