from sklearn.datasets import load_files
import numpy as np
from keras.preprocessing import image
import pandas as pd
import pickle
import json
from collections import OrderedDict


class DatasetForPrediction(object):

    def __init__(self):
        self.category_mapper = dict()
        self.MultiColumnOneHotEncoder = object()

    @staticmethod
    def initialize():
        ##
        with open('./dumps/MultiColProcessor.pkl', 'rb') as handle:
            DatasetForPrediction.MultiColumnOneHotEncoder = pickle.load(handle)
        ##
        with open('./dumps/category_mapper.json') as handle:
            DatasetForPrediction.category_mapper = json.load(handle)
        return

    @staticmethod
    def load_dataset(path):
        ##
        data = load_files(path, load_content=False)
        ##
        guage_files = np.array(data['filenames'])
        # gauge_categories = np.array(data['target'])
        gauge_categories_cat = np.array([data['target_names'][i] for i in data['target']])
        gauge_categories = np.array([DatasetForPrediction.category_mapper['cat_to_num']
                                     [gauge_categories_cat[i]] for i in
                                     range(len(gauge_categories_cat))])

        ##
        return guage_files, gauge_categories

    @staticmethod
    def path_to_tensor(img_path, img_size_x, img_size_y, color_mode):
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(img_size_x, img_size_y, color_mode))
        # convert PIL.Image.Image type to 3D tensor with shape (?, ?, ?)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, ?, ?, ?) and return 4D tensor
        return np.expand_dims(x, axis=0)

    @staticmethod
    def paths_to_tensor(imgs_path, img_size_x, img_size_y, color_mode):
        list_of_tensors = [DatasetForPrediction.path_to_tensor(
            img_path, img_size_x, img_size_y, color_mode) for img_path in list(imgs_path)]
        return np.vstack(list_of_tensors)

    @classmethod
    def return_datasets(cls, container_path, final_img_width, final_img_height,
                        color_mode='grayscale'):
        ##
        cls.initialize()
        ##
        guage_files, gauge_categories = cls.load_dataset(path=container_path)
        ##
        X_pred = cls.paths_to_tensor(imgs_path=guage_files, img_size_x=final_img_width,
                                     img_size_y=final_img_height,
                                     color_mode=color_mode)

        ##
        y_pred_cat = pd.DataFrame(data=gauge_categories, columns=['y_true'], dtype='category')
        y_pred = cls.MultiColumnOneHotEncoder.transform(data=y_pred_cat)

        ##
        return X_pred, y_pred.values, guage_files
