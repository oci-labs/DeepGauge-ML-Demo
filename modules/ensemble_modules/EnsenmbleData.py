from modules.prediction_modules import Prediction
import os
import tensorflow as tf
import glob
import numpy as np
import pickle
from modules import LoadImg
import cv2
from collections import OrderedDict


def save_dictionary(data, path, file_name):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, file_name), 'wb') as outfile:
        pickle.dump(data, outfile)
    return


def find_required_datasets_dim_and_save(models_directory='./logs/historical_models/'):
    dims = []
    dims_dict = OrderedDict()
    models_info = OrderedDict()
    for i, m in enumerate(os.listdir(models_directory)):
        checkpoint_path = glob.glob(os.path.join(models_directory, m))[0]

        ##
        with tf.Session() as session:
            Prediction.Predict.restore_model(checkpoint_path=checkpoint_path, sess=session)
            graph = tf.get_default_graph()
            X_image_tf = graph.get_tensor_by_name("X_image_tf:0")
            image_shape = X_image_tf.get_shape().as_list()
            if image_shape[1:4] not in dims:
                dims.append(image_shape[1:4])
            models_info[m] = image_shape[1:4]
        tf.reset_default_graph()
        ##
    dims_dict['models_info'] = models_info
    dims_dict["ensemble_models_dims"] = dims
    save_dictionary(data=dims_dict, path='./dumps', file_name='ensemble_models_dims.pkl')
    # with open('dumps/ensemble_models_dims.pkl', 'wb') as outfile:
    #     pickle.dump(dims_dict, outfile)
    return


def prep_datasets_and_save(data_path='data/ImageEveryUnit', ver_ratio=0.2):
    with open('dumps/ensemble_models_dims.pkl', 'rb') as pklFile:
        dims_dict = pickle.load(pklFile)

    unique_dims = dims_dict['ensemble_models_dims']

    X_train, X_test, y_train, y_test, cls_indices = LoadImg.Dataset.prep_datasets(
        ver_ratio=ver_ratio, container_path=data_path,
        final_img_width=None, final_img_height=None,
        color_mode="grayscale", random_state=1911)

    data_dict = {}
    for dim in unique_dims:
        temp_dict = {}

        current_X_train = []
        for img_num in range(X_train.shape[0]):
            current_X_train.append(cv2.resize(X_train[img_num], dsize=(dim[1], dim[0])))

        current_X_test = []
        for img_num in range(X_test.shape[0]):
            current_X_test.append(cv2.resize(X_test[img_num], dsize=(dim[1], dim[0])))

        temp_dict['X_train'] = np.array(current_X_train)
        temp_dict['X_test'] = np.array(current_X_test)
        temp_dict['y_train'] = y_train
        temp_dict['y_test'] = y_test
        temp_dict['cls_indices'] = cls_indices
        data_dict[str(dim)] = temp_dict
    save_dictionary(data=data_dict, path='./data/ensembleDatasets',
                    file_name='pre_ensembleData.pkl')
    return


def prep_data_for_ensemble(data_path='./data/ensembleDatasets/pre_ensembleData.pkl',
                           save_folder='./data/ensembleDatasets',
                           file_name='pre_ensembleData_final.pkl'):
    ##
    with open('dumps/ensemble_models_dims.pkl', 'rb') as pklFile:
        dims_dict = pickle.load(pklFile)

    with open(data_path, 'rb') as pklFile:
        data_dict = pickle.load(pklFile)

    models_info = dims_dict['models_info']

    # logits_appended_train = np.array([])
    # logits_appended_test = np.array([])

    for model_folder, dims in models_info.items():
        tf.reset_default_graph()
        data = data_dict[str(dims)]
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        cls_indices = data['cls_indices']

        checkpoint_path = os.path.join('./logs/historical_models/', model_folder)
        logits_pred_train, _, _ = Prediction. \
            Predict.predict_batch(checkpoint_path=checkpoint_path,
                                  X_pred=X_train,
                                  y_true=y_train,
                                  get_results_pandas=False)
        try:
            logits_appended_train = np.append(logits_appended_train, logits_pred_train, axis=1)
        except:
            logits_appended_train = logits_pred_train.copy()

        logits_pred_test, _, _ = Prediction. \
            Predict.predict_batch(checkpoint_path=checkpoint_path,
                                  X_pred=X_test,
                                  y_true=y_test,
                                  get_results_pandas=False)
        try:
            logits_appended_test = np.append(logits_appended_test, logits_pred_test, axis=1)
        except:
            logits_appended_test = logits_pred_test.copy()

        ##
        ensemble_data_dict = {}
        ensemble_data_dict['X_train'] = logits_appended_train
        ensemble_data_dict['y_train'] = y_train
        ensemble_data_dict['X_test'] = logits_appended_test
        ensemble_data_dict['y_test'] = y_test
        ensemble_data_dict['cls_indices'] = cls_indices

        save_dictionary(data=ensemble_data_dict, path=save_folder,
                        file_name=file_name)

    return

def do_all_data_prep_for_training():
    find_required_datasets_dim_and_save()
    print('The dims saved.')
    prep_datasets_and_save()
    print('The first data prep done.')
    prep_data_for_ensemble()
    print('The second data prep done.')
    return


def load_ensemble_data(data_path='./data/ensembleDatasets/pre_ensembleData_final.pkl'):
    ##
    with open(data_path, 'rb') as pklFile:
        data_dict = pickle.load(pklFile)
    ##
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    cls_indices = data_dict['cls_indices']
    return X_train, X_test, y_train, y_test, cls_indices
