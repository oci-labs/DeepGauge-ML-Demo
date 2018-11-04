import importlib

from CustomEstimator.modules.primary_models_modules import LoadImg
from modules import OptimizeAndLog

## load data
# The color_mode is either 'rgb' or 'grayscale' (default).
X_train, X_test, y_train, y_test, cls_indices = LoadImg.Dataset.prep_datasets(
    ver_ratio=0.2, container_path='data/ImageEveryUnit',
    final_img_width=79, final_img_height=79 * 2,
    color_mode="grayscale", random_state=1911)

##
importlib.reload(OptimizeAndLog)
OptimizeAndLog.OptimizerLogger. \
    train_and_save_logs_three_CNN(filter_size1=17, num_filters1=45, strides_1=[1, 7, 7, 1],
                                  use_pooling_1=True, pooling_ksize_1=[1, 4, 4, 1], pooling_strides_1=[1, 4, 4, 1],
                                  ##
                                  filter_size2=7, num_filters2=17, strides_2=[1, 5, 5, 1],
                                  use_pooling_2=True, pooling_ksize_2=[1, 3, 3, 1], pooling_strides_2=[1, 3, 3, 1],
                                  ##
                                  filter_size3=1, num_filters3=7, strides_3=[1, 1, 1, 1],
                                  use_pooling_3=False, pooling_ksize_3=None, pooling_strides_3=None,
                                  ##
                                  fc_size=86,
                                  num_iterations=2,
                                  learning_rate=2e-4, momentum=None,
                                  X_train=X_train, y_train=y_train,
                                  X_test=X_test, y_test=y_test,
                                  models_log_path='./logs/models/',
                                  cls_indices=cls_indices,
                                  padding='SAME',
                                  keep_best_model=False,
                                  device_name="/cpu:0")
