from modules.ensemble_modules import obsolete_EnsenmbleData
from modules import OptimizeAndLog
import importlib

## load data
X_train, X_test, y_train, y_test, cls_indices = \
    obsolete_EnsenmbleData.load_ensemble_data(data_path='./data/ensembleDatasets/pre_ensembleData_final.pkl')

##
importlib.reload(OptimizeAndLog)
OptimizeAndLog.OptimizerLogger. \
    train_and_save_two_fc_ensemble(fc_size_1=500, fc_size_2=100,
                                   use_drop_out_1=True, use_drop_out_2=True,
                                   num_iterations=250,
                                   learning_rate=3e-4, momentum=None,
                                   X_train=X_train, y_train=y_train,
                                   X_test=X_test, y_test=y_test,
                                   cls_indices=cls_indices,
                                   models_log_path='./logs/models_ensemble/',
                                   device_name="/cpu:0")
