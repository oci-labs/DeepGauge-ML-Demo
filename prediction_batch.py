from modules.prediction_modules import LoadImgPred
from modules.prediction_modules import Prediction
import numpy as np

# The color_mode is either 'rgb' or 'grayscale' (default).
X_pred, y_true, guage_files = LoadImgPred.DatasetForPrediction.return_datasets(
    container_path='data/testData_PILed',
    final_img_width=80*2,
    final_img_height=80, color_mode="grayscale")

logits_pred, pred_acc, pandas_final_result = Prediction. \
    Predict.predict_batch(checkpoint_path='./logs/historical_models/4th',
                          X_pred=X_pred,
                          y_true=y_true,
                          guage_files=guage_files)
