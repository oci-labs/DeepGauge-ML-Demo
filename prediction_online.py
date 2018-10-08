from modules.prediction_modules import Prediction

Prediction.Predict.predict_online(checkpoint_path='./logs/models/main/',
                                  final_img_width=79,
                                  final_img_height=237,
                                  color_mode="grayscale")
