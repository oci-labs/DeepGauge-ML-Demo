from modules.prediction_modules import Prediction

Prediction.Predict.predict_online(checkpoint_path='./logs/models/main/',
                                  final_img_width=80,
                                  final_img_height=160,
                                  color_mode="grayscale")
