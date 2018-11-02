import Prediction

a,b,c = Prediction.Predict.predict_online(checkpoint_path='./logs/models/main/',
                                  final_img_width=160,
                                  final_img_height=80,
                                  color_mode="grayscale",filename='./uploads/gauge_scale_11.jpg')
print(a,b,c)
