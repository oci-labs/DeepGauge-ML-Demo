import tensorflow as tf

filename = '/home/khodayarim/PycharmProjects/data/ImageEveryUnit/psi_1/gauge_scale_5.jpg'
def parse_image(filename):
    image_string = tf.read_file(filename)
    image = tf.encode_base64(input=image_string)
    return image

with tf.Session() as sess:
    image = parse_image(filename=filename)
    image = sess.run(image)

export_dir='/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/CustomEstimator/modules/ensemble_modules/trainer_from_storage/misc/exported_model/1542403650/'
predict_fn = tf.contrib.predictor.from_saved_model(export_dir)

a = predict_fn({'img_bytes':[image]})