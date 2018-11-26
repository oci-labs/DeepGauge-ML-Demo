import os
import tensorflow as tf
from tensorflow.python.framework import meta_graph

from CustomEstimator.modules.primary_models_modules import LoadImg

graph_model = tf.Graph()
with graph_model.as_default():
    ##################################
    params = {"checkpoint_path": "./logs/primary_models/1st_good"}
    with tf.Session(graph=graph_model) as sess:
        params = {"checkpoint_path": "./logs/primary_models/1st_good"}

        saver = tf.train.import_meta_graph(os.path.join(params["checkpoint_path"], 'best_model_main.meta'),
                                           clear_devices=True,
                                           import_scope='')
        saver.restore(tf.get_default_session(),
                      tf.train.latest_checkpoint(params["checkpoint_path"]))
    # meta_graph.import_scoped_meta_graph(os.path.join(params["checkpoint_path"],
    #                                                  'best_model_main.meta'),
    #                                     clear_devices=True,
    #                                     import_scope='graph_model')
    # tf.train.list_variables(params["checkpoint_path"])
    # tf.train.load_variable(params["checkpoint_path"], 'Variable_9/Adam_1')
    # tf.train.init_from_checkpoint(ckpt_dir_or_file=params["checkpoint_path"],
    #                               assignment_map={'/': '/'})
    # meta_graph.read_meta_graph_file(filename="./logs/primary_models/1st_good/best_model_main.meta")
    # graph_model.clear_collection('trainable_variables')
    # graph_model.clear_collection('train_op')
    # graph_model.clear_collection('variables')

    X_image_tf = graph_model.get_tensor_by_name("X_image_tf:0")
    logits_tf = graph_model.get_tensor_by_name("logits_tf:0")
    logits_tf_sg = tf.stop_gradient(logits_tf)
    ####################################

graph_pipeline = tf.Graph()
with graph_pipeline.as_default():
    ##################################
    X_raw = tf.placeholder(tf.float32, shape=[None, None, None, None], name="X_raw")
    params = {"checkpoint_path": "./logs/primary_models/1st_good"}
    meta_graph.import_scoped_meta_graph(os.path.join(params["checkpoint_path"],
                                                     'best_model_main.meta'),
                                        clear_devices=True,
                                        import_scope='')

    X_image_tf = graph_pipeline.get_tensor_by_name("X_image_tf:0")
    # X_image_tf = tf.placeholder(tf.float32, shape=[None, None, None, None], name="X_image_tf")

    resized_imgs = tf.identity(tf.image.resize_images(X_raw, (X_image_tf.get_shape().as_list()[1],
                                                              X_image_tf.get_shape().as_list()[2])),
                               name='resized_imgs')
    ####################################

graph = tf.get_default_graph()

raw_imgs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3],
                          name='raw_imgs')

meta_graph_1 = tf.train.export_meta_graph(graph=graph_pipeline)
meta_graph.import_scoped_meta_graph(meta_graph_1,
                                    input_map={"X_raw": raw_imgs},
                                    import_scope='graph_pipeline')

out_1 = graph.get_tensor_by_name('graph_pipeline/resized_imgs:0')

meta_graph_2 = tf.train.export_meta_graph(graph=graph_model)
meta_graph.import_scoped_meta_graph(meta_graph_2,
                                    input_map={"X_image_tf": out_1},
                                    import_scope='graph_model')

out_2 = graph.get_tensor_by_name('graph_model/logits_tf:0')

print(tf.global_variables())

X_train, X_test, y_train, y_test, cls_indices = LoadImg.Dataset.prep_datasets(
    ver_ratio=0.2, container_path='data/ImageEveryUnit',
    final_img_width=224, final_img_height=224,
    color_mode="grayscale", random_state=1911, is_trial=True)

# with tf.device('CPU:0'):
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./trial/writer', sess.graph)
    res = sess.run(out_2, feed_dict={raw_imgs: X_train})
    ##
    saver = tf.train.Saver(max_to_keep=1)
    saver.save(sess, './trial/best_model_main')
    writer.close()

##### prediction
import tensorflow as tf
from modules import LoadImg

X_train, X_test, y_train, y_test, cls_indices = LoadImg.Dataset.prep_datasets(
    ver_ratio=0.2, container_path='data/ImageEveryUnit',
    final_img_width=224, final_img_height=224,
    color_mode="grayscale", random_state=1911, is_trial=True)

with tf.Session() as session:
    writer = tf.summary.FileWriter('./trial/writer_prediction', session.graph)
    saver_3 = tf.train.import_meta_graph('./trial/best_model_main.meta',
                                         clear_devices=True)
    saver_3.restore(tf.get_default_session(),
                    tf.train.latest_checkpoint('./trial'))
    graph_pred = tf.get_default_graph()
    raw_imgs = graph_pred.get_tensor_by_name("raw_imgs:0")
    logits_tf = graph_pred.get_tensor_by_name("graph_model/logits_tf:0")

    ##
    feed_dict_pred = {raw_imgs: X_train}

    ##
    logits = session.run(logits_tf, feed_dict=feed_dict_pred)
    writer.close()

# graph_img_pipe = tf.Graph()
# with graph_img_pipe.as_default():
#     ########################################
#     X_raw = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="X_raw")
#
#     meta_graph_img_raw = tf.train.export_meta_graph(graph=graph)
#     meta_graph.import_scoped_meta_graph(meta_graph_img_raw,
#                                         input_map={"raw_imgs": X_raw},
#                                         import_scope='graph_img_pipe')
#
#     meta_graph_model = tf.train.export_meta_graph(graph=graph_model)
#     meta_graph.import_scoped_meta_graph(meta_graph_model,
#                                         import_scope='graph_img_pipe')
#
#     img_tf = graph_model.get_tensor_by_name('graph_model/X_image_tf:0')
#     ##################
#     resized_imgs = tf.identity(tf.image.resize_images(X_raw, (img_tf.get_shape().as_list()[1],
#                                                               img_tf.get_shape().as_list()[2])),
#                                name='resized_imgs')
#     ################################################

# graph = tf.get_default_graph()
# x = tf.placeholder(tf.float32, shape=[1], name="xx")
# raw_imgs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3],
#                                name='raw_imgs')
# meta_graph_1 = tf.train.export_meta_graph(graph=graph_img_raw)
# meta_graph.import_scoped_meta_graph(meta_graph_1,
#                                     input_map={"X_img_raw": raw_imgs},
#                                     import_scope='')


# a = tf.image.resize_images(X_train, (80,80))
#
# with tf.Session(graph=graph) as sess:
#     res = sess.run(a, feed_dict={raw_imgs: X_train})

# X_train, X_test, y_train, y_test, cls_indices = LoadImg.Dataset.prep_datasets(
#     ver_ratio=0.2, container_path='data/ImageEveryUnit',
#     final_img_width=244, final_img_height=244,
#     color_mode="grayscale", random_state=1911)
#
# imgs = X_train[:10, :, :, :]
#
# with tf.Session() as sess:
#     writer = tf.summary.FileWriter('trial', sess.graph)
#
#     img_size_x = imgs.shape[1]
#     img_size_y = imgs.shape[2]
#     num_channels = imgs.shape[3]
#     X_image_tf_ensemble = tf.placeholder(tf.float32,
#                                          shape=[None, img_size_x, img_size_y, num_channels],
#                                          name='X_image_tf_ensemble')  #############################
#
#     params = {"checkpoint_path": "./logs/primary_models/1st_good"}
#
#     saver = tf.train.import_meta_graph(os.path.join(params["checkpoint_path"], 'best_model_main.meta'),
#                                        clear_devices=True)
#     saver.restore(tf.get_default_session(),
#                   tf.train.latest_checkpoint(params["checkpoint_path"]))
#     graph = tf.get_default_graph()
#
#     ##
#     X_image_tf = graph.get_tensor_by_name("X_image_tf:0")
#
#     input_shape = X_image_tf.get_shape().as_list()
#
#     ##
#     resize_op = tf.image.resize_images(X_image_tf_ensemble,
#                                        (input_shape[1], input_shape[2]))
#
#     model_input = tf.get_default_session().run(resize_op,
#                                                feed_dict={X_image_tf_ensemble: imgs})
#
#     ##
#     logits_tf = graph.get_tensor_by_name("logits_tf:0")
#     # #
#     feed_dict_pred = {X_image_tf: model_input}
#     # #
#     logits_pred = tf.get_default_session().run(logits_tf, feed_dict=feed_dict_pred)
#     saver_2 = tf.train.Saver(max_to_keep=1000000)
#     saver_2.save(tf.get_default_session(), os.path.join('./trial/best_model_main'))
#     writer.close()
#
# ##### prediction
# with tf.Session() as session:
#     saver_3 = tf.train.import_meta_graph('./trial/best_model_main.meta',
#                                          clear_devices=True)
#     saver_3.restore(tf.get_default_session(),
#                     tf.train.latest_checkpoint(params["checkpoint_path"]))
#     graph_2 = tf.get_default_graph()
#     X_image_tf_ensemble_2 = graph_2.get_tensor_by_name("X_image_tf_ensemble:0")
#     logits_tf_2 = graph_2.get_tensor_by_name("logits_tf:0")
#
# ##
#     feed_dict_pred = {X_image_tf_ensemble_2: imgs}
#
# ##
#     logits_pred_2 = session.run(logits_tf_2, feed_dict=feed_dict_pred)


# graph_1 = tf.Graph()
# with graph_1.as_default():
#     a = tf.placeholder(tf.float32, shape=[1], name="aa")
#     c = tf.sqrt(a, name='out_1')
#
# graph_2 = tf.Graph()
# with graph_2.as_default():
#     u = tf.placeholder(tf.float32, shape=[1], name="uu")
#     y = tf.sqrt(u, name='out_2')
#
# graph = tf.get_default_graph()
# x = tf.placeholder(tf.float32, shape=[1], name="xx")
#
# meta_graph_1 = tf.train.export_meta_graph(graph=graph_1)
# meta_graph.import_scoped_meta_graph(meta_graph_1,
#                                     input_map={"aa": x},
#                                     import_scope='graph_1')
# out_1=graph.get_tensor_by_name('graph_1/out_1:0')
#
# meta_graph_2 = tf.train.export_meta_graph(graph=graph_2)
# meta_graph.import_scoped_meta_graph(meta_graph_2,
#                                     input_map={"uu": out_1},
#                                     import_scope='graph_2')
# out_2=graph.get_tensor_by_name('graph_2/out_2:0')
#
# print(tf.global_variables())
#
# with tf.Session(graph=graph) as sess:
#     res = sess.run(out_2, feed_dict={x:[16.0]})
path_to_images='data/ImageEveryUnit'
bin_path='dumps/'
prefetch_buffer_size=1700000

from modules.ensemble_modules.trainer_from_storage.input import Dataset

input_fn_train, input_fn_eval, n_output = Dataset.prep_dataflow_functions(
    ver_ratio=0.2,
    container_path=path_to_images,
    prefetch_buffer_size=prefetch_buffer_size,
    epochs_between_evals=1,
    random_state=19,
    is_trial='True',
    bin_path=bin_path,
    multi_threading='False',
    train_batch_size=4)



#######################
import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image_string = tf.read_file('./modules/ensemble_modules/trainer_from_storage/data/ImageEveryUnit/psi_0/gauge_scale_0.jpg')
image = tf.image.decode_jpeg(image_string, channels=3)
# image = tf.expand_dims(image, 0)
image = tf.image.resize_images(image, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# image = tf.image.resize_images(image, [224, 224])
# image = tf.squeeze(image, 0)
image = tf.image.convert_image_dtype(image, tf.float32)

with tf.Session() as sess:
    im = sess.run(image)

im.shape

# cv2.imshow("output", im)
plt.imshow(im)

img = Image.fromarray(im, 'RGB')
img.show()


###################
import tensorflow as tf

with tf.Session() as sess:
    a = tf.gfile.ListDirectory("CustomEstimator/modules/ensemble_modules/trainer_from_storage/misc/primary_models")


##########################

import tensorflow as tf

directory = "CustomEstimator"
file_names = tf.train.match_filenames_once(directory+'/*')

# file_names = tf.gfile.ListDirectory("CustomEstimator")


init = (tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
    f = sess.run(file_names)

image_paths_b=[str(path) for path in f]
image_paths=[path.split("'")[1] for path in image_paths_b]

targets_b = [str(path).split('/')[0] for path in f]
targets = [path.split("'")[2] for path in targets_b]

print(f)


###############################################
import tensorflow as tf

# predicted_classes = tf.constant([1])

l = ['cat_1', 'cat_2', 'cat_3']

ll = str(l)

category_map = tf.convert_to_tensor(l)
# category_map = tf.convert_to_tensor(params["category_map"])

# class_label = tf.gather_nd(category_map, predicted_classes)
# class_label = tf.convert_to_tensor([class_label]) #[:, tf.newaxis]

with tf.Session() as sess:
    cat = sess.run(category_map[tf.newaxis])











###########################################
import tensorflow as tf
import numpy as np


filename = '/home/khodayarim/PycharmProjects/data/ImageEveryUnit/psi_1/gauge_scale_5.jpg'
def parse_image(filename):
    image_string = tf.read_file(filename)
    # image = tf.image.decode_jpeg(image_string, channels=3)
    # image = tf.image.resize_images(image, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.encode_base64(input=image_string)
    return image_string, image

with tf.Session() as sess:
    image_string, image = parse_image(filename=filename)
    image = sess.run(image)
    image_string = sess.run(image_string)



###########################
from tensorflow.python.lib.io import file_io

def _open_file_read_binary(uri):
  try:
    return file_io.FileIO(uri, mode='rb')
  except errors.InvalidArgumentError:
    return file_io.FileIO(uri, mode='r')


with _open_file_read_binary(filename) as f:
    image_bytes = f.read()
###########################


img_func = b'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



export_dir='/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/CustomEstimator/modules/ensemble_modules/trainer_from_storage/misc/exported_model/1542378905'
predict_fn = tf.contrib.predictor.from_saved_model(export_dir)

a = predict_fn({'img_bytes':[image]})




###########################
from google.cloud import storage

client = storage.Client()
bucket_id = 'ocideepgauge-images'
bucket = client.get_bucket(bucket_id)
blob = bucket.blob(data['name'])
img = base64.b64encode(blob.download_as_string())