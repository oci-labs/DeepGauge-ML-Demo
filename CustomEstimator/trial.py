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