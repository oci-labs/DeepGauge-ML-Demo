import tensorflow as tf
from modules.prediction_modules import PerfMeasuresPred
import importlib
import numpy as np
import json
import pandas as pd
import cv2
import os
import modules.prediction_modules.LoadImgPred as imgLoader
from tkinter import *
from PIL import Image
import glob


class Predict(object):

    def __init__(self):
        self.top_cats = np.array(['', '', ''])
        self.top_logits = np.array([0, 0, 0])

    @staticmethod
    def restore_model(checkpoint_path, sess):
        # checkpoint_path = './logs/models/' + checkpoint_path + '/'
        meta_graph_path = glob.glob(os.path.join(checkpoint_path, '*.meta'))[0]
        saver = tf.train.import_meta_graph(meta_graph_path, clear_devices=True)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
        return

    @staticmethod
    def get_pandas_results(guage_files, logits_pred, y_true, save):
        ##
        y_pred = np.argmax(logits_pred, axis=1)
        y_hat = np.argmax(y_true, axis=1)

        with open('./dumps/category_mapper.json') as handle:
            category_mapper = json.load(handle)

        pred_cats_mapped = [category_mapper['num_to_cat'][str(y_pred[i])]
                            for i in range(len(y_pred))]

        y_hat_cats_mapped = [category_mapper['num_to_cat'][str(y_hat[i])]
                             for i in range(len(y_hat))]

        good_pred_indices = y_hat == y_pred

        pandas_final_result = pd.DataFrame()
        pandas_final_result['y_hat_cats'] = y_hat_cats_mapped
        pandas_final_result['pred_cats'] = pred_cats_mapped
        pandas_final_result['files'] = guage_files
        pandas_final_result['is_good_pred'] = good_pred_indices

        ##
        # pandas_final_result['logits'] = pandas_final_result['logits'].astype(object)
        pandas_final_result['logits'] = 0
        pandas_final_result['logits'] = pandas_final_result['logits'].astype(object)
        [pandas_final_result.set_value(i, 'logits', logits_pred[i])
         for i in range(len(logits_pred))]
        pandas_final_result.sort_values(by=['is_good_pred', 'y_hat_cats'], axis=0, inplace=True)
        if save:
            pandas_final_result.to_csv('./performance_logs/pandas_final_result.csv')

        return pandas_final_result

    @classmethod
    def predict_batch(cls, checkpoint_path, X_pred, y_true,
                      get_results_pandas=True, guage_files=[],
                      save_results_pandas=True):
        ##
        with tf.Session() as session:
            cls.restore_model(checkpoint_path=checkpoint_path, sess=session)
            graph = tf.get_default_graph()

            ##
            X_image_tf = graph.get_tensor_by_name("X_image_tf:0")
            logits_tf = graph.get_tensor_by_name("logits_tf:0")

            ##
            feed_dict_pred = {X_image_tf: X_pred}

            ##
            logits_pred = session.run(logits_tf, feed_dict=feed_dict_pred)

            ##
            pred_acc = PerfMeasuresPred.Measures.compute_measures_pred(
                logits_pred=logits_pred,
                y_data_pred=y_true)
            ##
            pandas_final_result = pd.DataFrame()
            if get_results_pandas:
                pandas_final_result = Predict.get_pandas_results(guage_files,
                                                                 logits_pred=logits_pred,
                                                                 y_true=y_true,
                                                                 save=save_results_pandas)

            ##
        return logits_pred, pred_acc, pandas_final_result

    @staticmethod
    def normalize_image(img):
        num_of_channels = img.shape[-1]
        for ch in range(num_of_channels):
            img[0][:, :, ch] = np.subtract(img[0][:, :, ch], np.min(img[0][:, :, ch]))
            img[0][:, :, ch] = img[0][:, :, ch] / np.max(img[0][:, :, ch]) * 255
        return img

    # @staticmethod
    # def update_cat_ranks(cat, logit):
    #     top_logits = np.append(Predict.top_logits, [logit])
    #     top_cats = np.append(Predict.top_cats, [cat])
    #     sort_indices = np.argsort(top_logits)
    #     sort_indices = np.flip(sort_indices)
    #     top_logits_sorted = top_logits[sort_indices]
    #     top_cats_sorted = top_cats[sort_indices]
    #     duplicate_cats_indices = np.array([])
    #     for i in range(1, len(top_cats_sorted)):
    #         if top_cats_sorted[i - 1] == top_cats_sorted[i]:
    #             np.append(duplicate_cats_indices, [i])
    #     top_cats_sorted = np.delete(top_cats_sorted, duplicate_cats_indices)
    #     top_logits_sorted = np.delete(top_logits_sorted, duplicate_cats_indices)
    #     Predict.top_logits = top_logits_sorted
    #     Predict.top_cats = top_cats_sorted
    #     return top_cats_sorted[0:3], top_logits_sorted[0:3]
    @staticmethod
    def update_cat_ranks(cat, logit, top_cats_sorted, top_logits_sorted):
        top_logits = np.append(top_logits_sorted, [logit])
        top_cats = np.append(top_cats_sorted, [cat])
        sort_indices = np.argsort(top_logits)
        sort_indices = np.flip(sort_indices, axis=0)
        top_logits_sorted = top_logits[sort_indices]
        top_cats_sorted = top_cats[sort_indices]

        top_cats_unique = np.unique(top_cats_sorted)
        top_logits_unique = np.array([top_logits_sorted[top_cats_sorted == cat][0] for cat in top_cats_unique])
        return top_cats_unique, top_logits_unique

    @classmethod
    def predict_online(cls, checkpoint_path,
                       final_img_width,
                       final_img_height,
                       color_mode,
                       frame_ratio=1):

        ## initialize
        with open('./dumps/category_mapper.json') as handle:
            category_mapper = json.load(handle)

        if not os.path.exists('./logs/temporary_latest_streaming_image'):
            os.makedirs('./logs/temporary_latest_streaming_image')

        path = './logs/temporary_latest_streaming_image/'
        ##
        importlib.reload(PerfMeasuresPred)
        with tf.Session() as session:
            cls.restore_model(checkpoint_path=checkpoint_path, sess=session)
            graph = tf.get_default_graph()

            ##
            X_image_tf = graph.get_tensor_by_name("X_image_tf:0")
            logits_tf = graph.get_tensor_by_name("logits_tf:0")

            #######################################
            cap = cv2.VideoCapture(0)

            currentFrame = 0
            top_cats_sorted = np.array(['', '', ''])
            top_logits_sorted = np.array([0, 0, 0])
            while (True):
                # Capture frame-by-frame
                frame = cv2.medianBlur(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY), 5)
                # ret, frame = cap.read()
                # small_frame = frame
                small_frame = frame[0:480, 0:480]

                ###############################################

                output = small_frame.copy()
                circles = cv2.HoughCircles(small_frame, cv2.HOUGH_GRADIENT, minDist=200, dp=1.2,
                                           param1=120, param2=50, minRadius=50, maxRadius=450)
                # ensure at least some circles were found
                if circles is not None:
                    # convert the (x, y) coordinates and radius of the circles to integers
                    circles = np.round(circles[0, :]).astype("int")

                    x = circles[0, 0]
                    y = circles[0, 1]
                    r = circles[0, 2]

                    ## to crop the original frame
                    square_half_side = int(frame_ratio * r)
                    cropped_frame = output[x - square_half_side:x + square_half_side,
                                    y - square_half_side:y + square_half_side]

                    # Saves image of the current frame in jpg file
                    img_path = path + 'latest_streaming_frame' + '.jpg'
                    cv2.imwrite(img_path, cropped_frame)

                    # draw the circle in the output image, then draw a rectangle
                    # corresponding to the center of the circle
                    cv2.circle(cropped_frame, (x, y), r, (0, 255, 0), 4)
                    cv2.rectangle(cropped_frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                    # cv2.rectangle(output, (x - r, y - r), (x + r, y + r), (0, 128, 255), -1)

                    try:

                        processed_img = imgLoader.DatasetForPrediction. \
                            path_to_tensor(img_path=img_path,
                                           img_size_x=final_img_width,
                                           img_size_y=final_img_height,
                                           color_mode=color_mode)
                        processed_img = cls.normalize_image(processed_img)

                        feed_dict_pred = {X_image_tf: processed_img}

                        ##
                        logits_pred = session.run(logits_tf, feed_dict=feed_dict_pred)

                        pred_cls_num, pred_logit = PerfMeasuresPred.Measures.compute_streaming_image_cat(
                            logits_pred=logits_pred)

                        pred_cat = category_mapper['num_to_cat'][str(pred_cls_num)]

                        top_cats_sorted, top_logits_sorted = \
                            cls.update_cat_ranks(cat=pred_cat, logit=pred_logit[0],
                                                 top_cats_sorted=top_cats_sorted,
                                                 top_logits_sorted=top_logits_sorted)

                        print(top_cats_sorted[1:5])
                        print(top_logits_sorted[1:5])

                        # print(pred_cat)

                        # show the output image
                        cv2.imshow("output", cropped_frame)
                        # Display the resulting frame
                        cv2.putText(small_frame, str(pred_cat),
                                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                                    2, cv2.LINE_AA)

                        cv2.putText(small_frame, 'logit={:.3f}'.format(pred_logit[0]),
                                    (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                                    2, cv2.LINE_AA)

                    except:
                        pass

                    cv2.imshow('camera frame', small_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                ##########################################################

            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()

        return
