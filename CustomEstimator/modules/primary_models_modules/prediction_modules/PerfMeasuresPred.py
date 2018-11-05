import numpy as np


class Measures(object):
    @staticmethod
    def correct_prediction_pred(logits, y_data):
        ##
        y_pred_cls = np.argmax(logits, axis=1)
        y_true_cls = np.argmax(y_data, axis=1)

        ##
        pred_correct_prediction = y_pred_cls == y_true_cls

        return pred_correct_prediction

    @classmethod
    def compute_measures_pred(cls,
                              logits_pred,
                              y_data_pred):
        # prediction accuracy
        pred_correct_prediction = cls.correct_prediction_pred(logits=logits_pred,
                                                              y_data=y_data_pred)
        pred_acc = np.sum(pred_correct_prediction) / len(pred_correct_prediction)

        return pred_acc

    @classmethod
    def compute_streaming_image_cat(cls, logits_pred):
        y_pred = np.argmax(logits_pred, axis=1)
        pred_logit = logits_pred[0][y_pred]
        return y_pred[0], pred_logit
