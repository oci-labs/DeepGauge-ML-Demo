import numpy as np
import json
import os


class Measures(object):

    ##
    @staticmethod
    def correct_prediction(logits, y_data):

        ##
        y_pred_cls = np.argmax(logits, axis=1)
        y_true_cls = np.argmax(y_data, axis=1)

        ##
        correct_prediction = y_pred_cls == y_true_cls

        return correct_prediction

    @staticmethod
    def find_true_cat_logit(logits, y_data):
        ##
        y_true_cls = np.argmax(y_data, axis=1)
        true_cat_predicted_logit = np.array([logits[i][y_true_cls[i]] for i in range(len(y_true_cls))])

        return true_cat_predicted_logit

    @classmethod
    def compute_measures(cls,
                         logits_train,
                         y_train,
                         logits_test,
                         y_test,
                         meta_dict):
        ##
        # train accuracy
        train_correct_prediction = cls.correct_prediction(logits=logits_train, y_data=y_train)
        train_acc = np.sum(train_correct_prediction) / len(train_correct_prediction)

        ##
        # test accuracy
        test_correct_prediction = cls.correct_prediction(logits=logits_test, y_data=y_test)
        test_acc = np.sum(test_correct_prediction) / len(test_correct_prediction)
        test_true_cat_predicted_logit = cls.find_true_cat_logit(logits=logits_test, y_data=y_test)
        test_logits_median = np.median(test_true_cat_predicted_logit)

        ##
        main_acc_and_logits = dict()
        main_acc_and_logits['acc'] = test_acc
        main_acc_and_logits['logits'] = logits_test
        main_acc_and_logits['logits_median'] = test_logits_median

        ##
        classes_info = dict()
        classes_info['main'] = main_acc_and_logits

        ##
        train_classes = dict()
        test_classes = dict()

        for cat, _ in meta_dict['cls_indices']['train_indices'].items():
            ##
            # classes accuracy for train dataset
            train_class_indices = meta_dict['cls_indices']['train_indices'][cat]
            train_class_correct_prediction = train_correct_prediction[train_class_indices]
            train_class_acc = np.sum(train_class_correct_prediction) / len(train_class_correct_prediction)

            # classes logits for train dataset
            train_class_logits = logits_train[train_class_indices]

            y_class_train = y_train[train_class_indices]
            train_cat_logit = cls.find_true_cat_logit(logits=train_class_logits, y_data=y_class_train)
            train_class_logits_median = np.median(train_cat_logit)

            #
            classes_acc_and_logits = dict()
            classes_acc_and_logits['acc'] = train_class_acc
            classes_acc_and_logits['logits'] = train_class_logits
            classes_acc_and_logits['logits_median'] = train_class_logits_median

            #
            train_classes[cat] = classes_acc_and_logits

            ##
            # classes accuracy for test dataset
            test_class_indices = meta_dict['cls_indices']['test_indices'][cat]
            test_class_correct_prediction = test_correct_prediction[test_class_indices]
            test_class_acc = np.sum(test_class_correct_prediction) / len(test_class_correct_prediction)

            # classes logits for test dataset
            test_class_logits = logits_test[test_class_indices]

            ############################
            y_class_test = y_test[test_class_indices]
            test_true_cat_logit = cls.find_true_cat_logit(logits=test_class_logits, y_data=y_class_test)
            test_class_logits_median = np.median(test_true_cat_logit)
            test_class_logits_median = np.median(test_class_logits_median)

            #
            classes_acc_and_logits = dict()
            classes_acc_and_logits['acc'] = test_class_acc
            classes_acc_and_logits['logits'] = test_class_logits
            classes_acc_and_logits['logits_median'] = test_class_logits_median

            #
            test_classes[cat] = classes_acc_and_logits

        ##
        classes_info['train_classes'] = train_classes
        classes_info['test_classes'] = test_classes

        return train_acc, test_acc, classes_info

    @classmethod
    def log_best_models(cls, classes_info, best_models_info, session,
                        saver, meta_dict, models_log_path, keep_best_model):

        # #### trial
        # if (best_models_info['psi_5']['best_acc'] < classes_info['test_classes']['psi_5']['acc']):
        #     saver.save(session, './logs/models/psi_5/best_model_psi_5')
        # ####

        for c, attributes in (classes_info['test_classes']).items():
            ##
            # model_path_cat = best_models_info[c]['folder']
            ##
            if best_models_info[c]['best_acc'] < attributes['acc']:
                #
                best_models_info[c]['best_acc'] = attributes['acc']
                best_models_info[c]['best_median'] = attributes['logits_median']
                best_models_info[c]['logits'] = attributes['logits']
                best_models_info[c]['hyper_params'] = meta_dict['hyper_params']
                #
                log_path = os.path.join(models_log_path, str(c))
                if not os.path.exists(log_path):
                    os.makedirs(log_path)
                saver.save(session, os.path.join(log_path, 'best_model_' + str(c)))
                # #
                with open(os.path.join(log_path, 'hyper_params_' + str(c) + '.json'), 'w') as outfile:
                    json.dump(meta_dict['hyper_params'], outfile)

            elif (best_models_info[c]['best_acc'] == attributes['acc']) and \
                    (best_models_info[c]['best_median'] < classes_info['test_classes'][c]['logits_median']):
                #
                # best_models_info[item]['best_acc'] = value['cls_acc']
                best_models_info[c]['best_median'] = attributes['logits_median']
                best_models_info[c]['logits'] = attributes['logits']
                best_models_info[c]['hyper_params'] = meta_dict['hyper_params']
                #
                log_path = os.path.join(models_log_path, str(c))
                if not os.path.exists(log_path):
                    os.makedirs(log_path)
                saver.save(session, os.path.join(log_path, 'best_model_' + str(c)))
                # #
                with open(os.path.join(log_path, 'hyper_params_' + str(c) + '.json'), 'w') as outfile:
                    json.dump(meta_dict['hyper_params'], outfile)

        save_model = False
        if keep_best_model == True:

            if (best_models_info['main']['best_acc'] < classes_info['main']['acc']) and \
                    (classes_info['main']['acc'] > 0.01):
                save_model = True
        else:
            save_model = True
            #
        if save_model:
            best_models_info['main']['best_acc'] = classes_info['main']['acc']
            best_models_info['main']['best_median'] = classes_info['main']['logits_median']
            best_models_info['main']['logits'] = classes_info['main']['logits']
            best_models_info['main']['hyper_params'] = meta_dict['hyper_params']
            #
            log_path = os.path.join(models_log_path, 'main')
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            saver.save(session, os.path.join(log_path, 'best_model_' + 'main'))
            #
            with open(os.path.join(log_path, 'hyper_params_main.json'), 'w') as outfile:
                json.dump(meta_dict['hyper_params'], outfile)

        return best_models_info
