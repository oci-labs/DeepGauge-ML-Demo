import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import multiprocessing


class Dataset():
    @staticmethod
    def load_dataset(path):
        directory = path
        images_folders = tf.train.match_filenames_once(directory + '/*')

        init = (tf.global_variables_initializer(), tf.local_variables_initializer())

        with tf.Session() as sess:
            sess.run(init)
            folders = sess.run(images_folders)

        folders = [str(path) for path in folders]

        imgs_paths_labels = pd.DataFrame(columns=['filenames', 'target'])
        for f in folders:
            with tf.Session() as sess:
                images = tf.train.match_filenames_once(f + '/*.*')
                sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
                image_paths = sess.run(images)
                imgs_paths_labels_current = pd.DataFrame(columns=['filenames', 'target'])
                image_paths = [str(p) for p in image_paths]
                imgs_paths_labels_current.loc[:, 'filenames'] = image_paths
                imgs_paths_labels_current.loc[:, 'target'] = str(f).split('/')[-1]
                imgs_paths_labels = imgs_paths_labels.append(imgs_paths_labels_current)

        imgs_paths_labels.reset_index(drop=True, inplace=True)

        labels_unique = np.unique(imgs_paths_labels.target.values)

        category_mapper = {labels_unique[i]: i for i in range(len(labels_unique))}
        category_mapper_ext = {y: x for x, y in category_mapper.items()}

        category_mapper.update(category_mapper_ext)

        target = imgs_paths_labels['target'].map(category_mapper)

        y_true_zeros = np.zeros((target.shape[0], labels_unique.shape[0]))

        for i in range(len(y_true_zeros)):
            y_true_zeros[i, target[i]] = 1

        y_true = y_true_zeros.copy()

        guage_files = imgs_paths_labels.filenames.values
        gauge_categories = y_true

        map = [category_mapper_ext[i] for i in range(len(category_mapper_ext))]

        return guage_files, gauge_categories, map

    @classmethod
    def split_data_files(cls, ver_ratio, path, random_state, is_trial):
        ##
        guage_files, gauge_categories, map = cls.load_dataset(path=path)

        if is_trial:
            guage_files = guage_files[:20]
            gauge_categories = gauge_categories[:20]
        #
        X_train_path_names, X_test_path_names, y_train, y_test = \
            train_test_split(guage_files,
                             gauge_categories,
                             test_size=ver_ratio,
                             random_state=random_state)

        return X_train_path_names, X_test_path_names, y_train, y_test, map

    @staticmethod
    def parse_function(filename, label):
        filename = filename['file']
        image_string = tf.read_file(filename)
        # image = tf.image.decode_jpeg(image_string, channels=3)
        # image = tf.image.resize_images(image, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # image = tf.image.convert_image_dtype(image, tf.float32)
        # image = tf.encode_base64(input=image_string)
        return {'img_bytes': image_string}, label

    @classmethod
    def prep_input_function(cls,
                            prefetch_buffer_size=None,
                            train_epochs=None,
                            multi_threading=True,
                            train_batch_size=None,
                            mode=None,
                            X_train_path_names=None,
                            X_test_path_names=None,
                            y_train=None,
                            y_test=None):
        ##
        num_threads = multiprocessing.cpu_count() if multi_threading else 1

        if mode == tf.estimator.ModeKeys.TRAIN:

            dataset_train = tf.data.Dataset.from_tensor_slices(({'file': X_train_path_names}, y_train))

            ##
            dataset_train = dataset_train.apply(
                tf.contrib.data.map_and_batch(
                    map_func=cls.parse_function,
                    batch_size=train_batch_size,
                    num_parallel_batches=num_threads))

            dataset_train = dataset_train. \
                cache().shuffle(len(X_train_path_names) + 100).repeat(train_epochs)
            #
            dataset_final = dataset_train.prefetch(buffer_size=prefetch_buffer_size)

        elif mode == tf.estimator.ModeKeys.EVAL:

            dataset_test = tf.data.Dataset.from_tensor_slices(({'file': X_test_path_names}, y_test))

            dataset_test = dataset_test.map(map_func=cls.parse_function,
                                            num_parallel_calls=num_threads)

            dataset_final = dataset_test.batch(y_test.shape[0])

        return dataset_final
