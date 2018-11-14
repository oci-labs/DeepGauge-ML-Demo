from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

##
from trainer.input import Dataset
import trainer.model as model


##
def initialise_hyper_params(parser):
    parser.add_argument('--path_to_images',
                        default='data/ImageEveryUnit',
                        type=str,
                        help='path to images (e.g. gs://...)')
    parser.add_argument('--primary_models_directory',
                        default='./logs/primary_models/',
                        type=str)
    parser.add_argument('--job_dir',
                        default='./logs/job_dir/',
                        type=str)
    parser.add_argument('--images_shape',
                        default='[None, 224, 224, 3]',
                        type=str)
    parser.add_argument('--hidden_units',
                        default='[500, 100]',
                        type=str)
    parser.add_argument('--learning_rate',
                        default=3e-4,
                        type=float)
    parser.add_argument('--retrain_primary_models',
                        choices=['True', 'False'],
                        default='False',
                        type=str)
    parser.add_argument('--batch_size',
                        default=500,
                        type=int)
    parser.add_argument('--train_epochs',
                        default=250,
                        type=int)
    parser.add_argument('--export_dir',
                        default='./logs/exported_model',
                        type=str)
    parser.add_argument('--ensemble_architecture_path',
                        default='./logs/ensemble_graph/',
                        type=str)
    parser.add_argument('--dev',
                        choices=['True', 'False'],
                        default='False',
                        type=str)
    parser.add_argument('--color_mode',
                        default='grayscale',
                        type=str)
    parser.add_argument('--random_state',
                        default=1911,
                        type=int)
    parser.add_argument('--prefetch_buffer_size',
                        default=1700000,
                        type=int)
    parser.add_argument('--verbosity',
                        choices=[
                            'DEBUG',
                            'ERROR',
                            'FATAL',
                            'INFO',
                            'WARN'
                        ],
                        default='INFO')
    parser.add_argument('--image_processing_multi_threading',
                        choices=['True', 'False'],
                        default='True',
                        type=str
                        )
    return parser


##
def main(argv):
    args = HYPER_PARAMS.parse_args(argv[1:])

    ##
    job_dir = args.job_dir
    images_shape = eval(args.images_shape)
    path_to_images = args.path_to_images
    is_trial = args.dev == 'True'
    primary_models_directory = args.primary_models_directory
    hidden_units = eval(args.hidden_units)
    ensemble_architecture_path = args.ensemble_architecture_path
    learning_rate = args.learning_rate
    retrain_primary_models = args.retrain_primary_models == 'True'
    train_epochs = args.train_epochs
    batch_size = int(args.batch_size)
    export_dir = args.export_dir
    prefetch_buffer_size = args.prefetch_buffer_size
    image_processing_multi_threading = args.image_processing_multi_threading == 'True'

    ##
    tf.logging.set_verbosity(args.verbosity)

    ##
    X_train_path_names, X_test_path_names, y_train, y_test, map = \
        Dataset.split_data_files(ver_ratio=0.2,
                                 path=path_to_images,
                                 random_state=19,
                                 is_trial=is_trial)

    ##
    model.create_ensemble_architecture(hidden_units=hidden_units,
                                       n_output=y_train.shape[1],
                                       primary_models_directory=primary_models_directory,
                                       images_shape=images_shape,
                                       save_path=ensemble_architecture_path)

    print('The ensemble architecture was made and is ready to be used.')

    classifier = tf.estimator.Estimator(
        model_fn=model.model_fn,
        params={
            'primary_models_directory': primary_models_directory,
            'images_shape': images_shape,
            'hidden_units': hidden_units,
            'learning_rate': learning_rate,
            'ensemble_architecture_path': ensemble_architecture_path,
            'retrain_primary_models': retrain_primary_models,
            'category_map': map
        },
        model_dir=job_dir)

    # Train and evaluate model.

    ##
    classifier.train(input_fn=lambda: Dataset.prep_input_function(
        prefetch_buffer_size=prefetch_buffer_size,
        train_epochs=train_epochs,
        multi_threading=image_processing_multi_threading,
        train_batch_size=batch_size,
        mode=tf.estimator.ModeKeys.TRAIN,
        X_train_path_names=X_train_path_names,
        X_test_path_names=X_test_path_names,
        y_train=y_train,
        y_test=y_test
    ))

    eval_result = classifier.evaluate(input_fn=lambda: Dataset.prep_input_function(
        multi_threading=image_processing_multi_threading,
        mode=tf.estimator.ModeKeys.EVAL,
        X_train_path_names=X_train_path_names,
        X_test_path_names=X_test_path_names,
        y_train=y_train,
        y_test=y_test))

    image = tf.placeholder(tf.string, shape=(None, None), name='export_input_image')
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({'X': image})
    classifier.export_savedmodel(export_dir_base=export_dir,
                                 serving_input_receiver_fn=input_fn)
    print('')
    print('')
    print('validation accuracy after {} epochs is: {}'.format(train_epochs, eval_result['accuracy']))
    print('')
    print('')


##
args_parser = argparse.ArgumentParser()
HYPER_PARAMS = initialise_hyper_params(args_parser)

##
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
