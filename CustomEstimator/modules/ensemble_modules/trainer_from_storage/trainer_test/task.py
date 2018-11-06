import argparse
import tensorflow as tf

## on
# from trainer_test.input import Dataset
import trainer.model as model

## dev
# from modules.ensemble_modules.trainer_from_storage.input import Dataset
# from modules.ensemble_modules.trainer_from_storage import model

##
def initialise_hyper_params(parser):
    parser.add_argument('--primary_models_directory',
                        default='./logs/primary_models/',
                        type=str)
    parser.add_argument('--images_shape',
                        default='[None, 224, 224, 3]',
                        type=str)
    parser.add_argument('--ensemble_architecture_path',
                        default='./logs/ensemble_graph/',
                        type=str)
    parser.add_argument('--verbosity',
                        choices=[
                            'DEBUG',
                            'ERROR',
                            'FATAL',
                            'INFO',
                            'WARN'
                        ],
                        default='INFO')
    return parser

##
def main(argv):
    args = HYPER_PARAMS.parse_args(argv[1:])

    ##     color_mode = args.color_mode
    images_shape = eval(args.images_shape)
    primary_models_directory = args.primary_models_directory
    ensemble_architecture_path = args.ensemble_architecture_path


    ##
    tf.logging.set_verbosity(args.verbosity)

    model.create_ensemble_architecture(hidden_units=[100, 200],
                                       n_output=31,
                                       primary_models_directory=primary_models_directory,
                                       images_shape=images_shape,
                                       save_path=ensemble_architecture_path)

    print('The ensemble architecture was made and is ready to be used.')

    # classifier = tf.estimator.Estimator(
    #     model_fn=model.model_fn,
    #     params={
    #         'primary_models_directory': primary_models_directory,
    #         'images_shape': images_shape,
    #         'hidden_units': hidden_units,
    #         'learning_rate': learning_rate,
    #         'ensemble_architecture_path': ensemble_architecture_path,
    #         'retrain_primary_models': retrain_primary_models
    #     })
    #
    # # Train and evaluate model.
    # model_criteria = 0.0
    # image = tf.placeholder(tf.float32, shape=images_shape, name='export_input_image')
    # input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({'X': image})
    # ##
    # for i in range(train_epochs // epochs_between_evals):
    #     print('epoch {} .........'.format(i * epochs_between_evals + 1))
    #     classifier.train(input_fn=lambda: Dataset.prep_input_function(
    #         ver_ratio=0.2,
    #         container_path=path_to_images,
    #         prefetch_buffer_size=prefetch_buffer_size,
    #         epochs_between_evals=epochs_between_evals,
    #         random_state=19,
    #         is_trial=is_trial,
    #         bin_path=bin_path,
    #         multi_threading=image_processing_multi_threading,
    #         train_batch_size=batch_size,
    #         mode=tf.estimator.ModeKeys.TRAIN))
    #
    #     eval_result = classifier.evaluate(input_fn=lambda: Dataset.prep_input_function(
    #         ver_ratio=0.2,
    #         container_path=path_to_images,
    #         random_state=19,
    #         is_trial=is_trial,
    #         bin_path=bin_path,
    #         multi_threading=image_processing_multi_threading,
    #         mode=tf.estimator.ModeKeys.EVAL))
    #     print('')
    #     print('')
    #     print('current validation dataset accuracy is: {}'.format(eval_result['accuracy']))
    #     print('current highest validation dataset accuracy is: {}'.format(model_criteria))
    #     if eval_result[metric] >= model_criteria:
    #         model_criteria = eval_result[metric].copy()
    #         print('current highest validation dataset accuracy updated to: {}'.format(model_criteria))
    #         print('')
    #         print('')
    #         classifier.export_savedmodel(export_dir, input_fn, strip_default_attrs=True)
    #         print('model updated')
    #         continue
    #     print('')
    #     print('')

##
args_parser = argparse.ArgumentParser()
HYPER_PARAMS = initialise_hyper_params(args_parser)

##
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
