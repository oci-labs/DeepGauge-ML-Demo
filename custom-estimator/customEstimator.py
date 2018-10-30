from modules.ensemble_modules.trainer_from_storage.task import main
import tensorflow as tf

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)