python customEstimator.py --dev=True --retrain_primary_models=False --hidden_units='[100, 200, 300, 500]' --verbosity=DEBUG


#######
gcloud ml-engine local train --module-name trainer_from_storage.task \
          --package-path /home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/modules/ensemble_modules/trainer_from_storage \
          -- --primary_models_directory=/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/modules/ensemble_modules/trainer_from_storage/logs/primary_models \
          --ensemble_architecture_path=/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/modules/ensemble_modules/trainer_from_storage/logs/temporary_models \
          --path_to_images=/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/modules/ensemble_modules/trainer_from_storage/data/ImageEveryUnit \
          --bin_path=/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/modules/ensemble_modules/trainer_from_storage/logs/dumps






