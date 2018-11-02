python customEstimator.py --dev=True --retrain_primary_models=False --hidden_units='[100, 200, 300, 500]' --verbosity=DEBUG


#######
gcloud ml-engine local train \
          --module-name trainer_from_storage.task \
          --package-path /home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/CustomEstimator/modules/ensemble_modules/trainer_from_storage \
          -- \
          --primary_models_directory=/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/CustomEstimator/modules/ensemble_modules/trainer_from_storage/logs/primary_models \
          --ensemble_architecture_path=/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/CustomEstimator/modules/ensemble_modules/trainer_from_storage/logs/temporary_models \
          --path_to_images=/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/CustomEstimator/modules/ensemble_modules/trainer_from_storage/data/ImageEveryUnit \
          --bin_path=/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/CustomEstimator/modules/ensemble_modules/trainer_from_storage/logs/dumps/ \
          --export_dir=/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/CustomEstimator/modules/ensemble_modules/trainer_from_storage/logs/exported_model




####### on GCP
gcloud ml-engine jobs submit training ensemble_training \
          --scale-tier basic \
          --package-path /home/khodayarim/PycharmProjects/ensamble_package.tar.gz  \
          --module-name ensemble_package.task \
          --job-dir gs://deep_gauge \
          --region "us-east1"
          -- \
          --primary_models_directory=gs://deep_gauge/ensemble_package/misc/primary_models \






          --staging-bucket gs://deep_gauge/ensemble_package \

          --packages additional-dep1.tar.gz,dep2.whl
          -- \
          --primary_models_directory=gs://deep_gauge/ensemble_package/misc/primary_models \

          --module-name trainer_from_storage.task \
          --package-path gs://deep_gauge/ensemble_package \
          -- \
          --primary_models_directory=/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/CustomEstimator/modules/ensemble_modules/trainer_from_storage/logs/primary_models \
          --ensemble_architecture_path=/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/CustomEstimator/modules/ensemble_modules/trainer_from_storage/logs/temporary_models \
          --path_to_images=/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/CustomEstimator/modules/ensemble_modules/trainer_from_storage/data/ImageEveryUnit \
          --bin_path=/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/CustomEstimator/modules/ensemble_modules/trainer_from_storage/logs/dumps/ \
          --export_dir=/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/CustomEstimator/modules/ensemble_modules/trainer_from_storage/logs/exported_model





