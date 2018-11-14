MODULE_NAME=trainer.task
PACKAGE_PATH=/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/CustomEstimator/modules/ensemble_modules/trainer_from_storage/trainer
PRIMARY_PATH=/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/CustomEstimator/modules/ensemble_modules/trainer_from_storage/misc/primary_models
ENSEMBLE_PATH=/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/CustomEstimator/modules/ensemble_modules/trainer_from_storage/misc/ensemble_graph
IMG_PATH=/home/khodayarim/PycharmProjects/data/ImageEveryUnit
BIN_PATH=/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/CustomEstimator/modules/ensemble_modules/trainer_from_storage/misc/logs/dumps
EXPORT_PATH=/home/khodayarim/PycharmProjects/DeepGauge-ML-Demo/CustomEstimator/modules/ensemble_modules/trainer_from_storage/misc/exported_model

##
gcloud ml-engine local train \
          --module-name "$MODULE_NAME" \
          --package-path "$PACKAGE_PATH" \
          -- \
          --primary_models_directory="$PRIMARY_PATH" \
          --ensemble_architecture_path="$ENSEMBLE_PATH" \
          --path_to_images="$IMG_PATH" \
          --bin_path="$BIN_PATH" \
          --export_dir="$EXPORT_PATH"
