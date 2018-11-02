gsutil -m cp -r ./data/ImageEveryUnit/. gs://deep_gauge/data/
gsutil -m cp -r ./trainer/. gs://deep_gauge

## make a package
tar -czvf ensemble_package.tar.gz /ensemble_package
