# chest_xray_xai_pipeline
Training, evaluation and explicability pipeline with Grad-CAM for CNN models that classify radiological images, with a focus on the clinical validation of the models' activations.

All notebooks are compatible with both CPU and GPU. GPU is strongly recommended for training (notebooks were developed on NVIDIA T4 and L4 via Google Colab). CPU execution is supported but training times will be significantly longer.

# General pre requisites
--Complete config.py: add the path where the project will generate

# Pre requisites for 01_import_dataset
--Complete verification course from https://physionet.org/content/vindr-cxr/1.0.0/
--Have a kaggle account
---Apply for this contest https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection

# Pre requisites for 02_preprocessing_images_and_training
--Have a dataset inside PROJECT_PATH/dataset with two resolutions -> PROJECT_PATH/dataset/images_256 and PROJECT_PATH/dataset/images_1024
