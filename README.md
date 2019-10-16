# Brain metatases detection using single shot detectors

All codes developed in our research of brain metastases detection using single shot detectors, including the deep learning models and result analysis scripts, can be found in this repository.

The deep learning codes used in our research are based on the repository https://github.com/pierluigiferrari/ssd_keras, which is under the Apache 2.0 License. It has full implementation of single-shot multibox detector using the Keras deep learning library with detailed explainations. We modified the models and part of the auxiliary codes to fit for our research. We truly appreciate the repository authors sharing their codes, and recommend researchers to refer to the repository for further exploit.

The deep learning models we used are baseline SSD, SSD with ResNet50 backbone, SSD with focal loss, and RetinaNet. Auxiliary codes for the four models are same, except the data generator, where one-channel images are generated for the baseline SSD and SSD with focal loss, and three-channel (channel duplicated 3 times) images are generated for the SSD with ResNet50 backbone and RetinaNet.
 
