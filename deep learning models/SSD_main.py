
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from math import ceil
import numpy as np

import h5py

# if use baseline SSD or SSD with focal loss
from SSD_model import SSD

# if use ResNet backbone
# from SSD_ResNet_model import SSD_ResNet

# if use RetinaNet
# from SSD_ResFPN_model import SSD_ResFPN

# original SSD loss
from keras_loss_function.keras_ssd_Loss import SSDLoss

# if use focal loss
# from keras_loss_function.keras_ssd_focalLoss import SSD_fcLoss

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from data_generator.data_generator import DataGenerator
from data_generator.data_augmentation_chain import DataAugmentationConstantInputSize

# if use ResNet backbone and RetinaNet
# from data_generator_3chan.data_generator import DataGenerator
# from data_generator_3chan.data_augmentation_chain import DataAugmentationConstantInputSize

img_height = 256 
img_width = 256 
img_channels = 1

# for SSD_ResNet and SSD_ResFPN
# img_channels = 3

intensity_mean = None 
intensity_range = 1 
swap_channels = False
n_classes = 1 
min_scale = None
max_scale = None
scales = [0.008, 0.016, 0.032, 0.064, 0.115, 0.2, 0.3] 
ar_global = [1.0]
ar_per_layer = None 

two_boxes_for_ar1 = True 
steps = None 
offsets = None 
clip_boxes = False 
variances = [0.5, 0.5, 0.5, 0.5] 
normalize_coords = True

# 1: build the model
model = SSD(image_size=(img_height, img_width, img_channels),
            n_classes=n_classes,
            l2_regularization=0,
            min_scale = min_scale,
            max_scale = max_scale,
            scales = scales,
            aspect_ratios_global = ar_global,
            aspect_ratios_per_layer = ar_per_layer,
            two_boxes_for_ar1 = two_boxes_for_ar1,
            steps=steps,
            offsets=offsets,
            clip_boxes=clip_boxes,
            variances=variances,
            normalize_coords=normalize_coords,
            subtract_mean=intensity_mean,
            divide_by_stddev=intensity_range,
            swap_channels = swap_channels,
            confidence_thresh=0.5,
            iou_threshold=0.1,
            top_k='all',
            nms_max_output_size=400)

# 2: Instantiate an Adam optimizer and the SSD loss function and compile the model
adam = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# 3: Initiate data generators
trn_dataset = DataGenerator(hdf5_dataset_path = None)
val_dataset = DataGenerator(hdf5_dataset_path = None)

# not-useful image dir
image_dir = '/user/path to image folder/'

trnPath = os.path.join(image_dir, 'imgTrn.h5')
valPath = os.path.join(image_dir, 'imgVal.h5')

trn_file = h5py.File(trnPath, 'r')
val_file = h5py.File(valPath, 'r')

trn_set = trn_file['imgTrn']
val_set = val_file['imgVal']

trnMax = np.max(trn_set)
valMax = np.max(val_set)

val_set = val_set/valMax * trnMax

trn_set = np.moveaxis(trn_set, [0, 1, 2], [-1, -2, -3])
val_set = np.moveaxis(val_set, [0, 1, 2], [-1, -2, -3])

# Ground truth
trn_labels_filename = os.path.join(image_dir, 'labels_trn.csv')
val_labels_filename = os.path.join(image_dir, 'labels_val.csv')

trn_dataset.parse_csv(images_dir=image_dir,
                      labels_filename=trn_labels_filename,
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')
                        
val_dataset.parse_csv(images_dir=image_dir,
                      labels_filename=val_labels_filename,
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                      include_classes='all')                       

trn_dataset.create_hdf5_datasetZZ(trn_set,
                                  file_path='zztemp_trn.h5',
                                  resize=False,
                                  variable_image_size=False,
                                  verbose=True)

val_dataset.create_hdf5_datasetZZ(val_set,
                                  file_path='zztemp_val.h5',
                                  resize=False,
                                  variable_image_size=False,
                                  verbose=True)

# Get the number of samples in the training and validations datasets.
trn_dataset_size = trn_dataset.get_dataset_size()
val_dataset_size = val_dataset.get_dataset_size()

print("Number of images in the trn dataset:\t{:>6}".format(trn_dataset_size))
print("Number of images in the val dataset:\t{:>6}".format(val_dataset_size))

# set the batchsize
batch_size = 16

# 4: Define the image processing chain.
data_augmentation_chain = DataAugmentationConstantInputSize(random_flip = 0.5,
                                                            random_translate = ((0.03,0.3), (0.03,0.3), 0.5),
                                                            random_scale = (0.9, 3.0, 0.5),
                                                            random_rotate = ((90, 180, 270), 0.5),
                                                            n_trials_max = 4,
                                                            clip_boxes=True,
                                                            overlap_criterion='area',
                                                            bounds_box_filter=(0.5, 1.0),
                                                            bounds_validator=(0.5, 1.0),
                                                            n_boxes_min=1,
                                                            background=0)

# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.
predictor_sizes = [model.get_layer('L128_mbox_conf').output_shape[1:3],
                   model.get_layer('L64_mbox_conf').output_shape[1:3],
                   model.get_layer('L32_mbox_conf').output_shape[1:3],
                   model.get_layer('L16_mbox_conf').output_shape[1:3],
                   model.get_layer('L8_mbox_conf').output_shape[1:3],
                   model.get_layer('L4_mbox_conf').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    min_scale = min_scale,
                                    max_scale = max_scale,
                                    scales=scales,
                                    aspect_ratios_global=ar_global,
                                    aspect_ratios_per_layer = ar_per_layer,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.2,
                                    neg_iou_limit=0.1,
                                    normalize_coords=normalize_coords)

# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.
trn_generator = trn_dataset.generate(batch_size=batch_size,
                                     shuffle=True,
                                     transformations=[data_augmentation_chain],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=True,
                                     transformations=[],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.2,
                                         patience=3,
                                         verbose=1,
                                         min_delta=0.001,
                                         cooldown=0,
                                         min_lr=1e-7)

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0.0,
                               patience=7,
                               verbose=1)

csv_logger = CSVLogger(filename='/path/SSD.csv',
                       separator=',',
                       append=False)

callbacks = [reduce_learning_rate,
             early_stopping,
             csv_logger]


initial_epoch   = 0
final_epoch     = 100
steps_per_epoch = 5*ceil(trn_dataset_size/batch_size)

history = model.fit_generator(generator=trn_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size/batch_size),
                              initial_epoch=initial_epoch)
# save the model
model.save('/path/SSD.h5')

