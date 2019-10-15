
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import h5py
import numpy as np
from matplotlib import pyplot as plt

from keras.models import load_model
from keras_loss_function.keras_ssd_Loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections
from data_generator.object_detection_2d_data_generator_1chan import DataGenerator

img_height = 256
img_width = 256
img_channels = 1

# for SSD_ResNet and SSD_ResFPN
# img_channels = 3

model_path = '/path/model.h5'

# if use SSD loss
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

# if use focal loss
# ssd_loss = SSD_fcLoss(alpha=1.0)

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'DecodeDetections': DecodeDetections,
                                               'compute_loss': ssd_loss.compute_loss})

# load test data
image_dir = '/user/image path and directory/'
testPath = os.path.join(image_dir, 'imgTst.h5')
testFile = h5py.File(testPath, 'r')
testSet = testFile['imgTst']
testSet = np.moveaxis(testSet, [0, 1, 2], [-1, -2, -3])

# Make the generator
test_dataset = DataGenerator(hdf5_dataset_path=None)

test_labels_filename = os.path.join(image_dir, 'labels_tst.csv')

test_dataset.parse_csv(images_dir=image_dir,
                       labels_filename=test_labels_filename,
                       input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                       include_classes= 'all',
                       ret = False)

test_dataset.create_hdf5_datasetZZ(testSet,
                                   file_path='zztemp_tst.h5',
                                   resize=False,
                                   variable_image_size=False,
                                   verbose=True)

test_dataset_size = test_dataset.get_dataset_size()
print("Number of images in the test dataset:\t{:>6}".format(test_dataset_size))

# Make predictions
predict_generator = test_dataset.generate(batch_size=9722,
                                          shuffle=False,
                                          transformations=[],
                                          label_encoder=None,
                                          returns={'processed_images',
                                                   'processed_labels',
                                                   'image_ids'},
                                          keep_images_without_gt=False,
                                          degenerate_box_handling = 'warn_off')

batch_images, batch_labels, slice_ids = next(predict_generator)

y_pred = model.predict(batch_images, verbose = 1)


# decode predictions
y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=0.50, # change confidence level here
                                   iou_threshold=0.1,
                                   top_k='all',
                                   normalize_coords=True,
                                   img_height=img_height,
                                   img_width=img_width)

# Look at samples slices
tstIdx = 77 # Which batch item to look at
i = slice_ids.index(str(tstIdx))
print("Image: brain mets 266 patients", "/slice: ", slice_ids[i])
print()
print("Ground truth boxes:\n")
print(batch_labels[i])

np.set_printoptions(precision=0, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_decoded[i])

# draw the ground truth and predicted box onto the image
plt.figure(figsize=(2,2), dpi = 400)
plt.imshow(batch_images[i,:,:,0], cmap = 'gray')
plt.axis('off')

current_axis = plt.gca()

# Draw the ground truth boxes in yellow (omit the label for more clarity)
for box in batch_labels[i]:
    xmin = box[1]
    ymin = box[2]
    xmax = box[3]
    ymax = box[4]
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin,
                                         ymax-ymin, color='yellow', fill=False, linewidth=0.2))  

# Draw the predicted boxes in red
for box in y_pred_decoded[i]:
    xmin = box[-4]
    ymin = box[-3]
    xmax = box[-2]
    ymax = box[-1]
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin,
                                         ymax-ymin, color='red', fill=False, linewidth=0.2))



# write the decoded results to list of lists
results = []
for i in range(9722):
    
    sliceNum = int(slice_ids[i])
    predLabel = y_pred_decoded[i]
    numObj = predLabel.shape[0]
    if numObj == 0:
        obj = [sliceNum, 0, 0, 0, 0]
        results.append(obj)
    else:
        for j in range(numObj):
            objLabel = predLabel[j]
            objLabelCord = objLabel[2:]
        
            obj = [sliceNum] + np.ndarray.tolist(objLabelCord)
            results.append(obj)
            
results.sort(key = lambda x:x[0])
            
# write the results to csv
import csv

with open('/path/SSD_conf50.csv', 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerows(results)

    
# save images
folderPath = '/save_images/'
for i in range(9722):
    sliceNum = int(slice_ids[i])

    # draw the ground truth and predicted box onto the image
    plt.figure(figsize=(2,2), dpi = 400)
    plt.imshow(batch_images[i,:,:,0], cmap = 'gray')
    plt.axis('off')

    current_axis = plt.gca()

    colors = plt.cm.hsv(np.linspace(0, 1, 2)).tolist() # Set the colors for the bounding boxes
    classes = ['background', 'mets'] # Just so we can print class names onto the image instead of IDs

    # draw the ground truth boxes in yellow (omit the label for more clarity)
    for box in batch_labels[i]:
        xmin = box[1]
        ymin = box[2]
        xmax = box[3]
        ymax = box[4]
        label = '{}'.format(classes[int(box[0])])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin,
                                             ymax-ymin, color='yellow', fill=False, linewidth=0.2))  

    # draw the predicted boxes in red
    for box in y_pred_decoded[i]:
        xmin = box[-4]
        ymin = box[-3]
        xmax = box[-2]
        ymax = box[-1]
        color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin,
                                             ymax-ymin, color=color, fill=False, linewidth=0.2))
        
    plt.savefig(os.path.join(folderPath, 'slice_{0}.tif'.format(sliceNum)))
    plt.close()
    
    
    
    
    

