
from __future__ import division
import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate
from keras.regularizers import l2
import keras.backend as K

from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_L2Normalization import L2Normalization

def SSD(image_size,
        n_classes,
        l2_regularization=0.0005,
        min_scale=None,
        max_scale=None,
        scales=None,
        aspect_ratios_global=None,
        aspect_ratios_per_layer=[1.0],
        two_boxes_for_ar1=True,
        steps=[8, 16, 32, 64, 100, 300],
        offsets=None,
        clip_boxes=False,
        variances=[0.1, 0.1, 0.2, 0.2],
        coords='centroids',
        normalize_coords=True,
        subtract_mean=[123, 117, 104],
        divide_by_stddev=None,
        swap_channels=[2, 1, 0],
        confidence_thresh=0.75,
        iou_threshold=0.1,
        top_k='all',
        nms_max_output_size=400,
        return_predictor_sizes=False):
    
    # Pre-processes
    n_predictor_layers = 6 
    n_classes += 1 
    l2_reg = l2_regularization
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]


    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError("`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers+1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
    else: # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")



    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1) # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else: # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers


    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]], tensor[...,swap_channels[3]]], axis=-1)

    
    # Build the model

    x = Input(shape=(img_height, img_width, img_channels))

    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
    if not (subtract_mean is None):
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels), name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels), name='input_stddev_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(x1)

    L256 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(x1)
    L256 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(L256)
    pool1 =MaxPooling2D(pool_size=(2,2))(L256)

    L128 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(pool1)
    L128 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(L128)
    L128 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(L128)
    pool2 =MaxPooling2D(pool_size=(2,2))(L128)

    L64 =  Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(pool2)
    L64 =  Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(L64)
    L64 =  Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(L64)
    pool3 =MaxPooling2D(pool_size=(2,2))(L64)

    L32 =  Conv2D(1024, (3,3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(pool3)
    L32 =  Conv2D(1024, (1,1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(L32)
    
    L16 =  Conv2D(256, (1,1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(L32)
    L16 =  ZeroPadding2D(padding=((1,1), (1,1)))(L16)
    L16 =  Conv2D(512, (3,3), strides=(2,2), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(L16)

    L8 =   Conv2D(128, (1,1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(L16)
    L8 =   ZeroPadding2D(padding=((1,1), (1,1)))(L8)
    L8 =   Conv2D(256, (3,3), strides=(2,2), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(L8)

    L4 =   Conv2D(128, (1,1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(L8)
    L4 =   ZeroPadding2D(padding=((1,1), (1,1)))(L4)
    L4 =   Conv2D(256, (3,3), strides=(2,2), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(L4)
    
    # L2 normalization
    L128_norm = L2Normalization(gamma_init=10)(L128)
    L64_norm = L2Normalization(gamma_init=5)(L64)

    # Prediction layers: confidence, location, and anchor boxes
    L128_mbox_conf = Conv2D(n_boxes[0] * n_classes, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='L128_mbox_conf')(L128_norm)
    L64_mbox_conf  = Conv2D(n_boxes[1] * n_classes, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='L64_mbox_conf')(L64_norm)
    L32_mbox_conf  = Conv2D(n_boxes[2] * n_classes, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='L32_mbox_conf')(L32)
    L16_mbox_conf  = Conv2D(n_boxes[3] * n_classes, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='L16_mbox_conf')(L16)
    L8_mbox_conf   = Conv2D(n_boxes[4] * n_classes, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='L8_mbox_conf')(L8)
    L4_mbox_conf   = Conv2D(n_boxes[5] * n_classes, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='L4_mbox_conf')(L4)

    L128_mbox_loc = Conv2D(n_boxes[0] * 4, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='L128_mbox_loc')(L128_norm)
    L64_mbox_loc  = Conv2D(n_boxes[1] * 4, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='L64_mbox_loc')(L64_norm)
    L32_mbox_loc  = Conv2D(n_boxes[2] * 4, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='L32_mbox_loc')(L32)
    L16_mbox_loc  = Conv2D(n_boxes[3] * 4, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='L16_mbox_loc')(L16)
    L8_mbox_loc   = Conv2D(n_boxes[4] * 4, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='L8_mbox_loc')(L8)
    L4_mbox_loc   = Conv2D(n_boxes[5] * 4, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='L4_mbox_loc')(L4)
    
    L128_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                                     two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0], clip_boxes=clip_boxes,
                                     variances=variances, coords=coords, normalize_coords=normalize_coords, name='L128_mbox_priorbox')(L128_mbox_loc)
    
    L64_mbox_priorbox  = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
                                     two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1], clip_boxes=clip_boxes,
                                     variances=variances, coords=coords, normalize_coords=normalize_coords, name='L64_mbox_priorbox')(L64_mbox_loc)
    
    L32_mbox_priorbox  = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
                                     two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2], this_offsets=offsets[2], clip_boxes=clip_boxes,
                                     variances=variances, coords=coords, normalize_coords=normalize_coords, name='L32_mbox_priorbox')(L32_mbox_loc)
    
    L16_mbox_priorbox  = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
                                     two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3], this_offsets=offsets[3], clip_boxes=clip_boxes,
                                     variances=variances, coords=coords, normalize_coords=normalize_coords, name='L16_mbox_priorbox')(L16_mbox_loc)
    
    L8_mbox_priorbox   = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5], aspect_ratios=aspect_ratios[4],
                                     two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4], this_offsets=offsets[4], clip_boxes=clip_boxes,
                                     variances=variances, coords=coords, normalize_coords=normalize_coords, name='L8_mbox_priorbox')(L8_mbox_loc)
    
    L4_mbox_priorbox   = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6], aspect_ratios=aspect_ratios[5],
                                     two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5], this_offsets=offsets[5], clip_boxes=clip_boxes,
                                     variances=variances, coords=coords, normalize_coords=normalize_coords, name='L4_mbox_priorbox')(L4_mbox_loc)
    
    # Reshape and concatenate
    L128_mbox_conf_reshape = Reshape((-1, n_classes), name='L128_mbox_conf_reshape')(L128_mbox_conf)
    L64_mbox_conf_reshape  = Reshape((-1, n_classes), name='L64_mbox_conf_reshape')(L64_mbox_conf)
    L32_mbox_conf_reshape  = Reshape((-1, n_classes), name='L32_mbox_conf_reshape')(L32_mbox_conf)
    L16_mbox_conf_reshape  = Reshape((-1, n_classes), name='L16_mbox_conf_reshape')(L16_mbox_conf)
    L8_mbox_conf_reshape   = Reshape((-1, n_classes), name='L8_mbox_conf_reshape')(L8_mbox_conf)
    L4_mbox_conf_reshape   = Reshape((-1, n_classes), name='L4_mbox_conf_reshape')(L4_mbox_conf)

    L128_mbox_loc_reshape = Reshape((-1, 4), name='L128_mbox_loc_reshape')(L128_mbox_loc)
    L64_mbox_loc_reshape  = Reshape((-1, 4), name='L64_mbox_loc_reshape')(L64_mbox_loc)
    L32_mbox_loc_reshape  = Reshape((-1, 4), name='L32_mbox_loc_reshape')(L32_mbox_loc)
    L16_mbox_loc_reshape  = Reshape((-1, 4), name='L16_mbox_loc_reshape')(L16_mbox_loc)
    L8_mbox_loc_reshape   = Reshape((-1, 4), name='L8_mbox_loc_reshape')(L8_mbox_loc)
    L4_mbox_loc_reshape   = Reshape((-1, 4), name='L4_mbox_loc_reshape')(L4_mbox_loc)
    
    L128_mbox_priorbox_reshape = Reshape((-1, 8), name='L128_mbox_priorbox_reshape')(L128_mbox_priorbox)
    L64_mbox_priorbox_reshape  = Reshape((-1, 8), name='L64_mbox_priorbox_reshape')(L64_mbox_priorbox)
    L32_mbox_priorbox_reshape  = Reshape((-1, 8), name='L32_mbox_priorbox_reshape')(L32_mbox_priorbox)
    L16_mbox_priorbox_reshape  = Reshape((-1, 8), name='L16_mbox_priorbox_reshape')(L16_mbox_priorbox)
    L8_mbox_priorbox_reshape   = Reshape((-1, 8), name='L8_mbox_priorbox_reshape')(L8_mbox_priorbox)
    L4_mbox_priorbox_reshape   = Reshape((-1, 8), name='L4_mbox_priorbox_reshape')(L4_mbox_priorbox)
    
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([L128_mbox_conf_reshape,
                                                       L64_mbox_conf_reshape,
                                                       L32_mbox_conf_reshape,
                                                       L16_mbox_conf_reshape,
                                                       L8_mbox_conf_reshape,
                                                       L4_mbox_conf_reshape])

    mbox_loc = Concatenate(axis=1, name='mbox_loc')([L128_mbox_loc_reshape,
                                                     L64_mbox_loc_reshape,
                                                     L32_mbox_loc_reshape,
                                                     L16_mbox_loc_reshape,
                                                     L8_mbox_loc_reshape,
                                                     L4_mbox_loc_reshape])

    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([L128_mbox_priorbox_reshape,
                                                               L64_mbox_priorbox_reshape,
                                                               L32_mbox_priorbox_reshape,
                                                               L16_mbox_priorbox_reshape,
                                                               L8_mbox_priorbox_reshape,
                                                               L4_mbox_priorbox_reshape])

    # Softmax loss, concatenation and model return
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)
    
    predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])
    
    model = Model(inputs=x, outputs=predictions)


    if return_predictor_sizes:
        predictor_sizes = np.array([L128_mbox_conf._keras_shape[1:3],
                                    L64_mbox_conf._keras_shape[1:3],
                                    L32_mbox_conf._keras_shape[1:3],
                                    L16_mbox_conf._keras_shape[1:3],
                                    L8_mbox_conf._keras_shape[1:3],
                                    L4_mbox_conf._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model