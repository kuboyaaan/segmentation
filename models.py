from keras_applications import imagenet_utils as utils
from keras import models, layers

NET_SCALING=(1, 1)
GAUSSIAN_NOISE=0.1
EDGE_CROP=16
#物体検知(segmentation)用
def upsample_conv(filters, kernel_size, strides, padding):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)
def upsample_simple(filters, kernel_size, strides, padding):
    return layers.UpSampling2D(strides)

if UPSAMPLE_MODE=='DECONV':
    upsample=upsample_conv
else:
    upsample=upsample_simple

def unet():
    input_img = layers.Input((256, 256, 3), name = 'RGB_Input')
    pp_in_layer = input_img
    if NET_SCALING is not None:
        pp_in_layer = layers.AvgPool2D(NET_SCALING)(pp_in_layer)

    pp_in_layer = layers.GaussianNoise(GAUSSIAN_NOISE)(pp_in_layer)
    pp_in_layer = layers.BatchNormalization()(pp_in_layer)

    c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (pp_in_layer)
    c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    p1 = layers.MaxPooling2D((2, 2)) (c1)

    c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    p2 = layers.MaxPooling2D((2, 2)) (c2)

    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    p3 = layers.MaxPooling2D((2, 2)) (c3)

    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2)) (c4)


    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

    u6 = upsample(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

    u7 = upsample(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

    u8 = upsample(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
    c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

    u9 = upsample(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
    c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

    d = layers.Conv2D(1, (1, 1), activation='sigmoid') (c9)
    d = layers.Cropping2D((EDGE_CROP, EDGE_CROP))(d)
    d = layers.ZeroPadding2D((EDGE_CROP, EDGE_CROP))(d)
    if NET_SCALING is not None:
        d = layers.UpSampling2D(NET_SCALING)(d)
        
    seg_model = models.Model(input[input_img], outputs=[d])
    return seg_model


def fit_segmentation():
    seg_model.compile(optimizer=Adam(1e-3, decay=1e-6), loss=IoU, metrics=['binary_accuracy'])
    
    step_count = min(MAX_TRAIN_STEPS, train_df.shape[0]//BATCH_SIZE)
    aug_gen = create_aug_gen(make_image_gen(train_df))
    loss_history = [seg_model.fit_generator(aug_gen,
                                 steps_per_epoch=step_count,
                                 epochs=MAX_TRAIN_EPOCHS,
                                 validation_data=(valid_x, valid_y),
                                 callbacks=callbacks_list_ship,
                                workers=1 # the generator is not very thread safe
                                           )]
    return loss_history


def show_loss(loss_history):
    epoch = np.cumsum(np.concatenate(
        [np.linspace(0.5, 1, len(mh.epoch)) for mh in loss_history]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    _ = ax1.plot(epoch, np.concatenate([mh.history['loss'] for mh in loss_history]),
                 'b-', epoch, np.concatenate(
                     [mh.history['val_loss'] for mh in loss_history]),'r-')
    ax1.legend(['Training', 'Validation'])
    ax1.set_title('Loss')
    
    _ = ax2.plot(epoch, np.concatenate(
        [mh.history['binary_accuracy'] for mh in loss_history]), 'b-',
                     epoch, np.concatenate(
            [mh.history['val_binary_accuracy'] for mh in loss_history]),
                     'r-')
    ax2.legend(['Training', 'Validation'])
    ax2.set_title('Binary Accuracy (%)')

def raw_prediction(img, path=test_image_dir):
    c_img = imread(os.path.join(path, c_img_name))
    c_img = np.expand_dims(c_img, 0)/255.0
    cur_seg = fullres_model.predict(c_img)[0]
    return cur_seg, c_img[0]

def smooth(cur_seg):
    return binary_opening(cur_seg>0.99, np.expand_dims(disk(2), -1))

def predict(img, path=test_image_dir):
    cur_seg, c_img = raw_prediction(img, path=path)
    return smooth(cur_seg), c_img



#画像分類用
def ResNet(stack_fn, preact, use_bias, model_name='resnet', include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000, **kwargs):
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = keras.backend, keras.layers, keras.models, keras.utils
    
    if not (weights in {'imagenet': None} or os.path.exists(weights)):
        raise ValueError('The weights argument should be either None (ranodm initialization), imagenet (pre-training on ImageNet), or the path to the weights file to be loaded')
        
    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError("If using 'weights' as 'imagenet' with include_top' as true, classes should be 1000")
    
    input_shape = utils._obtain_input_shape(input_shape,
                                            default_size=224,
                                            min_size=32,
                                            data_format=backend.image_data_format(),
                                            require_flatten=include_top,
                                            weights=weights)
    
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
            
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)
    if preact is False:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
        x = layers.Activation('relu', name='conv1_relu')(x)
    
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)
    
    x = stack_fn(x)
    
    if preact is True:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='post_bn')(x)
        x = layers.Activation('relu', name='post_relu')(x)
    
    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='probs')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling(name='max_pool')(x)
            
    if input_tensor is not None:
        inputs = keras.utils.get_score_inputs(input_tensor)
    else:
        inputs = img_input
        
    model = models.Model(inputs, x, name=model_name)
    
    if (weights == 'imagenet') and (model_name in WEIGHT_HASHES):
        if include_top:
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels.h5'
            file_hash = WEIGHTS_HASHES[model_name][0] 
        else:
            fine_name = model_name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
            file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = keras_utils.get_file(file_name, BASE_WEIGHTS_PATH+file_name, cache_subdir='models', file_hash=file_hash)
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
        
    return model


