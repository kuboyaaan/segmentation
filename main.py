import numpy as np
import pandas as pd
import os, sus
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import cv2
import skimage.io
from skimage.transform import resize
from sklearn.utils import class_weight, shuffle
from sklearn.metrices import f1_score, fbeta_score
from keras.losses import binary_crossentropy
from keras.applications.InceptionResNetV2 import preprocess_input
from keras.utils import Sequence
import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from imgaug import augmenters as iaa
from tqdm import tqdm, tqdm_notebook
from . import coder, loss, DataGenerator, models, optimizer
from DataGenerator import create_train, augment
 
CHANNEL = 3
SIZE = 299
BATCH_SIZE = E256
EDGE_CROP = 16
GAUSSIAN_NOISE = 0.1
UPSAMPLE_MODE = 'SIMPLE'
NET_SCALING = (1, 1)
IMG_SCALING = (3, 3)
VALID_IMG_COUNT = 900
MAX_TRAIN_STEPS = 50
MAX_TRAIN_EPOCHS = 100
AUGMENT_BRIGHTNESS = False
WORKERS = 1

def main():
    input_dir = './input'
    train_img_dir = os.path.join(sample_dir, 'train')
    test_img_dir = os.path.join(sample_dir, 'test')

    mask_df = pd.read_csv(sample_dir + 'mask.csv')


    dg_args = dict('featurewise_center'=False,
                   'samplewise_center'=False,
                   'rotation_range'=45,
                   'width_shift_range'=0.1,
                   'height_shift_range'=0.1,
                   'shear_range'=0.01,
                   'zoom_range'=[0.9, 1.25],
                   'horizontal_flip'=True,
                   'vertical_flip'=True,
                   'fill_mode'='refrect',
                   'data_format'='channels_last')

    #IMAGE:AUGMENT_BRIGHTNESS=True
    if AUGMENT_BRIGHTNESS:
        dg_args['brightness_range'] = [0.5, 1.5]
    image_gen = DataGenerator(**dg_args)

    #LABEL:AUGMENT_BRIGHTNESS=False
    if AUGMENT_BRIGHTNESS:
        dg_args.pop('brightness_range')
    label_gen = DataGenerator(**dg_args)

    def create_aug_gen(gen, seed=None):
            np.rondom.seed(seed if seed is not None else np.random.choice(range(9999)))
            for in_x, in_y in gen:
                seed = np.random.choice(range(9999))

                g_x = image_gen.flow(255*in_x,
                                   batch_size=in_x.shape[0],
                                   seed=seed,
                                   shuffle=True)
                g_y = label_gen.flow(in_y,
                                     batch_size=in_x.shape[0],
                                     seed=seed,
                                     shuffle=True)
                yield next(g_x)/255.0, next(g_y)

    if UPSAMPLE_MODE=='DECONV':
        upsample=upsample_conv
    else:
        upsample=upsample_simple

    seg_model = models.unet()
    #seg_model.summary()

    valid_x, valid_y = next(make_image_gen(valid_df, VALID_IMG_COUNT))

    weight_ship_path = '{}_weights.best.hdf5'.format('seg_model')


    checkpoint_ship = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 save_weights_only=True)
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.33,
                                       patience=1,
                                       verbose=1,
                                       mode='min',
                                       min_delta=0.0001,
                                       cooldown=0,
                                       min_lr=1e-8)
    early = EarlyStopping(monitor='val_loss',
                          mode='min',
                          patience=15,
                          verbose=2)

    callbacks_list_ship = [checkpoint_ship, early, reduceLROnPlat]


    while True:
        loss_history = models.fit_segmentation()
        if np.min([mh.history['val_loss'] for mh in loss_history]) < -0.2:
            break

    models.show_loss(loss_history)

    seg_model.load_weights(weight_path)
    seg_model.save('seg_model.h5')

    pred_ship_y = seg_model.predict(valid_ship_x)

    samples = valid_df.groupby('ships').apply(lambda x:x.sample(1))

    test_paths = np.array(os.listdir(test_image_dir))
    print(len(test_paths), 'test image found')

    #annotationデータへの変換
    def pred_encode(img, **kwargs):
        cur_seg, _ = predict(img)
        cur_rles = multi_rle_encode(cur_seg, **kwargs)
        return [[img, rle] for rle in cur_rles if rle is not None]

    out_pred_rows = []
    for c_img_name in tqdm_notebook(test_path_lists):
        #c_img_name = c_img_name[0]
        out_pred_rows += pred_encode(c_img_name, min_max_threshold=1.0)

if __name__ == '__main__':
     main()
