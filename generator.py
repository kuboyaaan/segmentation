from .coder import mask_as_image, rle_decode, rle_encode, masks_as_image

class DataGenerator(Sequence):
    #mix_up関数
    def mix_up(x, y):
        x = np.array(x, np.float32)
        lam = np.random.beta(1.0, 1.0)
        ori_index = np.arange(int(len(x)))
        index_array = np.arange(int(len(x)))
        np.random.shuffle(index_array)
        #beta_distributionに従い、結合
        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]
        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]
        
        return mixed_x, mixed_y
    
    #バッチ生成関数
    def create_train(dataset_info, batch_size, shape, augument=True, mix=False):
        assert shape[2:] == 3
        while True:
            dataset_info = shuffle(dataset_info)
            #batch_dataの作成
            for start in range(0, len(dataset_info), batch_size):
                end = min(start + batch_size, len(dataset_info))
                batch_images = []
                X_train_batch = dataset_info[start:end]
                batch_labels = np.zeros((len(X_train_batch), NUM_CLASSES))
                
                for i in range(len(X_train_batch)):
                    image = data_generator.load_image(
                        X_train_batch[i]['path'], shape)
                    if augument:
                        image = data_generator.augment(image)#data_generator.augmentの処理内容
                    #作成したdataをresnetの学習に合わせて前処理
                    batch_images.append(preprocess_input(image))
                    #one-hot_encoding
                    batch_labels[i][X_train_batch[i]['labels']] = 1
                    batch_images, batch_labels = data_generator.min_up(batch_images, batch_labels)
                yield np.array(batch_images, np.float32), batch_labels
    
    #画像の読み込み
    def load_image(path, shape):
        image = cv2.imread(path+'.png')
        #画像の縦横比が異なる場合に0paddingして正方形の画像を生成
        image = make_square_image(image)
        image = cv2.resize(image, (SIZE, SIZE))
        return image
    
    #正方形画像の生成関数
    def make_square_image(img, val=0):
        height, width, _ = img.shape
        if height == width:
            return img
        else:
            NO_PADDING = (0, 0)
            if height > width:
                span = (height - width) // 2
                padding = (NO_PADDING, (span, span), NO_PADDING)
            else:
                span = (width - height) // 2
                padding = ((span, span), NO_PADDING, NO_PADDING)
            
            return np.pad(img, padding, mode='constant', constant_values=val) 
    
    #augmentメソッドの作成(前処理)
    def augment(image):
        augment_img = iaa.Sequential([
            iaa.SomeOf((0, 4), [iaa.Crop(percent=(0, 0.1)),
                                iaa.ContrastNormalization((0.8, 1.2)),
                                iaa.Multiply((0.9, 1.1), per_channel=0.2),
                                iaa.Fliplr(0.5),
                                iaa.GaussianBlur(sigma=(0, 0.6)),
                                iaa.Affine(scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)},
                                           translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                                           rotate=(-180, 180),　)])], rondom_order=True)
        
        image_aug = augment_img.augment_image(image)
        return image_aug
    
    def make_image_gen(dataset_info, batch_size=BATCH_SIZE, train_img_dir):
        all_batches = list(dataset_info.groupby('ID'))
        out_rgb = []
        out_mask = []
        while True:
            np.random.shuffle(all_batches)
            for c_img_id, c_masks in all_batches:
                rgb_path = os.path.join(train_img_dir, c_img_id)
                c_img = imread(rgb_path)
                c_mask = np.expand_dims(masks_as_image(c_masks['AnnotationPixels'].values), -1)
            if IMG_SCALING is not None:
                c_img = c_img[::IMG_SCALING[0], ::IMG_SCALING[1]]
                c_mask = c_mask[::IMG_SCALING[0], ::IMG_SCALING[1]]
                
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask=[], []
                
            