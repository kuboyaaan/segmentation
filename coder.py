#画像等前処理に関する関数群
#1.annotationデータからマスク画像の生成
#2.マスク画像からannotationデータの作成(.json)

#decode: list->mask image
#encode: mask image->list

#decoder#############################################

#annotationデータを用いて、全体のマスク画像を生成
def mask_as_image(mask_list):
    all_masks = np.zeros((256, 256), dtype=np.uint8)
    for mask in mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks

#mask部分の個別のリストを受け取って、画像内の個別のmask画像を生成
def rle_decode(mask_rle, shape=(256, 256)):
    s_mask = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


#encoder#############################################
#mask画像からannotationデータの作成
def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    if np.max(img) < min_max_threshold:
        return ''
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return ''
    #平滑化
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    #差分
    runs = np.where(pixels[1:] != pixels[-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ''.join(str(x) for x in runs)

#画像全体のannotationデータの作成
def masks_as_image(mask_list):
    all_masks = np.zeros((256, 256), dtype=np.uint8)
    for mask in mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks

