# coding=UTF-8
# This Python file uses the following encoding: utf-8

import cv2
import numpy as np
import xml.etree.cElementTree as ET
from random import sample

#default args:
default_args = {'noise_prob': 0.1,
                'gasuss_mean': 0,
                'gasuss_var': 0.001,
                'rand_hug': 30,
                'rand_saturation':30,
                'rand_light': 30,
                'rot_angle': 15,
                'bordervalue': (127, 127, 127),
                'zoom_out_value': 0.7,
                'output_shape': (416, 416),
                'take_value' : 5
               }


#添加黑色noise
def sp_noise(image, box_loc=None, **kwargs):
    h, w = image.shape[0:2]
    noise = np.random.rand(h,w)
    out_img = image.copy()
    out_img[noise < kwargs['noise_prob']] = 0
    if box_loc is None:
        return out_img
    else:
        return out_img, box_loc

#高斯noise
def gasuss_noise(image, box_loc=None, **kwargs):
    out_img = (image / 255.) - 0.5
    noise = np.random.normal(kwargs['gasuss_mean'], kwargs['gasuss_var']** 0.5, image.shape)
    out_img = out_img + noise + 0.5
    out_img[out_img < 0] = 0
    out_img[out_img > 1] = 1
    out_img = (out_img * 255).astype(np.uint8)
    if box_loc is None:
        return out_img
    else:
        return out_img, box_loc

#調整彩度(彩度通道加上隨機-N~N之值)
def mod_hue(image, box_loc=None, **kwargs):
    out_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    out_img[:,:,0] += np.random.randint(-kwargs['rand_hug'], kwargs['rand_hug'])
    out_img = cv2.cvtColor(np.clip(out_img, 0, 180).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if box_loc is None:
        return out_img
    else:
        return out_img, box_loc

#調整飽和度(飽和度通道加上隨機-N~N之值)
def mod_saturation(image, box_loc=None, **kwargs):
    out_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    out_img[:,:,1] += np.random.randint(-kwargs['rand_saturation'], kwargs['rand_saturation'])
    out_img = cv2.cvtColor(np.clip(out_img, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if box_loc is None:
        return out_img
    else:
        return out_img, box_loc

#調整亮度(亮度通道加上隨機-N~N之值)
def mod_light(image, box_loc=None, **kwargs):
    out_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    out_img[:,:,2] += np.random.randint(-kwargs['rand_light'], kwargs['rand_light'])
    out_img = cv2.cvtColor(np.clip(out_img, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if box_loc is None:
        return out_img
    else:
        return out_img, box_loc

#水平翻轉
def horizontal_flip(image, box_loc=None, **kwargs):
    '''
    Args:
        box_loc: bounding box location(x_min, y_min, x_max, y_max)
    '''
    if box_loc is None:
        return cv2.flip(image, 1)
    else:
        w = image.shape[1]
        for i in box_loc:
            if i[2] == 0:
                break
            else:
                x_min, x_max = i[0], i[2]
                i[0] = w - x_max
                i[2] = w - x_min
        return cv2.flip(image, 1), box_loc

#垂直翻轉
def vertical_flip(image, box_loc=None, **kwargs):
    '''
    Args:
        box_loc: bounding box location(num box,(x_min, y_min, x_max, y_max, label))
    '''
    if box_loc is None:
        return cv2.flip(image, 0)
    else:
        h = image.shape[0]
        for i in box_loc:
            if i[3] == 0:
                break
            else:
                y_min, y_max = i[1], i[3]
                i[1] = h - y_max
                i[3] = h - y_min
        return cv2.flip(image, 0), box_loc

#旋轉-n~n度
def rot_image(image, box_loc=None, **kwargs):
    '''
    Args:
        box_loc: bounding box location(num box,(x_min, y_min, x_max, y_max, label))
        rot: 要選轉的範圍
        bordervalue: 空白處補的值
    '''
    h, w, _ = image.shape
    center = ( w // 2, h // 2)
    angle = np.random.randint(-kwargs['rot_angle'], kwargs['rot_angle'])
    M = cv2.getRotationMatrix2D(center, angle, 1)
    out_img = cv2.warpAffine(image, M, (w, h), borderValue = kwargs['bordervalue'])
    if box_loc is None:
        return out_img
    else:
        loc = box_loc[:,0:4].copy()
        loc = np.append(loc, loc[:, 0:1], axis=-1)
        loc = np.append(loc, loc[:, 3:4], axis=-1)
        loc = np.append(loc, loc[:, 2:3], axis=-1)
        loc = np.append(loc, loc[:, 1:2], axis=-1)
        loc = loc.reshape(-1, 4, 2)
        loc = loc - np.array(center)
        rot_loc = loc.dot(np.transpose(M[:,0:2]))
        rot_loc = rot_loc + np.array(center)
        rot_box = np.hstack([np.min(rot_loc, axis=-2), np.max(rot_loc, axis=-2), box_loc[:, 4:5]])
        rot_box = np.floor(rot_box)
        rot_box[...,0:4] = np.clip(rot_box[...,0:4], [0,0,0,0], [w-1, h-1, w-1, h-1])

        return out_img, rot_box

#等比例縮放影像
def resize_img(image, box_loc=None, **kwargs):
    h, w, _ = image.shape
    max_edge = max(kwargs['output_shape'][0], kwargs['output_shape'][1])
    scale = min( max_edge / h, max_edge / w)
    h = int(h * scale)
    w = int(w * scale)

    if box_loc is None:
        return cv2.resize(image, (w, h))
    else:
        box_loc[:,0] = box_loc[:,0] * scale
        box_loc[:,1] = box_loc[:,1] * scale
        box_loc[:,2] = box_loc[:,2] * scale
        box_loc[:,3] = box_loc[:,3] * scale
        return cv2.resize(image, (w, h)), box_loc.astype(np.int32)

#將樸片補至指定大小
def padding_img(image, box_loc=None, **kwargs):
    h, w, _ = image.shape

    dx = int((kwargs['output_shape'][1] - w) / 2)
    dy = int((kwargs['output_shape'][0] - h) / 2)

    out_img = np.ones((kwargs['output_shape'][0], kwargs['output_shape'][1], 3), np.uint8) * kwargs['bordervalue'][0]
    out_img[dy: dy + h, dx: dx + w] = cv2.resize(image, (w, h))

    if box_loc is None:
        return out_img
    else:
        box_loc[:,0] = box_loc[:,0] + dx
        box_loc[:,1] = box_loc[:,1] + dy
        box_loc[:,2] = box_loc[:,2] + dx
        box_loc[:,3] = box_loc[:,3] + dy
        return out_img, box_loc.astype(np.int32)

#隨機縮小 value~1倍
def random_zoom_out(image, box_loc=None, **kwargs):

    h, w, _ = image.shape
    scale = np.random.uniform(kwargs['zoom_out_value'], 1)
    h = int(h * scale)
    w = int(w * scale)
    dx = int((image.shape[1] - w) / 2)
    dy = int((image.shape[0] - h) / 2)
    out_img = np.ones(image.shape, np.uint8) * kwargs['bordervalue'][0]
    out_img[dy: dy + h, dx: dx + w] = cv2.resize(image, (w, h))

    if box_loc is None:
        return out_img
    else:
        box_loc[:,0] = box_loc[:,0] * scale + dx
        box_loc[:,1] = box_loc[:,1] * scale + dy
        box_loc[:,2] = box_loc[:,2] * scale + dx
        box_loc[:,3] = box_loc[:,3] * scale + dy
        return out_img, box_loc.astype(np.int32)


#load csv data
def load_csv(xml_path, max_boxes=4):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    #location list
    loc_list = np.zeros((0, 5))
    box_count = 0

    for obj in root.iter('object'):
        if box_count >= max_boxes:
            break

        '''
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue

        cls_id = classes.index(cls)
        '''
        loc = obj.find('bndbox')
        x_min = int(loc.find('xmin').text)
        y_min = int(loc.find('ymin').text)
        x_max = int(loc.find('xmax').text)
        y_max = int(loc.find('ymax').text)

        loc_list = np.vstack([loc_list, np.array([x_min, y_min, x_max, y_max, 0])])

        box_count += 1
    
    return loc_list.astype(np.float32)

#draw rectangle
def draw_rect(image, box_loc):
    for i in box_loc:
        cv2.rectangle(image, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0, 255, 0), 4)

def print_args(**kwargs):
    for key, value in kwargs.items():
        print('key name: {}\nvalue:{}\n'.format(key, value))

#隨機選擇0~N個 image augmentation方法
def rand_aug_image(image, box_loc=None, **kwargs):

    if box_loc is None:
        out_img = resize_img(image, **kwargs)
    else:
        out_img, box_loc = resize_img(image, box_loc, **kwargs)

    #total augmentation function
    func_list = [sp_noise, gasuss_noise, mod_hue, mod_saturation, mod_light,
                 horizontal_flip, vertical_flip, rot_image, random_zoom_out]
    
    #rand take function
    take_func = sample(func_list, np.random.randint(kwargs['take_value']))

    for func in take_func:
        if box_loc is None:
            out_img = func(out_img, **kwargs)
        else:
            out_img, box_loc = func(out_img, box_loc, **kwargs)
    if box_loc is None:
        out_img = padding_img(out_img, **kwargs)
        return out_img
    else:
        out_img, box_loc = padding_img(out_img, box_loc, **kwargs)
        return out_img, box_loc


if __name__ == "__main__":
    img = cv2.imread('./00002.jpg')
    bbox = load_csv('./00002.xml')

    #黑點noise
    #aug_img = sp_noise(img, **default_args)
    #aug_img, bbox = sp_noise(img, bbox, **default_args)

    #gasuss_noise
    #aug_img = gasuss_noise(img, **default_args)
    #aug_img, bbox = gasuss_noise(img, bbox, **default_args)

    #調整Hue
    #aug_img = mod_hue(img, **default_args)
    #aug_img, bbox = mod_hue(img, bbox, **default_args)

    #調整saturation
    #aug_img = mod_saturation(img, **default_args)
    #aug_img, bbox = mod_saturation(img, bbox, **default_args)

    #調整light
    #aug_img = mod_light(img, **default_args)
    #aug_img, bbox = mod_light(img, bbox, **default_args)

    #水平翻轉
    #aug_img = horizontal_flip(img, **default_args)
    #aug_img, bbox = horizontal_flip(img, bbox, **default_args)

    #垂直翻轉
    #aug_img = vertical_flip(img, **default_args)
    #aug_img, bbox = vertical_flip(img, bbox, **default_args)

    #旋轉角度
    #aug_img = rot_image(img, **default_args)
    #aug_img, bbox = rot_image(img, bbox, **default_args)

    #等比例resize至指定大小
    #aug_img = resize_img(img, **default_args)
    #aug_img, bbox = resize_img(img, bbox, **default_args)

    #補形狀至指定大小
    #aug_img = padding_img(aug_img, **default_args)
    #aug_img, bbox = padding_img(aug_img, bbox, **default_args)

    #隨機縮小 N~1倍
    #aug_img = random_zoom_out(img, **default_args)
    #aug_img, bbox = random_zoom_out(img, bbox, **default_args)

    #隨機選擇augmentation方法
    aug_img = rand_aug_image(img, **default_args)
    #aug_img, bbox = rand_aug_image(img, bbox, **default_args)
    print(bbox)

    draw_rect(aug_img, bbox)
    cv2.imshow('img', img)
    cv2.imshow('aug img', aug_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()