'''
Description: Detect and mark key points of human faces
Author: shaonianruntu
Github: https://github.com/shaonianruntu
Date: 2020-11-04 11:29:35
LastEditTime: 2020-11-06 21:46:32
'''
import argparse
import os
import numpy as np

import dlib
import cv2

DLIB_PATH = '/data/fangnan/dlib/'
# download dlib files from http://dlib.net/files/ and put in your `DLIB_PATH`

def get_landmarks(im, detector, predictor, num):
    try:
        rects = detector(im, 0)
        return np.matrix([[int(p.x), int(p.y)]
                          for p in predictor(im, rects[0]).parts()])
    except IndexError as e:
        """can not find face and any keypoint"""
        print("!!! Can not find a face, and return is zero.")
        return np.matrix([[int(0), int(0)] for i in range(num)])


def dlib_detect_5(dlib_path, img):
    detector = dlib.get_frontal_face_detector()
    predictor_path = str(
        os.path.join(dlib_path, 'shape_predictor_5_face_landmarks.dat'))
    predictor = dlib.shape_predictor(predictor_path)
    return np.array(get_landmarks(img, detector, predictor, 5))


def dlib_detect_68(dlib_path, img):
    detector = dlib.get_frontal_face_detector()
    predictor_path = str(
        os.path.join(dlib_path, 'shape_predictor_68_face_landmarks.dat'))
    predictor = dlib.shape_predictor(predictor_path)
    return np.array(get_landmarks(img, detector, predictor, 68))


def dlib_detect_81(dlib_path, img):
    detector = dlib.get_frontal_face_detector()
    predictor_path = str(
        os.path.join(dlib_path, 'shape_predictor_81_face_landmarks.dat'))
    predictor = dlib.shape_predictor(predictor_path)
    return np.array(get_landmarks(img, detector, predictor, 81))


def calc_eye_point(landmark, is_right_eye=0):
    offset = is_right_eye * 6
    t = np.array([
        landmark[36 + offset],
        landmark[39 + offset],
    ])
    temp = t.mean(axis=0)

    return temp


def get_img5point(landmark):
    return np.array([
        calc_eye_point(landmark, is_right_eye=0),  # Left eye
        calc_eye_point(landmark, is_right_eye=1),  # Right eye
        landmark[30],  # Nose tip
        landmark[48],  # Mouth left corner
        landmark[54],  # Mouth right corner
    ])


def choose_five(landmarks):
    # left eye;right eye;Nose tip;Mouth left corner;Mouth right corner
    return get_img5point(landmarks)


def dlib_landmarks_five(img):
    landmarks = dlib_detect_68(img)
    return choose_five(landmarks)


def visible_points(img, landmarks):
    for point in landmarks:
        cv2.circle(img, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
    return img


def is_image_file(filename):
    IMG_EXTENSIONS = [
        '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff',
        'webp'
    ]
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def dlib_detect_mode(dlib_path, img, mode="68", dlib_five=False):
    if mode == "5":
        landmarks = dlib_detect_5(dlib_path, img)
        return landmarks

    if mode == "68":
        landmarks = dlib_detect_68(dlib_path, img)
    else:
        landmarks = dlib_detect_81(dlib_path, img)

    if dlib_five:
        landmarks = choose_five(landmarks)

    return landmarks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dlib_path',
                        type=str,
                        default='/data/fangnan/dlib/',
                        help="dataset_path")
    parser.add_argument('--dlib_mode',
                        type=str,
                        default='81',
                        choices=['5', '68', '81'],
                        help="dlib detection fineness")
    parser.add_argument('--dlib_five',
                        action='store_true',
                        default=False,
                        help="just dlib five sense organs")
    parser.add_argument('--save_path',
                        type=str,
                        default='./dlib',
                        help="dataset_path")
    parser.add_argument('--photo_path',
                        type=str,
                        default='/data/fangnan/photostick/CUHK4/photos',
                        help="dataset_path")
    parser.add_argument('--gpu', default='0', help='GPU Number')
    opts = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu

    if opts.dlib_five:
        opts.save_path = os.path.join(opts.save_path + opts.dlib_mode, 'five')
    else:
        opts.save_path = os.path.join(opts.save_path + opts.dlib_mode, 'all')

    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)

    for name in sorted(os.listdir(opts.photo_path)):
        faces_path = os.path.join(opts.photo_path, name)
        if not is_image_file(faces_path):
            continue
        save_path = os.path.join(opts.save_path, name)
        print(name)

        img = cv2.imread(faces_path, cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        landmarks = dlib_detect_mode(opts.dlib_path,
                                     img_gray,
                                     mode=opts.dlib_mode,
                                     dlib_five=opts.dlib_five)
        img_vis = visible_points(img, landmarks)
        cv2.imwrite(save_path, img_vis)
        print(save_path)
