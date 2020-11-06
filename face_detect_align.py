'''
Description: Align and unify all faces in dataset according to the distance between the eyes
Author: shaonianruntu
Github: https://github.com/shaonianruntu
Date: 2020-11-04 11:29:35
LastEditTime: 2020-11-06 21:39:24
'''
import argparse
import os
import math
import numpy as np
from collections import defaultdict

import face_recognition
import cv2
from PIL import Image, ImageDraw
# from matplotlib.pyplot import imshow


def detect_landmark(image_array, model_type="large"):
    """ return landmarks of a given image array
    :param image_array: numpy array of a single image
    :param model_type: 'large' returns 68 landmarks; 'small' return 5 landmarks
    :return: dict of landmarks for facial parts as keys and tuple of coordinates as values
    """
    face_landmarks_list = face_recognition.face_landmarks(image_array,
                                                          model=model_type)
    if len(face_landmarks_list) > 0:
        face_landmarks_list = face_landmarks_list[0]
    else:
        return None
    return face_landmarks_list


def align_face(image_array, landmarks):
    """ align faces according to eyes position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    rotated_img:  numpy array of aligned image
    eye_center: tuple of coordinates for eye center
    angle: degrees of rotation
    """
    # get list landmarks of left and right eye
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    # calculate the mean point of landmarks of left and right eye
    left_eye_center = np.mean(left_eye, axis=0).astype("int")
    right_eye_center = np.mean(right_eye, axis=0).astype("int")
    # compute the angle between the eye centroids
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    # compute angle between the line of 2 centeroids and the horizontal line
    angle = math.atan2(dy, dx) * 180. / math.pi
    # calculate the center of 2 eyes
    eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                  (left_eye_center[1] + right_eye_center[1]) // 2)
    # at the eye_center, rotate the image by the angle
    rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    rotated_img = cv2.warpAffine(image_array, rotate_matrix,
                                 (image_array.shape[1], image_array.shape[0]))
    return rotated_img, eye_center, angle


def rotate(origin, point, angle, row):
    """ rotate coordinates in image coordinate system
    :param origin: tuple of coordinates,the rotation center
    :param point: tuple of coordinates, points to rotate
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated coordinates of point
    """
    x1, y1 = point
    x2, y2 = origin
    y1 = row - y1
    y2 = row - y2
    angle = math.radians(angle)
    x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
    y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
    y = row - y
    return int(x), int(y)


def rotate_landmarks(landmarks, eye_center, angle, row):
    """ rotate landmarks to fit the aligned face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param eye_center: tuple of coordinates for eye center
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated_landmarks with the same structure with landmarks, but different values
    """
    rotated_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            rotated_landmark = rotate(origin=eye_center,
                                      point=landmark,
                                      angle=angle,
                                      row=row)
            rotated_landmarks[facial_feature].append(rotated_landmark)
    return rotated_landmarks


def corp_face(image_array, size, landmarks):
    """ crop face according to eye,mouth and chin position
    :param image_array: numpy array of a single image
    :param size: single int value, size for w and h after crop
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    cropped_img: numpy array of cropped image
    left, top: left and top coordinates of cropping
    """
    x_min = np.min(landmarks['chin'], axis=0)[0]
    x_max = np.max(landmarks['chin'], axis=0)[0]
    x_center = (x_max - x_min) / 2 + x_min
    left, right = (x_center - size / 2, x_center + size / 2)

    eye_landmark = landmarks['left_eye'] + landmarks['right_eye']
    eye_center = np.mean(eye_landmark, axis=0).astype("int")
    lip_landmark = landmarks['top_lip'] + landmarks['bottom+lip']
    lip_center = np.mean(lip_landmark, axis=0).astype("int")
    mid_part = lip_center[1] - eye_center[1]
    top, bottom = eye_center[1] - (size - mid_part) / 2, lip_center[1] + (
        size - mid_part) / 2

    pil_img = Image.fromarray(image_array)
    left, top, right, bottom = [int(i) for i in [left, top, right, bottom]]
    cropped_img = pil_img.crop((left, top, right, bottom))
    cropped_img = np.array(cropped_img)
    return cropped_img, left, top


def resize_and_crop(image_array, size, eye_dis, left_eye_x, left_eye_y,
                    landmarks):
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']

    left_eye_center = np.mean(left_eye, axis=0).astype("int")
    right_eye_center = np.mean(right_eye, axis=0).astype("int")
    raw_eye_distance = np.sqrt((left_eye_center[0] - right_eye_center[0])**2 +
                               (left_eye_center[1] - right_eye_center[1])**2)

    ratio = eye_dis / raw_eye_distance
    h, w, c = image_array.shape

    pro_h, pro_w = int(h * ratio), int(w * ratio)
    image_array = cv2.resize(image_array, (pro_w, pro_h),
                             interpolation=cv2.INTER_CUBIC)

    transferred_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            transferred_landmark = (int(landmark[0] * ratio),
                                    int(landmark[1] * ratio))
            transferred_landmarks[facial_feature].append(transferred_landmark)

    img = np.ones((size[0], size[1], 3)) * 255  # 0 250  1 200
    left_eye = transferred_landmarks['left_eye']
    left_eye_center = np.mean(left_eye, axis=0).astype("int")
    h, w, c = image_array.shape
    padding_x = padding_y = 0
    left_upper_x = left_eye_center[0] - left_eye_x
    if left_upper_x < 0:
        padding_x = abs(left_upper_x)
        left_upper_x = 0
    left_upper_y = left_eye_center[1] - left_eye_y
    if left_upper_y < 0:
        padding_y = abs(left_upper_y)
        left_upper_y = 0
    right_low_x = left_eye_center[0] + left_eye_y
    right_low_x = right_low_x if right_low_x < w else w
    right_low_y = left_eye_center[1] + left_eye_y
    right_low_y = right_low_y if right_low_y < h else h
    img[padding_y:padding_y + right_low_y - left_upper_y,
        padding_x:padding_x + right_low_x -
        left_upper_x, :] = image_array[left_upper_y:right_low_y,
                                       left_upper_x:right_low_x, :]
    return img, transferred_landmarks, padding_x, padding_y


def transfer_landmark(landmarks, left, top):
    """transfer landmarks to fit the cropped face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param left: left coordinates of cropping
    :param top: top coordinates of cropping
    :return: transferred_landmarks with the same structure with landmarks, but different values
    """
    transferred_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            transferred_landmark = (landmark[0] + left, landmark[1] + top)
            transferred_landmarks[facial_feature].append(transferred_landmark)
    return transferred_landmarks


def face_preprocess(image,
                    landmark_model_type='large',
                    crop_size=[250, 200],
                    eye_dis=50,
                    left_eye_x=75,
                    left_eye_y=125):
    """ for a given image, do face --+-- and crop face
    :param image: numpy array of a single image
    :param landmark_model_type: 'large' returns 68 landmarks; 'small' return 5 landmarks
    :param crop_size: single int value, size for w and h after crop
    :return:
    cropped_face: image array with face aligned and cropped
    transferred_landmarks: landmarks that fit cropped_face
    """
    # detect landmarks
    face_landmarks_dict = detect_landmark(image_array=image,
                                          model_type=landmark_model_type)
    if face_landmarks_dict is None:
        return None
    # rotate image array to align face
    aligned_face, eye_center, angle = align_face(image_array=image,
                                                 landmarks=face_landmarks_dict)
    # rotate landmarks coordinates to fit the aligned face
    rotated_landmarks = rotate_landmarks(landmarks=face_landmarks_dict,
                                         eye_center=eye_center,
                                         angle=angle,
                                         row=image.shape[0])
    # crop face according to landmarks
    # cropped_face, left, top = corp_face(image_array=aligned_face, size=crop_size, landmarks=rotated_landmarks)
    cropped_face, transferred_landmarks, left, top = resize_and_crop(
        image_array=aligned_face,
        size=crop_size,
        eye_dis=eye_dis,
        left_eye_x=75,
        left_eye_y=125,
        landmarks=rotated_landmarks)
    # transfer landmarks to fit the cropped face
    transferred_landmarks = transfer_landmark(landmarks=transferred_landmarks,
                                              left=left,
                                              top=top)
    return cropped_face, transferred_landmarks


def visualize_landmark(image_array, landmarks):
    """ plot landmarks on image
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values+
    :return: plots of images with landmarks on
    """
    origin_img = Image.fromarray(np.uint8(image_array))
    draw = ImageDraw.Draw(origin_img)
    for facial_feature in landmarks.keys():
        draw.point(landmarks[facial_feature], fill=(0, 0, 255))
    # imshow(origin_img)
    img_vis = np.array(origin_img)
    return img_vis


def visualize_eyes(image_array, landmarks):
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    left_eye_center = np.mean(left_eye, axis=0).astype("int")
    right_eye_center = np.mean(right_eye, axis=0).astype("int")

    for point in [left_eye_center, right_eye_center]:
        cv2.circle(img, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
    return img


def save_landmark(image_array, landmarks, img_name):
    """ plot landmarks on image
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return: plots of images with landmarks on
    """
    for facial_feature in landmarks.keys():
        for point in landmarks[facial_feature]:
            cv2.circle(image_array, (int(point[0]), int(point[1])), 1,
                       (0, 0, 255), 4)
    cv2.imwrite(img_name, image_array)


def is_image_file(filename):
    IMG_EXTENSIONS = [
        '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff',
        'webp'
    ]
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dlib_path',
                        type=str,
                        default='/data/fangnan/dlib/',
                        help="dataset_path")
    parser.add_argument('--dlib_mode',
                        type=str,
                        default='81',
                        choices=['68', '81'],
                        help="dlib detection fineness")
    parser.add_argument('--dlib_five',
                        action='store_true',
                        default=False,
                        help="just dlib five sense organs")

    parser.add_argument('--photo_path',
                        type=str,
                        default='/data/fangnan/photostick/CUHK4/photos',
                        help="dataset_path")
    parser.add_argument('--save_path',
                        type=str,
                        default='./dlib',
                        help="dataset_path")

    parser.add_argument('--landmark_visible',
                        action='store_true',
                        default=False,
                        help="visible all face key points")
    parser.add_argument('--eyes_visible',
                        action='store_true',
                        default=False,
                        help="visible two eyes")

    parser.add_argument('--gpu', default='0', help='GPU Number')
    opts = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu

    if not os.path.exists(opts.save_path):
        os.mkdir(opts.save_path)

    for name in sorted(os.listdir(opts.photo_path)):
        faces_path = os.path.join(opts.photo_path, name)
        if not is_image_file(faces_path):
            continue
        save_path = os.path.join(opts.save_path, name)

        img = cv2.imread(faces_path, cv2.IMREAD_COLOR)
        face, landmarks = face_preprocess(
            image=img,
            landmark_model_type='large',
        )
        if face is None:
            print("No face dectected")
        else:
            print(name)

        if opts.landmark_visible:
            face = visualize_landmark(image_array=face, landmarks=landmarks)
        if opts.eyes_visible:
            face = visualize_eyes(image_array=face, landmarks=landmarks)

        cv2.imwrite(save_path, face)
