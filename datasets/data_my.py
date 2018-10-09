import os
import json
import glob
import cv2
import numpy as np


def read_data(data_dir, img_size=(416, 416), pixel_per_grid=32):
    """
    욜로에서 쓰일 데이터를 읽어오는 함수
    Load data for yolo detector
    :param data_dir: str, 데이터 디렉토리
    :param img_size: tuple, 리사이즈 하기 위해 알아야 하는 이미지 사이즈
    :param pixel_per_grid: int, 그리드 당 픽셀 갯수
    :return: x_set : np.ndarray, shape(N, H, W, C)
              y_set : np.ndarray, shape(N, grid_H, grid_W, 앵커, 5 + 클래스 갯수)
    """
    # 이미지 디렉토리
    img_dir = os.path.join(data_dir, 'images')
    # 클래스 디렉토리
    class_map_path = os.path.join(data_dir, 'classes.json')
    # 앵커 디렉토리
    anchors_path = os.path.join(data_dir, 'anchors.json')
    # 클래스와 앵커는 이미 만들어져 있음

    # classes.json file 읽어드리는 부분
    with open(class_map_path, 'r') as f:
        class_map = json.load(f)
        # print(type(class_map)) # 클래스는 dict 임
        print("classes : ", class_map)

    # anchors.json file 읽어드리는 부분
    with open(anchors_path, 'r') as f:
        anchors = json.load(f)
        # print(type(anchors)) # 앵커는 list 임 2차원 배열
        print("anchors : ", anchors)

    num_classes = len(class_map)
    print("num_of_classes : ", num_classes)
    # gird 의 세로, 가로갯수
    grid_h = img_size[0] // pixel_per_grid
    grid_w = img_size[1] // pixel_per_grid
    print("gird_h 가로 : ", grid_h, "\ngrid_w 세로 : ", grid_w)

    # 이미지 path 들의 배열 .jpg 로 받아오기 (다른값도 가능)
    img_paths = []
    img_paths.extend(glob.glob(os.path.join(img_dir, '*.jpg')))

    # print(img_paths)
    # print(len(img_paths))

    images = []
    labels = []

    # 이미지 패스를 돌면서
    for img_path in img_paths:
        # 이미지 읽어오기
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # 이미지 어래이로 변경
        img = np.array(img, dtype=np.float32)
        img_origin_size = img.shape[:2]
        img = cv2.resize(img, (img_size[1], img_size[0]))
        cv2.imshow('result', img)
        cv2.waitKey(0)

    return img_dir


if __name__ == "__main__":
    data_directory = '../data/face/train'
    read_data(data_directory)





