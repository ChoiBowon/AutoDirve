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
    # 어노테이션 디렉토리
    anno_dir = os.path.join(data_dir, 'annotations')
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

    images = []
    labels = []

    # 이미지 패스를 돌면서
    # for img_path in img_paths:
    for img_path in img_paths:
        # 이미지 읽어오기
        img = cv2.imread(img_path)
        # 이미지 어래이로 변경
        img = np.array(img, dtype=np.float32)
        img_origin_size = img.shape[:2]
        img = cv2.resize(img, (img_size[1], img_size[0]))

        images.append(img)
        # 확인 해 보는 부분
        # cv2.imshow just read unit8 : issue 01
        # cv2.imshow('result', images[0].astype(np.uint8))
        # cv2.waitKey(0)

        # load bboxes and reshape for yolo model

        # os.path.basename : 맨마지막 파일 반환
        # os.path.splitext : 입력 받은 경로를 확장자 부분과 그 외의 부분으로 나눕니다. (extension : 경로확장자)

        # os.path.basename(img_path).split('.')[0]
        # os.path.splitext(os.path.basename(img_path))[0]
        # 위의 2개가 여기서는 같은 역할을 한다.

        # 파일 이름만 가져와서
        name = os.path.splitext(os.path.basename(img_path))[0]
        # 뒤에 어노 붙여서 path 만든다.
        anno_path = os.path.join(anno_dir, '{}.anno'.format(name))

        # 어노테이션 읽는 부분
        with open(anno_path, 'r') as f:
            anno = json.load(f)

        label = np.zeros((grid_h, grid_w, len(anchors), 5 + num_classes))

        # 어노 내부에서
        # class 에 key ( 번호 ) 와 value ( object 이름 )
        for c_idx, c_name in class_map.items():
            # 이미지를 받았는데 클래스가 없을 수도 있자너
            if c_name not in anno:
                continue
            # 어노테이션의 class 가 있다면, 좌상점, 우하점 받아옴
            for x_min, y_min, x_max, y_max in anno[c_name]:

                # original height, original weight
                origin_height, origin_weight = img_origin_size
                # 0 - 1 로 scaling 해 주는 부분
                x_min, y_min, x_max, y_max = x_min / origin_weight, y_min / origin_height, \
                                             x_max / origin_weight, y_max / origin_height
                # print("before : ", c_name, x_min, y_min, x_max, y_max)

                # np.clip ( a, b, c ) : a 가 b 와 c 의 범위로 들어가는 것! : 음수인 부분 잡아준다. 근데 음수?
                x_min, y_min, x_max, y_max = np.clip([x_min, y_min, x_max, y_max], 0, 1)
                # print("afters : ", c_name, x_min, y_min, x_max, y_max)

                # anchor box 를 각 이미지에 맞게 조정 0~1 scaling
                anchor_boxes = np.array(anchors) / np.array([origin_weight, origin_height])

                # print(anchor_boxes)
                # 바운딩 박스 만든다.
                box_wh = np.array([x_max - x_min, y_max - y_min])
                best_iou = 0
                best_anchor = 0

                # enumerate : 앞에 숫자를 붙여서 세면서 뒤의 내용을 같이 보내주는 역할

                # anchor_boxes 가
                # [[0.17380581 0.31096117]
                #  [0.2302566  0.40886933]
                #  [0.24987417 0.44630213]
                #  [0.09947592 0.17172213]
                #  [0.16759337 0.2942268 ]]
                # 라면
                # 0, [0.17380581 0.31096117]
                # 1, [0.2302566  0.40886933]
                # 2, [0.24987417 0.44630213]
                # 3, [0.09947592 0.17172213]
                # 4, [0.16759337 0.2942268 ]
                # 요렇게 변환

                # enumerate 하는 부분
                for k, anchor in enumerate(anchor_boxes):

                    intersect_wh = np.maximum(np.minimum(box_wh, anchor), 0.0)
                    # 겹치는 부분의 넓이를 구한다.
                    intersect_area = intersect_wh[0] * intersect_wh[1]
                    # bounding box 부분의 넓이를 구한다.
                    box_area = box_wh[0] * box_wh[1]
                    # anchor_area 부분의 넓이를 구한다.
                    anchor_area = anchor[0] * anchor[1]
                    # intersection over union = 겹치는 넓이 / 전체 넓이
                    iou = intersect_area / (box_area + anchor_area - intersect_area)
                    if iou > best_iou:
                        best_iou = iou
                        best_anchor = k

                best_anchor = best_anchor
                # print("best_anchor", best_anchor)
                # 중심 grid 부분 찾는 부분
                cx = int(np.floor(0.5 * (x_min + x_max) * grid_w))
                cy = int(np.floor(0.5 * (y_min + y_max) * grid_h))
                print(cx, cy)
                label[cy, cx, best_anchor, 0:4] = [x_min, y_min, x_max, y_max]

                # confidence
                label[cy, cx, best_anchor, 4] = 1.0

                # probability
                label[cy, cx, best_anchor, 5 + int(c_idx)] = 1.0
        labels.append(label)

    x_set = np.array(images, dtype=np.float32)
    y_set = np.array(labels, dtype=np.float32)

    return x_set, y_set


if __name__ == "__main__":
    # data_directory = '../data/face/train'
    # read_data(data_directory)
    #
    grid_w = 13
    grid_h = 13
    grid_wh = np.reshape([13, 13], [1, 1, 1, 1, 2]).astype(np.float32)
    print(grid_wh)
    # transpose 가 axis 를 변경하는 것인데,
    cxcy = np.transpose([np.tile(np.arange(13), 13), np.repeat(np.arange(grid_h), grid_w)])
    #
    arr = np.array([1, 2, 3, 4])
    rep = np.repeat(arr, 3)
    print(rep)
    tras = np.transpose([(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3),
                         (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3)])
    print(tras)
    cxcy = np.reshape(cxcy, (1, grid_h, grid_w, 1, 2))
    # print(cxcy)

    # print(cxcy[..., 0:2])




