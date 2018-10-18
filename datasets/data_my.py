import numpy as np
import utils as ut


class DataSet(object):
    def __init__(self, images, labels, ):
        self.images = images
        self.labels = labels
        self.num_of_examples = images.shape[0]
        self.indices = np.arange(self.num_of_examples)
        self.now_epoch = 0
        self.now_idx = 0

    def next_batch(self, batch_size, shuffle=True):
        """
        입력된 data를 batch size 씩 배츌하는 함수
        :param batch_size: int, 원하는 배치 사이즈
        :param shuffle: bool, 섞는지의 여부
        :return: 배치사이즈의 이미지와 라벨
        batch_images : ndarray, shape : (N, H, W, C)
        batch_labels : ndarray, shape : (N, g_h, g_w, anchors, 5+num_classes)
        """
        # 첫 인덱스 생성
        idx = self.now_idx

        # 처음이라면 섞어준다.
        if idx == 0 and self.now_epoch == 0 and shuffle:
            np.random.shuffle(self.indices)

        #  넘어 갈 때 --> epoch 늘어나는 부분
        if batch_size + idx > self.num_of_examples:
            rest_num = batch_size + idx - self.num_of_examples
            old_indices = self.indices[idx:self.num_of_examples]

            self.now_epoch += 1

            if shuffle:
                np.random.shuffle(self.indices)

            idx = 0
            new_indices = self.indices[idx:rest_num]
            idx = rest_num
            self.now_idx = idx

            old_images = self.images[old_indices]
            new_images = self.images[new_indices]

            batch_images = np.concatenate(old_images, new_images)

            if self.labels is not None:
                old_labels = self.labels[old_indices]
                new_labels = self.labels[new_indices]
                batch_labels = np.concatenate(old_labels, new_labels)
            else:
                batch_labels = None

        # 안넘어갈 때
        else:
            self.now_idx = idx + batch_size
            batch_images = self.images[idx:idx + batch_size]

            if self.labels is not None:
                batch_labels = self.labels[idx:idx + batch_size]
            else:
                batch_labels = None

        return batch_images, batch_labels


if __name__ == "__main__":
    data_directory = '../data/face/train'
    img, lab = ut.read_data(data_directory)
    data_set = DataSet(img, lab)
    print(data_set.next_batch(10))


