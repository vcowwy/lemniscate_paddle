from __future__ import print_function

import paddle
from PIL import Image
from paddle.vision import datasets
import paddle.io as data
import numpy as np


class CIFAR10Instance(datasets.Cifar10):
    """CIFAR10Instance Dataset.
    """

    def __getitem__(self, index):
        image, label = self.data[index]
        image = np.reshape(image, [3, 32, 32])
        image = image.transpose([1, 2, 0])

        if self.backend == 'pil':
            image = Image.fromarray(image.astype('uint8'))
        if self.transform is not None:
            image = self.transform(image)

        if self.backend == 'pil':
            return paddle.to_tensor(image, dtype=self.dtype), paddle.to_tensor(label, dtype=paddle.int64)

        return paddle.to_tensor(image, dtype=self.dtype), paddle.to_tensor(label, dtype=paddle.int64), index
        #return image.astype(self.dtype), np.array(label).astype('int64')


class CIFAR100Instance(CIFAR10Instance):
    """CIFAR100Instance Dataset.

    This is a subclass of the `CIFAR10Instance` Dataset.
    """
    base_folder = 'cifar-100-python'
    URL_PREFIX = 'https://dataset.bj.bcebos.com/cifar/'
    url = URL_PREFIX + 'cifar-100-python.tar.gz'
    filename = 'cifar-100-python.tar.gz'
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [['train', '16019d7e3df5f24257cddd939b257f8d']]
    test_list = [['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc']]
