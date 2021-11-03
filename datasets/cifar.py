from __future__ import print_function

from PIL import Image
from paddle.vision import datasets
import numpy as np


URL_PREFIX = 'https://dataset.bj.bcebos.com/cifar/'
CIFAR100_URL = URL_PREFIX + 'cifar-100-python.tar.gz'
CIFAR100_MD5 = 'eb9058c3a382ffc7106e4002c42a8d85'

MODE_FLAG_MAP = {
    'train10': 'data_batch',
    'test10': 'test_batch',
    'train100': 'train',
    'test100': 'test'
}


class CIFAR10Instance(datasets.Cifar10):
    def __getitem__(self, index):
        image, label = self.data[index]
        image = np.reshape(image, [3, 32, 32])
        image = image.transpose([1, 2, 0])

        if self.backend == 'pil':
            image = Image.fromarray(image.astype('uint8'))

        if self.transform is not None:
            image = self.transform(image)

        if self.backend == 'pil':
            return image, label.astype('int64')

        return image.astype(self.dtype), np.array(label).astype('int64'), index


class CIFAR100Instance(CIFAR10Instance):
    """
    base_folder = 'cifar-100-python'
    URL_PREFIX = 'https://dataset.bj.bcebos.com/cifar/'
    url = URL_PREFIX + 'cifar-100-python.tar.gz'
    filename = 'cifar-100-python.tar.gz'
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [['train', '16019d7e3df5f24257cddd939b257f8d']]
    test_list = [['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc']]
    """
    def __init__(self,
                 data_file=None,
                 mode='train',
                 transform=None,
                 download=True,
                 backend=None):
        super(CIFAR100Instance, self).__init__(data_file, mode, transform, download, backend)

    def _init_url_md5_flag(self):
        self.data_url = CIFAR100_URL
        self.data_md5 = CIFAR100_MD5
        self.flag = MODE_FLAG_MAP[self.mode + '100']
