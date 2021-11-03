from __future__ import print_function

import paddle
from PIL import Image
from paddle.vision import datasets
import paddle.io as data
import numpy as np


class MNISTInstance(datasets.MNIST):
    def __getitem__(self, index):

        image, label = self.images[index], self.labels[index]
        image = np.reshape(image, [28, 28])

        if self.backend == 'pil':
            image = Image.fromarray(image.astype('uint8'), mode='L')

        if self.transform is not None:
            image = self.transform(image)

        if self.backend == 'pil':
            return image, np.array(label).astype('int64')

        return image.astype(self.dtype), label.astype('int64')
