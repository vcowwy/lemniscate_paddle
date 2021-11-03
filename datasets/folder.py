from paddle.vision import datasets


class ImageFolderInstance(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        """if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index"""
        return img, target, index
