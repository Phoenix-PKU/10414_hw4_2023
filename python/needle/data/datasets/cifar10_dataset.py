import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.X, self.y = [], []
        self.transforms = transforms
        self.p = p
        def unpickle(file):
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict
        for file in os.listdir(base_folder):
            if 'data_batch' in file and train:
                data = unpickle(base_folder + '/' + file)
                self.X.append(data[b'data'])
                self.y.append(data[b'labels'])
            elif 'test_batch' in file and not train:
                data = unpickle(base_folder + '/' + file)
                self.X.append(data[b'data'])
                self.y.append(data[b'labels'])

        self.X = np.concatenate(self.X, axis = 0).reshape((-1, 3, 32, 32)) / 255
        self.y = np.concatenate(self.y, axis = None)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        if self.transforms:
            image = np.array([self.apply_transforms(img) for img in self.X[index]])
        else:
            image = self.X[index]
        return image, self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION
