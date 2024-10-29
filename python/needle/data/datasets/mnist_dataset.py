from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.transforms = transforms
        import gzip, struct
        with gzip.open(image_filename, 'rb') as image_gz:
            magic, num_images, rows, cols = struct.unpack('>IIII', image_gz.read(16))
            X = np.frombuffer(image_gz.read(), dtype=np.uint8). \
                reshape(num_images, rows*cols)
            X = X.astype(np.float32) / 255.0
            self.X = X
        with gzip.open(label_filename, 'rb') as label_gz:
            magic, num_labels = struct.unpack('>II', label_gz.read(8))
            y = np.frombuffer(label_gz.read(), dtype=np.uint8).reshape(num_labels)
            self.y = y
        ### END YOUR SOLUTION


    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        img = self.X[index].reshape(28, 28, 1)
        return self.apply_transforms(img), self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION