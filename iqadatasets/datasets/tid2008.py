# AUTOGENERATED! DO NOT EDIT! File to edit: ../../Notebooks/01_datasets/01_01_tid2008.ipynb.

# %% auto 0
__all__ = ['TID2008']

# %% ../../Notebooks/01_datasets/01_01_tid2008.ipynb 3
from pathlib import Path
from typing import List

import pandas as pd
import tensorflow as tf
import cv2

# %% ../../Notebooks/01_datasets/01_01_tid2008.ipynb 5
class TID2008():
    """Builder for the TID2008 dataset"""

    def __init__(self,
                 path, # Path to the root directory of the dataset.
                 exclude_imgs: List[int] = None, # Image ID's to exclude.
                 exclude_dist: List[int] = None, # Distortion ID's to exclude.
                 exclude_ints: List[int] = None, # Distortion Intensities ID's to exclude.
                 ):
        self.path_root = Path(path) if isinstance(path, str) else path
        self.path_ref = self.path_root/"reference_images"
        self.path_dist = self.path_root/"distorted_images"
        self.path_csv = self.path_root/"image_pairs_mos.csv"
        self.data = self.load_data(self.path_csv, exclude_imgs, exclude_dist, exclude_ints)

    @property
    def dataset(self):
        """tf.data.Dataset object built from the TID2008 dataset."""
        return tf.data.Dataset.from_generator(
                self.data_gen,
                output_signature=(
                    tf.TensorSpec(shape=(384, 512, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(384, 512, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.float32)
                )
            ) 

    def data_gen(self):
        """Dataset generator to build the tf.data.Dataset."""
        for i, row in self.data.iterrows():
            ref, dist, mos = row.Reference, row.Distorted, row.MOS
            dist = cv2.imread(str(self.path_dist/dist))
            dist = cv2.cvtColor(dist, cv2.COLOR_BGR2RGB)/255.0
            ref = cv2.imread(str(self.path_ref/ref))
            ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)/255.0
            yield ref, dist, mos

    def load_data(self,
                  path,
                  exclude_imgs,
                  exclude_dist,
                  exclude_ints,
                  ):
        data = pd.read_csv(self.path_csv, index_col=0)
        data = data[~data.Reference_ID.isin(exclude_imgs)] if exclude_imgs is not None else data
        data = data[~data.Reference_ID.isin(exclude_dist)] if exclude_dist is not None else data
        data = data[~data.Reference_ID.isin(exclude_ints)] if exclude_ints is not None else data
        return data

