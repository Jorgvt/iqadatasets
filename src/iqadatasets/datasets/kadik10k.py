
__all__ = ['KADIK10K']

from pathlib import Path
from typing import List

import pandas as pd
import tensorflow as tf
import cv2

class KADIK10K():
    """Builder for the KADIK10K dataset"""

    def __init__(self,
                 path, # Path to the root directory of the dataset.
                 exclude_imgs: List[int] = None, # Image ID's to exclude.
                 exclude_dist: List[int] = None, # Distortion type ID's to exclude.
                 exclude_ints: List[int] = None, # Distortion intensity ID's to exclude.
                 exclude_identical_pairs: bool = False, # Whether to exclude pairs where the distorted image is pixel-identical to the reference (distortion type 1, intensity 1).
                 num_parallel_calls: int = tf.data.AUTOTUNE, # Number of parallel calls when loading the images.
                 ):
        self.path_root = Path(path) if isinstance(path, str) else path
        self.path_csv = self.path_root/"dmos.csv"
        self.path_images = self.path_root/"images"
        self.data = self.load_data(self.path_csv, exclude_imgs, exclude_dist, exclude_ints, exclude_identical_pairs)
        self.paths_ref = [str(self.path_images/p) for p in self.data["ref_img"]]
        self.paths_dist = [str(self.path_images/p) for p in self.data["dist_img"]]
        self.num_parallel_calls = num_parallel_calls

    @property
    def dataset(self):
        """tf.data.Dataset object built from the TID2013 dataset."""
        return tf.data.Dataset.from_tensor_slices((self.paths_ref, self.paths_dist, self.data["dmos"]))\
                              .map(self.preprocess, num_parallel_calls=self.num_parallel_calls)

    @staticmethod
    def preprocess(path_ref,
                   path_dist,
                   mos,
                   ):
        img_ref = tf.io.read_file(path_ref)
        img_dist = tf.io.read_file(path_dist)

        img_ref = tf.image.decode_png(img_ref, channels=3)
        img_dist = tf.image.decode_png(img_dist, channels=3)

        img_ref = tf.image.convert_image_dtype(img_ref, dtype=tf.float32)
        img_dist = tf.image.convert_image_dtype(img_dist, dtype=tf.float32)

        return img_ref, img_dist, mos

    def load_data(self,
                  path,
                  exclude_imgs,
                  exclude_dist,
                  exclude_ints,
                  exclude_identical_pairs,
                  ):
        data = pd.read_csv(path)
        # Parse reference image ID, distortion type, and intensity from filenames.
        # ref_img format:  I01.png           → ref_id=1
        # dist_img format: I01_01_01.png     → ref_id=1, distortion=1, intensity=1
        data["ref_id"] = data["ref_img"].str.extract(r"I(\d+)\.png").astype(int)
        parsed = data["dist_img"].str.extract(r"I\d+_(\d+)_(\d+)\.png")
        data["distortion"] = parsed[0].astype(int)
        data["intensity"] = parsed[1].astype(int)
        if exclude_imgs is not None:
            data = data[~data["ref_id"].isin(exclude_imgs)]
        if exclude_dist is not None:
            data = data[~data["distortion"].isin(exclude_dist)]
        if exclude_ints is not None:
            data = data[~data["intensity"].isin(exclude_ints)]
        if exclude_identical_pairs:
            # The following distortion/intensity combinations produce images
            # pixel-identical to the reference (verified by exhaustive pixel scan).
            # Including them causes dist=0 → infinite gradients via d/dx sqrt(0).
            identical_mask = (
                ((data["distortion"] == 1)  & (data["intensity"] == 1)) |
                ((data["distortion"] == 3)  & (data["intensity"] == 1)) |
                ((data["distortion"] == 8)  & (data["intensity"] == 1)) |
                ((data["distortion"] == 18) & (data["intensity"] == 3))
            )
            data = data[~identical_mask]
        return data
