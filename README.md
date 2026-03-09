# iqadatasets

Loading Image Quality Assessment (IQA) datasets from our private server. This package provides easy-to-use builders for several popular IQA datasets, returning `tf.data.Dataset` objects.

## Installation

You can install `iqadatasets` using `uv`:

```bash
uv add iqadatasets
```

Or using `pip`:

```bash
pip install iqadatasets
```

## Available Datasets

The following datasets are supported:

- **LIVE**
- **TID2008**
- **TID2013**
- **KADIK10K**
- **PIPAL**
- **BAPPS**

## How to use

All datasets follow a similar API. You just need to provide the path to the root directory of the dataset.

### Example: LIVE Dataset

```python
from iqadatasets import LIVE
from pathlib import Path

# Initialize the dataset builder
path = Path("/path/to/LIVE/")
builder = LIVE(path)

# Get the tf.data.Dataset object
ds = builder.dataset

# Use it in your training loop or iterate through it
for ref, dist, score in ds.take(1):
    print(ref.shape, dist.shape, score)
```

### Example: TID2008 Dataset

Some datasets like TID2008 allow for excluding specific images, distortions, or intensities:

```python
from iqadatasets import TID2008

builder = TID2008(
    path="/path/to/TID2008/",
    exclude_imgs=[24, 25], # Exclude specific reference images
    exclude_dist=[1, 2, 3], # Exclude specific distortion types
)

ds = builder.dataset
```

## Dataset Structure

The `dataset` property returns a `tf.data.Dataset` where each element is a tuple of:
1. `Reference Image`: (Tensor) The original, undistorted image.
2. `Distorted Image`: (Tensor) The distorted version of the image.
3. `Score`: (Tensor) The MOS/DMOS score for the image pair.
