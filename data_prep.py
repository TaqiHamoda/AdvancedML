import cv2
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm

from pyxtf import xtf_read, concatenate_channel, XTFHeaderType

INPUT_DIR = Path("data/dataset")
OUTPUT_DIR = Path("data/processed")
NUM_THREADS = 70

NADIR_SIZE = 200              # Number of bins to exclude around the nadir (center) to avoid noise
INPUT_SHAPE = (768, 768)      # Original shape of the sonar tiles
DOWNSAMPLE_FACTOR = 2         # Factor by which to downsample the tiles


def downsample(data, factor=2):
    # Use mean pooling to downsample instead of interpolation to preserve sonar characteristics
    h, w = data.shape
    reshaped = data.reshape(h // factor, factor, w // factor, factor)
    return reshaped.mean(axis=(1, 3))


def xtf_to_data(xtf_path: Path) -> np.ndarray:
    try:
        (fh, p) = xtf_read(xtf_path)

        # Toggle concatenate_channel weighted argument to fit your data requirements.
        port = concatenate_channel(p[XTFHeaderType.sonar], file_header=fh, channel=0, weighted=False)
        stbd = concatenate_channel(p[XTFHeaderType.sonar], file_header=fh, channel=1, weighted=False)
        data = np.concatenate((port, stbd), axis=1)

        # Clip to range (max cannot be used due to outliers)
        data = data.clip(0,  2 ** 16 - 1)
        data = np.log1p(data, dtype=np.float32)
        data = np.clip(data, 0, np.percentile(data, 90))  # Clip extreme outliers
        data /= np.max(data)  # Scale to [0,1]

        return data
    except Exception as e:
        return None


def cut_tiles(xtf_path: Path, output_path: Path, prefix: str) -> np.ndarray:
    data = xtf_to_data(xtf_path)
    if data is None:
        return

    middle = data.shape[1] // 2
    nadir_range = (middle - NADIR_SIZE, middle + NADIR_SIZE)

    padding = (INPUT_SHAPE[0] // 2, INPUT_SHAPE[1] // 2)

    row_range = (padding[0], data.shape[0] - padding[0])
    port_range = (padding[1], nadir_range[0] - padding[1])
    stbd_range = (nadir_range[1] + padding[1], data.shape[1] - padding[1])

    cols = list(range(port_range[0], port_range[1], padding[1])) + list(range(stbd_range[0], stbd_range[1], padding[1]))
    for row in range(row_range[0], row_range[1], padding[0]):
        for col in cols:
            output_file = output_path / f"{prefix}_{row}_{col}.npz"
            if output_file.exists():
                continue

            tile = data[row - padding[0]:row + padding[0], col - padding[1]:col + padding[1]]
            distances = np.ones_like(tile) * np.arange(col - padding[1], col + padding[1])

            tile = downsample(tile, factor=DOWNSAMPLE_FACTOR)
            distances = downsample(distances, factor=DOWNSAMPLE_FACTOR)

            # Normalize distances to [0, 2] (encodes port vs stbd)
            distances /= middle

            if tile.shape[0] != tile.shape[1]:
                continue
            elif tile.shape[0] != INPUT_SHAPE[0] // DOWNSAMPLE_FACTOR:
                continue

            np.savez_compressed(output_file, data=tile, distances=distances)
            cv2.imwrite(str(output_file.with_suffix('.png')), (tile * 255).astype(np.uint8))


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    waterfalls = []
    for directory in INPUT_DIR.iterdir():
        for file in directory.iterdir():
            if file.suffix.lower() != ".xtf":
                continue

            prefix = f"{directory.name}_{file.stem}"
            waterfalls.append((file, prefix))

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        list(tqdm(
            executor.map(
                lambda x: cut_tiles(x[0], OUTPUT_DIR, x[1]),
                waterfalls
            ),
            total=len(waterfalls),
            desc="Processing tiles in dataset"
        ))
