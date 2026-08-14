import cv2
import numpy as np

from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from pathlib import Path
from typing import Tuple

from pyxtf import xtf_read, concatenate_channel, XTFHeaderType

NUM_THREADS = 70

DATA_DIR = Path("data")
INPUT_DIR = DATA_DIR / "dataset"
OUTPUT_DIR = DATA_DIR / "processed"

NADIR_SIZE = 200              # Number of bins to exclude around the nadir (center) to avoid noise
INPUT_SHAPE = (768, 768)      # Original shape of the sonar tiles
OVERLAP_FACTOR = 0.75         # Factor by which to overlap the tiles (e.g. 0.75 means 75% overlap)


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


def cut_tiles(xtf_path: Path, output_path: Path, prefix: str) -> Tuple[float, float]:
    data = xtf_to_data(xtf_path)
    if data is None:
        return 0.0, 0.0

    middle = data.shape[1] // 2
    nadir_range = (middle - NADIR_SIZE, middle + NADIR_SIZE)

    padding = (INPUT_SHAPE[0] // 2, INPUT_SHAPE[1] // 2)
    stride = (int(INPUT_SHAPE[0] * (1 - OVERLAP_FACTOR)), int(INPUT_SHAPE[1] * (1 - OVERLAP_FACTOR)))

    row_range = (padding[0], data.shape[0] - padding[0])
    port_range = (padding[1], nadir_range[0] - padding[1])
    stbd_range = (nadir_range[1] + padding[1], data.shape[1] - padding[1])

    cols = list(range(port_range[0], port_range[1], stride[1])) + list(range(stbd_range[0], stbd_range[1], stride[1]))

    mean, std, count = 0.0, 0.0, 0
    for row in range(row_range[0], row_range[1], stride[0]):
        for col in cols:
            output_file = output_path / f"{prefix}_{row}_{col}.npz"
            if output_file.exists():
                continue

            tile = data[row - padding[0]:row + padding[0], col - padding[1]:col + padding[1]]
            distances = np.ones_like(tile) * np.arange(col - padding[1], col + padding[1])

            # Normalize distances to [0, 2] (encodes port vs stbd)
            distances /= middle

            if tile.shape[0] != tile.shape[1]:
                continue
            elif tile.shape[0] != INPUT_SHAPE[0]:
                continue

            mean += np.mean(tile)
            std += np.std(tile)
            count += 1

            np.savez_compressed(output_file, data=tile, distances=distances)
            cv2.imwrite(str(output_file.with_suffix('.png')), (tile * 255).astype(np.uint8))

    return mean / count if count > 0 else 0, std / count if count > 0 else 0


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    waterfalls = []
    for directory in INPUT_DIR.iterdir():
        for file in directory.iterdir():
            if file.suffix.lower() != ".xtf":
                continue

            prefix = f"{directory.name}_{file.stem}"
            waterfalls.append((file, prefix))

    stats = []
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        list(tqdm(
            executor.map(
                lambda x: stats.append(cut_tiles(x[0], OUTPUT_DIR, x[1])),
                waterfalls
            ),
            total=len(waterfalls),
            desc="Processing tiles in dataset"
        ))

    stats = np.array(stats)
    means, stds = stats[:, 0], stats[:, 1]
    with open(DATA_DIR / "stats.txt", "w") as f:
        f.write(f"Mean: {means[means != 0].mean()}\n")
        f.write(f"Std: {stds[stds != 0].mean()}\n")