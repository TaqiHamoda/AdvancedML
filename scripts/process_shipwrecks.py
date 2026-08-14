from typing import List, Tuple

import cv2
import numpy as np

from tqdm import tqdm
from pathlib import Path

NUM_THREADS = 70

DATA_DIR = Path("data")

WF_DIR = DATA_DIR / "labelled/AI4Shipwrecks/waterfalls/"
LABEL_DIR = DATA_DIR / "labelled/AI4Shipwrecks/labels/"

IMG_DIR = DATA_DIR / "labelled/AI4Shipwrecks/images/"
MASK_DIR = DATA_DIR / "labelled/AI4Shipwrecks/masks/"

CROP_DIM = 640  # Original shape of the sonar tiles

BG_SAMPLE_COUNT = 10
SHIP_SAMPLE_COUNT = 5


def get_locs(mask: np.ndarray, sample_count: int) -> List[Tuple[int, int]]:
    ys, xs = np.where(mask)

    all_locs = np.array([(ys[i], xs[i]) for i in range(len(ys))])
    if all_locs.shape[0] <= sample_count:
        return all_locs.tolist()

    locs = [(np.median(ys), np.median(xs)), ]
    dists = np.linalg.norm(all_locs - np.array(locs[-1]), axis=1)
    while len(locs) - 1 < sample_count:
        ind = np.argmax(dists)
        locs.append((ys[ind], xs[ind]))

        dists = np.min(
            np.vstack([
                dists,
                np.linalg.norm(all_locs - np.array(locs[-1]), axis=1)
            ]).T,
            axis=1
        )

    return locs[1:]  # Discards center view


def cut_tiles(waterfall_path: Path, label_path: Path, prefix: str):
    waterfall = cv2.imread(str(waterfall_path), cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)

    if waterfall.shape[0] < CROP_DIM or waterfall.shape[1] < CROP_DIM:
        return

    pixel_locs = get_locs(label == 0, BG_SAMPLE_COUNT) + get_locs(label == 1, SHIP_SAMPLE_COUNT)
    for row, col in pixel_locs:
        output_name = f"{prefix}_{row}_{col}.png"

        row_start = max(row - CROP_DIM, 0)
        if row_start == 0:
            row_end = row_start + CROP_DIM
        else:
            row_end = row

        col_start = max(col - CROP_DIM, 0)
        if col_start == 0:
            col_end = col_start + CROP_DIM
        else:
            col_end = col

        img = waterfall[row_start:row_end, col_start:col_end]
        mask = label[row_start:row_end, col_start:col_end]

        cv2.imwrite(str(IMG_DIR / output_name), img)
        cv2.imwrite(str(MASK_DIR / output_name), mask)


if __name__ == "__main__":
    IMG_DIR.mkdir(exist_ok=True)
    MASK_DIR.mkdir(exist_ok=True)

    for file in tqdm(list(WF_DIR.glob('*.png'))):
        cut_tiles(WF_DIR / file.name, LABEL_DIR / file.name, file.stem)
