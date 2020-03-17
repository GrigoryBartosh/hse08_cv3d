#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

from typing import List, Optional, Tuple

import click
import cv2
import numpy as np
import pims
import sortednp as snp
from scipy.spatial.distance import cdist 

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli

MIN_LEVEL = 1
MAX_LEVEL = 3

POINTS_MIN_DIST = 5

FEATURE_PARAMS = dict(
    maxCorners = 3000,
    qualityLevel = 0.005,
    minDistance = POINTS_MIN_DIST,
    blockSize = 5,
    useHarrisDetector = False,
)

LK_PARAMS = dict(
    winSize = (9, 9),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.001),
    minEigThreshold=0.001
)

CONSIST_LEN = 1
CONSIST_DIST_THRESHOLD = 0.05

DO_TRACK_BACK = True

MIN_TRACK_LEN = 20


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


class MultyLevelCornerStorage():

    def __init__(self, levels: int, base_radius: int):
        self._levels = levels
        self._base_radius = base_radius
        self._corner_count = 0

        self._corners = [{} for _ in range(levels)]
        self._ids = [{} for _ in range(levels)]

    def _init_frame(self, frame: int, level: int) -> None:
        if frame not in self._corners[level]:
            self._corners[level][frame] = np.zeros((0, 2), dtype=np.float32)
            self._ids[level][frame] = np.zeros(0, dtype=np.int32)

    def add(self, corners: np.ndarray, frame: int, level: int,
            ids: Optional[np.ndarray] = None) -> None:
        if ids is None:
            n = len(corners)
            ids = np.arange(n) + self._corner_count
            self._corner_count += n

        self._init_frame(frame, level)

        self._corners[level][frame] = np.concatenate((self._corners[level][frame], corners))
        self._ids[level][frame] = np.concatenate((self._ids[level][frame], ids))

    def corners(self, frame: int, level: int) -> Tuple[np.ndarray, np.ndarray]:
        self._init_frame(frame, level)
        return self._corners[level][frame], self._ids[level][frame]

    def frame_corners(self, frame: int) -> FrameCorners:
        corners = np.zeros((0, 2), dtype=np.float32)
        ids = np.zeros(0, dtype=np.int32)
        radiuses = np.zeros(0, dtype=np.int32)

        for level in range(self._levels):
            if frame not in self._ids[level]:
                continue

            corners = np.concatenate((corners, self._corners[level][frame]))
            ids = np.concatenate((ids, self._ids[level][frame]))

            n = len(self._corners[level][frame])
            radius = (self._base_radius - 1) * 2 ** level + 1
            radiuses = np.concatenate((radiuses, np.ones(n) * radius))

        return FrameCorners(ids, corners, radiuses)


def ft32_to_uint8(img):
    return np.round(img * 255).astype(np.uint8)


def find_new_corners(image: np.ndarray, level: int) -> np.ndarray:
    feature_prams = FEATURE_PARAMS.copy()

    blockSize = feature_prams['blockSize']
    blockSize = (blockSize - 1) * 2 ** level + 1
    feature_prams['blockSize'] = blockSize

    corners = cv2.goodFeaturesToTrack(image, mask=None, **feature_prams)

    if corners is None:
        corners = np.ndarray([], dtype=np.float32)
    else:
        corners = corners.reshape(-1, 2)

    return corners


def move_corners(image_0: np.ndarray, image_1: np.ndarray,
                 corners: np.ndarray, level: int) -> np.ndarray:
    if len(corners) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.int32)

    image_0 = ft32_to_uint8(image_0)
    image_1 = ft32_to_uint8(image_1)

    lk_params = LK_PARAMS.copy()

    win_size = lk_params['winSize'][0]
    win_size = (win_size - 1) * 2 ** level + 1
    lk_params['winSize'] = (win_size, win_size)

    lk_params['maxLevel'] += level

    moved_corners, status, _ = cv2.calcOpticalFlowPyrLK(
        image_0, image_1, corners, None, **lk_params)

    status = status.reshape(-1)

    return moved_corners, status


def check_consist(image_0: np.ndarray, image_1: np.ndarray,
                  corners_0: np.ndarray, corners_1: np.ndarray,
                  level: int) -> np.ndarray:
    good_corners = np.ones(len(corners_0), dtype=np.int32)

    moved_corners, status = move_corners(image_0, image_1, corners_0, level)

    good_corners[status == 0] = 0

    moved_corners = moved_corners[status == 1]
    corners_1 = corners_1[status == 1]

    dists = np.linalg.norm(corners_1 - moved_corners, axis=1)
    
    dist_threshold = CONSIST_DIST_THRESHOLD * 2 ** level

    good_corners[status == 1] = (dists < dist_threshold)

    return good_corners


def bad_ids_to_mask(ids: np.ndarray, bad_ids: np.ndarray) -> np.ndarray:
    bad_ids = np.unique(bad_ids)
    bad_id_poses = np.searchsorted(ids, bad_ids)
    good_ids_mask = np.ones(len(ids))
    good_ids_mask[bad_id_poses] = 0
    return good_ids_mask


def track_forward(consist_frames: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                  image: np.ndarray, level: int) -> Tuple[np.ndarray, np.ndarray]:
    last_image, last_corners, last_ids = consist_frames[-1]
    last_moved_corners, last_status = move_corners(last_image, image, last_corners, level)

    last_moved_corners = last_moved_corners[last_status == 1]
    last_ids = last_ids[last_status == 1]

    bad_ids = np.zeros(0, dtype=np.int32)
    for image_0, corners, ids in consist_frames[:-1]:
        _, (indices_0, indices_1) = snp.intersect(ids, last_ids, indices=True)
        ids = ids[indices_0]
        corners = corners[indices_0]
        l_last_moved_corners = last_moved_corners[indices_1]

        good_corners = check_consist(image_0, image, corners, l_last_moved_corners, level)
        bad_ids = np.concatenate((bad_ids, ids[good_corners == 0]))

    good_ids_mask = bad_ids_to_mask(last_ids, bad_ids)
    last_ids = last_ids[good_ids_mask == 1]
    last_moved_corners = last_moved_corners[good_ids_mask == 1]

    return last_moved_corners, last_ids


def check_track_back(consist_frames: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                     cur_frame: Tuple[np.ndarray, np.ndarray, np.ndarray],
                     level: int) \
    -> np.ndarray:
    cur_image, cur_corners, cur_ids = cur_frame

    bad_ids = np.zeros(0, dtype=np.int32)
    for image_0, corners, ids in consist_frames:
        _, (indices_0, indices_1) = snp.intersect(ids, cur_ids, indices=True)
        corners = corners[indices_0]
        l_cur_ids = cur_ids[indices_1]
        l_cur_corners = cur_corners[indices_1]

        good_corners = check_consist(cur_image, image_0, l_cur_corners, corners, level)
        bad_ids = np.concatenate((bad_ids, l_cur_ids[good_corners == 0]))

    return bad_ids_to_mask(cur_ids, bad_ids)


def check_distance(corners: np.ndarray, new_corners: np.ndarray) -> np.ndarray:
    if len(corners) == 0:
        return np.ones(len(new_corners), dtype=np.int32)

    dists = cdist(corners, new_corners)

    dists = dists.min(axis=0)
    good_new = dists > POINTS_MIN_DIST

    return good_new


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    frame_sequence = [image for image in frame_sequence]

    corner_storage = MultyLevelCornerStorage(
        levels=MAX_LEVEL + 1,
        base_radius=FEATURE_PARAMS['blockSize']
    )

    image_0 = frame_sequence[0]
    for level in range(MIN_LEVEL, MAX_LEVEL + 1):
        new_corners = find_new_corners(image_0, level)
        corner_storage.add(new_corners, frame=0, level=level)
    builder.set_corners_at_frame(0, corner_storage.frame_corners(0))

    for frame, image in enumerate(frame_sequence[1:], 1):
        for level in range(MIN_LEVEL, MAX_LEVEL + 1):
            consist_frame_nums = list(range(max(0, frame - CONSIST_LEN), frame))
            consist_frames = [(frame_sequence[i], *corner_storage.corners(i, level))
                                for i in consist_frame_nums]

            tracked_corners, tracked_corner_ids = track_forward(consist_frames, image, level)

            if DO_TRACK_BACK:
                good_tracked = check_track_back(
                    consist_frames,
                    (image, tracked_corners, tracked_corner_ids),
                    level
                )
                tracked_corners = tracked_corners[good_tracked == 1]
                tracked_corner_ids = tracked_corner_ids[good_tracked == 1]

            corner_storage.add(tracked_corners, frame=frame, level=level, ids=tracked_corner_ids)

            new_corners = find_new_corners(image, level)

            good_new = check_distance(tracked_corners, new_corners)
            new_corners = new_corners[good_new == 1]
            corner_storage.add(new_corners, frame=frame, level=level)

        builder.set_corners_at_frame(frame, corner_storage.frame_corners(frame))


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)

    corner_storage = builder.build_corner_storage()
    corner_storage = without_short_tracks(corner_storage, MIN_TRACK_LEN)

    return corner_storage


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
