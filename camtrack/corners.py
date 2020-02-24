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

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


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


def find_first_points(image, feature_prams):
    p0 = cv2.goodFeaturesToTrack(image, mask=None, **feature_prams)
    if p0 is not None:
        ids = np.arange(len(p0)).reshape(-1, 1)
    else:
        p0, ids = np.arange([]), np.arange([])
    id_cnt = len(ids)
    return p0, ids, id_cnt


def ft32_to_uint8(img):
    return np.round(img * 255).astype(np.uint8)


def track_points(image_0, image_1, p0, ids, lk_params, lk_error_threshold):
    if len(p0) > 0:
        p1, st, err = cv2.calcOpticalFlowPyrLK(ft32_to_uint8(image_0),
                                               ft32_to_uint8(image_1),
                                               p0, None, **lk_params)
    else:
        p1, st, err = None, None, None

    if st is not None:
        tracks = np.logical_and(st == 1, err < lk_error_threshold)
        p0 = p1[tracks].reshape(-1, 1, 2)
        ids = ids[tracks].reshape(-1, 1)
    else:
        p0, ids = np.array([]), np.array([])

    return p0, ids


def find_new_points(image, p0, ids, id_cnt, feature_prams, new_points_min_dist):
    p0_new = cv2.goodFeaturesToTrack(image, mask=None, **feature_prams)
    if p0_new is not None:
        mask = np.ones(image.shape, dtype=np.uint8)
        p0_int = p0.astype(np.uint32)
        p0_int = np.maximum(p0_int, 0)
        p0_int[:, :, 0] = np.minimum(p0_int[:, :, 0], mask.shape[1] - 1)
        p0_int[:, :, 1] = np.minimum(p0_int[:, :, 1], mask.shape[0] - 1)
        mask.T[p0_int] = 0
        kernel = np.ones((new_points_min_dist, new_points_min_dist), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

        p0_new_int = p0_new.reshape(-1, 2).astype(np.uint32)
        p0_new = p0_new[mask.T[p0_new_int[:, 0], p0_new_int[:, 1]] == 1]

        ids_new = np.arange(len(p0_new)).reshape(-1, 1) + id_cnt
        id_cnt += len(ids_new)
        p0 = np.concatenate((p0, p0_new), axis=0)
        ids = np.concatenate((ids, ids_new), axis=0)

    return p0, ids, id_cnt


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    FEATURE_PARAMS = dict(maxCorners = 10000,
                          qualityLevel = 0.0001,
                          minDistance = 5,
                          blockSize = 51,
                          gradientSize = 3,
                          useHarrisDetector = False,
                          k = 0.04)
    LK_PARAMS = dict(winSize = (15,15),
                     maxLevel = 2,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    LK_ERROR_THRESHOLD = 2.5
    NEW_POINTS_MIN_DIST = 5

    image_0 = frame_sequence[0]

    p0, ids, id_cnt = find_first_points(image_0, FEATURE_PARAMS)

    corners = FrameCorners(ids.reshape(-1), p0.reshape(-1, 2), np.ones(len(p0)) * 10)
    builder.set_corners_at_frame(0, corners)

    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        p0, ids = track_points(image_0, image_1, p0, ids, LK_PARAMS, LK_ERROR_THRESHOLD)

        p0, ids, id_cnt = find_new_points(image_1, p0, ids, id_cnt,
                                          FEATURE_PARAMS, NEW_POINTS_MIN_DIST)

        corners = FrameCorners(ids.reshape(-1), p0.reshape(-1, 2), np.ones(len(p0)) * 10)

        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1


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
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
