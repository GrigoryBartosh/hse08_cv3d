#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import copy

import cv2
import numpy as np
import sortednp as snp
from sklearn.preprocessing import normalize
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _corners import FrameCorners, StorageImpl
from _camtrack import (
    Correspondences,
    PointCloudBuilder,
    TriangulationParameters,
    remove_correspondences_with_ids,
    create_cli,
    calc_point_cloud_colors,
    eye3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    pose_to_view_mat3x4,
    to_camera_center,
    to_homogeneous,
    calc_inliers_mask,
    triangulate_correspondences,
    rodrigues_and_translation_to_view_mat3x4
)

EPS = 1e-8

MAX_EPIPOL_LINE_DIST = 1
MAX_REPROJECTION_ERROR = 10
MIN_TRIANGULATION_ANGLE_DEG = 1
RANSAC_P = 0.999

TRIANGULATION_PARAMETERS = TriangulationParameters(
    max_reprojection_error=MAX_REPROJECTION_ERROR,
    min_triangulation_angle_deg=MIN_TRIANGULATION_ANGLE_DEG,
    min_depth=0
)


class PointStorage():

    def __init__(self,
                 corner_storage: Optional[CornerStorage] = None,
                 n_frames: Optional[int] = None,
                 n_points: Optional[int] = None):
        if corner_storage is not None:
            self._init_by_corner_storage(corner_storage)
        else:
            self._init_data(n_frames, n_points)
            self._init_events()

#    def _verify(self) -> None:
#        self.process_all_events()
#
#        assert self._n_points == len(self._points2d)
#        assert self._n_points == len(self._point_first_frames)
#        assert self._n_points == len(self._point_last_frames)
#        assert self._n_points == len(self._points3d)
#        assert self._n_points == len(self._point_sizes)
#
#        assert self._n_frames == len(self._frame_point_ids)
#
#        for point_id in range(self._n_points):
#            assert len(self._points2d[point_id]) == self._point_last_frames[point_id] - self._point_first_frames[point_id], \
#                   f'{point_id} {len(self._points2d[point_id])} {self._point_first_frames[point_id]} {self._point_last_frames[point_id]}'
#
#        for frame_id in range(self._n_frames):
#            for point_id in self._frame_point_ids[frame_id]:
#                assert self._point_first_frames[point_id] <= frame_id, \
#                       f'{frame_id} {point_id} {self._point_first_frames[point_id]} {self._point_last_frames[point_id]}'
#                assert frame_id < self._point_last_frames[point_id], \
#                       f'{frame_id} {point_id} {self._point_first_frames[point_id]} {self._point_last_frames[point_id]}'

    def _init_data(self, n_frames: int, n_points: int) -> None:
        self._n_frames = n_frames
        self._n_points = n_points

        self._point_cloud_size = 0

        self._points3d = np.full((self._n_points, 3), None)
        self._point_sizes = np.zeros((self._n_points,), dtype=np.int32)
        self._point_first_frames = np.full((self._n_points,), self._n_frames, dtype=np.int32)
        self._point_last_frames = np.full((self._n_points,), -1, dtype=np.int32)
        self._points2d = [np.zeros((0, 2), dtype=np.float64) for _ in range(self._n_points)]

        self._frame_point_ids = [np.zeros((0,), dtype=np.int32) for _ in range(self._n_frames)]

    def _init_events(self) -> None:
        self._events_remove_old_point_id = [np.zeros((0,), dtype=np.int32)
                                            for _ in range(self._n_frames)]

        self._no_remove_2d = True

        self._events_remove_point_id = [np.zeros((0,), dtype=np.int32)
                                        for _ in range(self._n_frames)]
        self._events_add_point_id = [np.zeros((0,), dtype=np.int32)
                                     for _ in range(self._n_frames)]
        self._cur_remove_point_ids = np.zeros((0,), dtype=np.int32)
        self._cur_add_point_ids = np.zeros((0,), dtype=np.int32)

        self._cur_frame = 0

    def _init_by_corner_storage(self, corner_storage: CornerStorage) -> None:
        all_point_ids = [point_ids.reshape(-1) for point_ids, _, _ in corner_storage]
        all_point_ids = np.concatenate(all_point_ids, axis=0)
        all_point_ids = np.unique(all_point_ids)
        dict_ids = dict((point_id, i) for i, point_id in enumerate(all_point_ids))

        self._init_data(
            n_frames=len(corner_storage),
            n_points=len(dict_ids)
        )
        self._init_events()

        for frame_id, (point_ids, points2d, _) in enumerate(corner_storage):
            point_ids = point_ids.reshape(-1)
            point_ids = [dict_ids[point_id] for point_id in point_ids]
            point_ids = np.array(point_ids, dtype=np.int32)

            for point_id, point2d in zip(point_ids, points2d):
                self._point_first_frames[point_id] = min(
                    self._point_first_frames[point_id], frame_id)

                self._point_last_frames[point_id] = max(
                    self._point_last_frames[point_id], frame_id + 1)

                self._points2d[point_id] = np.concatenate(
                    (self._points2d[point_id], np.expand_dims(point2d, axis=0)), axis=0)

            self._frame_point_ids[frame_id] = point_ids

    def _remove_indeces(self, ids: np.ndarray, indices: np.ndarray) -> np.ndarray:
        good_indices = np.ones((len(ids),), dtype=np.int32)
        good_indices[indices] = 0
        ids = ids[good_indices == 1]
        return ids

    def _remove_common(self, ids_1: np.ndarray, ids_2: np.ndarray) -> np.ndarray:
        _, (indices, _) = snp.intersect(ids_1, ids_2, indices=True)
        ids_1 = self._remove_indeces(ids_1, indices)
        return ids_1

    def _update_remove_add_ids(self,
                               remove_ids: np.ndarray,
                               add_ids: np.ndarray,
                               new_remove_ids: np.ndarray,
                               new_add_ids: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
        remove_ids = np.concatenate((remove_ids, new_remove_ids), axis=0)
        add_ids = np.concatenate((add_ids, new_add_ids), axis=0)

        remove_ids = np.unique(remove_ids)
        add_ids = np.unique(add_ids)

        _, (indices_1, indices_2) = snp.intersect(remove_ids, add_ids, indices=True)
        remove_ids = self._remove_indeces(remove_ids, indices_1)
        add_ids = self._remove_indeces(add_ids, indices_2)

        return remove_ids, add_ids

    def _process_cur_events(self) -> None:
        self._cur_remove_point_ids, self._cur_add_point_ids = self._update_remove_add_ids(
            self._cur_remove_point_ids,
            self._cur_add_point_ids,
            self._events_remove_point_id[self._cur_frame],
            self._events_add_point_id[self._cur_frame]
        )

        ids = self._frame_point_ids[self._cur_frame]
        ids = self._remove_common(ids, self._cur_remove_point_ids)

        ids = np.concatenate((ids, self._cur_add_point_ids), axis=0)
        self._frame_point_ids[self._cur_frame] = ids

        self._cur_frame += 1

    def _process_events_until(self, target_frame: int) -> None:
        while self._cur_frame < target_frame:
            self._process_cur_events()

    def _process_old_events(self) -> None:
        remove_ids = np.zeros((0,), dtype=np.int32)

        for frame in range(self._n_frames)[::-1]:
            new_remove_ids = self._events_remove_old_point_id[frame]
            remove_ids = np.concatenate((remove_ids, new_remove_ids), axis=0)
            remove_ids = np.unique(remove_ids)

            ids = self._frame_point_ids[frame]

            _, (indices, _) = snp.intersect(ids, remove_ids, indices=True)

            for point_id in ids[indices]:
                if self._point_first_frames[point_id] <= frame:
                    n = frame + 1 - self._point_first_frames[point_id]
                    self._point_first_frames[point_id] = frame + 1
                    self._points2d[point_id] = self._points2d[point_id][n:]

            ids = self._remove_indeces(ids, indices)
            self._frame_point_ids[frame] = ids

        self._no_remove_2d = True

    def process_all_events(self) -> None:
        self._process_events_until(self._n_frames)
        self._process_old_events()
        self._init_events()

    def get_prefix(self, prefix_size: int) -> 'PointStorage':
        self.process_all_events()

        other = PointStorage(n_frames=prefix_size, n_points=self._n_points)

        other._point_cloud_size = self._point_cloud_size

        other._points3d = self._points3d
        other._point_sizes = self._point_sizes
        other._point_first_frames = copy.deepcopy(self._point_first_frames)
        other._point_last_frames = copy.deepcopy(self._point_last_frames)
        other._points2d = copy.deepcopy(self._points2d)
        for i in range(self._n_points):
            point_first_frame = other._point_first_frames[i]
            point_last_frame = other._point_last_frames[i]
            points2d = other._points2d[i]

            if point_first_frame > prefix_size:
                points2d = np.zeros((0, 2), dtype=np.float64)
                point_first_frame = prefix_size
                point_last_frame = prefix_size

            if point_last_frame > prefix_size:
                n = point_last_frame - prefix_size
                points2d = points2d[:-n]
                point_last_frame = prefix_size

            other._point_first_frames[i] = point_first_frame
            other._point_last_frames[i] = point_last_frame
            other._points2d[i] = points2d

        other._frame_point_ids = self._frame_point_ids[:prefix_size]

        return copy.deepcopy(other)

    def set_prefix(self, other: 'PointStorage', prefix_size: int) -> None:
        self.process_all_events()
        other.process_all_events()
        other = copy.deepcopy(other)

        self._point_cloud_size = other._point_cloud_size

        self._points3d = other._points3d

        self._point_sizes = other._point_sizes

        self._point_first_frames = np.concatenate(
            (self._point_first_frames, other._point_first_frames[self._n_points:]), axis=0)
        self._point_last_frames = np.concatenate(
            (self._point_last_frames, other._point_last_frames[self._n_points:]), axis=0)
        self._points2d += other._points2d[self._n_points:]

        self._frame_point_ids[:prefix_size] = other._frame_point_ids

        n_new_points = 0
        remove_ids = []
        for i in range(self._n_points):
            if self._point_first_frames[i] >= prefix_size:
                continue

            if self._point_last_frames[i] <= prefix_size:
                self._point_first_frames[i] = other._point_first_frames[i]
                self._point_last_frames[i] = other._point_last_frames[i]
                self._points2d[i] = other._points2d[i]
                continue

            n = prefix_size - self._point_first_frames[i]

            if other._point_last_frames[i] == prefix_size:
                self._points2d[i] = np.concatenate(
                    (other._points2d[i], self._points2d[i][n:]), axis=0)
                self._point_first_frames[i] = other._point_first_frames[i]
            else:
                self._points3d = np.concatenate(
                    (self._points3d, np.full((1, 3), None)), axis=0)
                self._point_sizes = np.concatenate(
                    (self._point_sizes, np.array([self._point_sizes[i]], dtype=np.int32)), axis=0)
                self._point_first_frames = np.concatenate(
                    (self._point_first_frames,
                     np.array([prefix_size], dtype=np.int32)), axis=0)
                self._point_last_frames = np.concatenate(
                    (self._point_last_frames,
                     np.array([self._point_last_frames[i]], dtype=np.int32)), axis=0)
                self._points2d += [self._points2d[i][n:]]

                self._point_first_frames[i] = other._point_first_frames[i]
                self._point_last_frames[i] = other._point_last_frames[i]
                self._points2d[i] = other._points2d[i]

                j = other._n_points + n_new_points
                self._frame_point_ids[prefix_size] = np.concatenate(
                    (self._frame_point_ids[prefix_size], np.array([j], dtype=np.int32)), axis=0)

                n_new_points += 1

                remove_ids += [i]

        self._n_points = other._n_points + n_new_points

        remove_ids = np.array(remove_ids, dtype=np.int32)
        for frame in range(prefix_size, self._n_frames):
            ids = self._frame_point_ids[frame]
            ids = self._remove_common(ids, remove_ids)
            self._frame_point_ids[frame] = ids

    def reverse(self) -> None:
        self.process_all_events()

        tmp = self._point_first_frames
        self._point_first_frames = self._n_frames - self._point_last_frames
        self._point_last_frames = self._n_frames - tmp
        for i in range(len(self._points2d)):
            self._points2d[i] = self._points2d[i][::-1]

        self._frame_point_ids = self._frame_point_ids[::-1]

    def frame_points(self, frame_id: int, detected: bool) -> List[np.ndarray]:
        assert (self._cur_frame <= frame_id + 1) or self._no_remove_2d

        self._process_events_until(frame_id + 1)

        point_ids, points2d, points3d = [], [], []
        for point_id in self._frame_point_ids[frame_id]:
            if self._point_first_frames[point_id] == self._point_last_frames[point_id]:
                continue

            if detected == (self._points3d[point_id, 0] is None):
                continue

            i = frame_id - self._point_first_frames[point_id]

            point_ids += [np.array([point_id], dtype=np.int32)]
            points2d += [np.expand_dims(self._points2d[point_id][i], axis=0)]
            if detected:
                points3d += [np.expand_dims(self._points3d[point_id], axis=0)]

        if len(point_ids) == 0:
            point_ids = np.zeros((0,), dtype=np.int32)
            points2d = np.zeros((0, 2), dtype=np.float64)
        else:
            point_ids = np.concatenate(point_ids, axis=0)
            points2d = np.concatenate(points2d, axis=0)

        if detected:
            if len(points3d) == 0:
                points3d = np.zeros((0, 3), dtype=np.float64)
            else:
                points3d = np.concatenate(points3d, axis=0).astype(np.float64)

        if detected:
            return point_ids, points2d, points3d
        else:
            return point_ids, points2d

    def point_frames(self, point_id: int) -> Tuple[np.ndarray, np.ndarray]:
        first_frame = self._point_first_frames[point_id]
        last_frame = self._point_last_frames[point_id]

        frame_ids = np.arange(first_frame, last_frame)
        points2d = self._points2d[point_id]

        return frame_ids, points2d

    def set_3d(self, point_ids: np.ndarray, points3d: np.ndarray) -> None:
        self._points3d[point_ids] = points3d
        self._point_cloud_size += len(point_ids)

    def cut_tracks(self, point_ids: np.ndarray, frame_id: int) -> None:
        assert frame_id + 1 == self._cur_frame

        n_new_points = len(point_ids)

        new_points3d = np.full((n_new_points, 3), None)
        self._points3d = np.concatenate((self._points3d, new_points3d), axis=0)

        self._point_sizes = np.concatenate(
            (self._point_sizes, self._point_sizes[point_ids]), axis=0)

        new_point_first_frames = np.full((n_new_points,), frame_id, dtype=np.int32)
        self._point_first_frames = np.concatenate(
            (self._point_first_frames, new_point_first_frames), axis=0)

        new_point_last_frames = self._point_last_frames[point_ids]
        self._point_last_frames[point_ids] = frame_id
        self._point_last_frames = np.concatenate(
            (self._point_last_frames, new_point_last_frames), axis=0)

        new_point_ids = np.arange(self._n_points, self._n_points + n_new_points, dtype=np.int32)
        self._n_points = self._n_points + n_new_points

        for point_id, new_point_id in zip(point_ids, new_point_ids):
            i = frame_id - self._point_first_frames[point_id]
            self._points2d += [self._points2d[point_id][i:]]
            self._points2d[point_id] = self._points2d[point_id][:i]

            new_point_last_frame = self._point_last_frames[new_point_id]
            if new_point_last_frame < self._n_frames:
                self._events_remove_point_id[new_point_last_frame] = np.concatenate(
                    (self._events_remove_point_id[new_point_last_frame],
                     np.array([new_point_id], dtype=np.int32)), axis=0)

        frame_point_ids = self._frame_point_ids[frame_id]
        frame_point_ids = self._remove_common(frame_point_ids, point_ids)
        frame_point_ids = np.concatenate((frame_point_ids, new_point_ids), axis=0)
        self._frame_point_ids[frame_id] = frame_point_ids

        if self._cur_frame < self._n_frames:
            self._events_add_point_id[self._cur_frame] = np.concatenate(
                (self._events_add_point_id[self._cur_frame], new_point_ids), axis=0)
            self._events_remove_point_id[self._cur_frame] = np.concatenate(
                (self._events_remove_point_id[self._cur_frame], point_ids), axis=0)

    def remove_2d(self, point_id: int, frame_id: int) -> None:
        ids = self._events_remove_old_point_id[frame_id]
        ids = np.concatenate((ids, np.array([point_id], dtype=np.int32)), axis=0)
        self._events_remove_old_point_id[frame_id] = ids

        self._no_remove_2d = False

    def n_frames(self) -> int:
        return self._n_frames

    def n_points(self) -> int:
        return self._n_points

    def point_cloud_size(self) -> int:
        return self._point_cloud_size

    def point_cloud(self) -> Tuple[np.ndarray, np.ndarray]:
        mask = (self._points3d[:, 0] == None)
        ids = np.where(mask == False)[0]
        point_cloud = self._points3d[ids].astype(np.float64)
        return ids, point_cloud

    def corresponding_view_point(self, detected) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        camera_indices, point_indices, points2d = [], [], []
        point_output_id = 0
        for point_id in range(self._n_points):
            if detected == (self._points3d[point_id, 0] is None):
                continue

            ff = self._point_first_frames[point_id]
            lf = self._point_last_frames[point_id]
            camera_indices += [np.arange(ff, lf)]
            point_indices += [np.ones((lf - ff,), dtype=np.int32) * point_output_id]
            points2d += [self._points2d[point_id]]

            point_output_id += 1

        camera_indices = np.concatenate(camera_indices, axis=0)
        point_indices = np.concatenate(point_indices, axis=0)
        points2d = np.concatenate(points2d, axis=0)

        return camera_indices, point_indices, points2d

    def build_corner_storage(self) -> CornerStorage:
        self.process_all_events()

        frame_corners = []
        for frame_id in range(self._n_frames):
            ids = self._frame_point_ids[frame_id]
            sizes = self._point_sizes[ids]

            points2d = []
            for point_id in ids:
                i = frame_id - self._point_first_frames[point_id]
                points2d += [np.expand_dims(self._points2d[point_id][i], axis=0)]

            points2d = np.concatenate(points2d, axis=0)

            frame_corners += [FrameCorners(ids, points2d, sizes)]

        return StorageImpl(frame_corners)


def build_correspondences(ids_1: np.ndarray, ids_2: np.ndarray,
                          corners_1: np.ndarray, corners_2: np.ndarray) \
    -> Correspondences:
    _, (indices_1, indices_2) = snp.intersect(ids_1, ids_2, indices=True)
    corrs = Correspondences(
        ids_1[indices_1],
        corners_1[indices_1],
        corners_2[indices_2]
    )
    return corrs


def find_view_by_two_frames(ids_1: np.ndarray, ids_2: np.ndarray,
                            corners_1: np.ndarray, corners_2: np.ndarray,
                            intrinsic_mat: np.ndarray) \
    -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    correspondences = build_correspondences(ids_1, ids_2, corners_1, corners_2)

    if len(correspondences.ids) < 7:
        return None, None, 0

    mat, mat_mask = cv2.findEssentialMat(
        correspondences.points_1,
        correspondences.points_2,
        intrinsic_mat,
        method=cv2.RANSAC,
        prob=RANSAC_P,
        threshold=MAX_EPIPOL_LINE_DIST
    )

    mat_mask = mat_mask.flatten()
    correspondences = remove_correspondences_with_ids(
        correspondences, np.argwhere(mat_mask == 0).astype(np.int32))

    view_1 = eye3x4()

    best_view = None
    best_count = -1

    rotation_1, rotation_2, translation = cv2.decomposeEssentialMat(mat)
    rotation_1, rotation_2 = rotation_1.T, rotation_2.T
    for r in (rotation_1, rotation_2):
        for t in (translation, -translation):
            view_2 = pose_to_view_mat3x4(Pose(r, r @ t))

            _, point_ids, _ = triangulate_correspondences(
                correspondences,
                view_1, view_2,
                intrinsic_mat,
                TRIANGULATION_PARAMETERS
            )

            if best_count < len(point_ids):
                best_view = view_2
                best_count = len(point_ids)

    return view_1, best_view, best_count


def find_init_views(point_storage: PointStorage,
                    intrinsic_mat: np.ndarray) \
    -> Tuple[Tuple[int, np.ndarray], Tuple[int, np.ndarray]]:
    n_frames = point_storage.n_frames()

    MIN_DIST = 10
    MAX_DIST = 60
    frame_pairs = [(i, j) for i in range(n_frames - MIN_DIST)
                          for j in range(i + MIN_DIST, min(i + MAX_DIST + 1, n_frames))]
    frame_pairs = np.array(frame_pairs)
    frame_pair_ids = np.random.choice(
        len(frame_pairs),
        min(len(frame_pairs), 2 * n_frames),
        replace=False
    )
    frame_pairs = frame_pairs[frame_pair_ids]

    best_frame_num_1 = None
    best_frame_num_2 = None
    best_view_1 = None
    best_view_2 = None
    best_count = -1

    for frame_num_1, frame_num_2 in frame_pairs:
        ids_1, corners_1 = point_storage.frame_points(frame_num_1, detected=False)
        ids_2, corners_2 = point_storage.frame_points(frame_num_2, detected=False)

        view_1, view_2, count = find_view_by_two_frames(
            ids_1, ids_2,
            corners_1, corners_2,
            intrinsic_mat
        )

        if best_count < count:
            best_frame_num_1 = frame_num_1
            best_frame_num_2 = frame_num_2
            best_view_1 = view_1
            best_view_2 = view_2
            best_count = count

    return (best_frame_num_1, best_view_1), (best_frame_num_2, best_view_2)


def find_init_points(frame_num_1: int, frame_num_2: int,
                     view_1: np.ndarray, view_2: np.ndarray,
                     point_storage: PointStorage,
                     intrinsic_mat: np.ndarray) -> None:
    ids_1, corners_1 = point_storage.frame_points(frame_num_1, detected=False)
    ids_2, corners_2 = point_storage.frame_points(frame_num_2, detected=False)

    correspondences = build_correspondences(ids_1, ids_2, corners_1, corners_2)
    points3d, point_ids, _ = triangulate_correspondences(
        correspondences,
        view_1, view_2,
        intrinsic_mat,
        TRIANGULATION_PARAMETERS
    )

    point_storage.set_3d(point_ids, points3d)


def solve_pnp(points2d: np.ndarray,
              points3d: np.ndarray,
              intrinsic_mat: np.ndarray) -> np.ndarray:
    _, rvev, tvec = cv2.solvePnP(
        objectPoints=points3d.astype(np.float64),
        imagePoints=points2d,
        cameraMatrix=intrinsic_mat,
        distCoeffs=None
    )

    return rodrigues_and_translation_to_view_mat3x4(rvev, tvec)


def find_view_until_converged(points2d: np.ndarray,
                              points3d: np.ndarray,
                              intrinsic_mat: np.ndarray) \
    -> Tuple[np.ndarray, np.ndarray]:
    m = len(points2d)
    n = 6

    best_view = None
    best_inliers = None
    bs = 0

    inliers = np.zeros((m,), dtype=np.int32)
    s = 0

    old_sums = set([m])

    converged = False
    while not converged:
        if s < n:
            ids = np.random.choice(m, n, replace=False)
            inliers = np.zeros((m,), dtype=np.int32)
            inliers[ids] = 1

            old_sums = set([m])

        l_points2d = points2d[inliers == 1]
        l_points3d = points3d[inliers == 1]
        view = solve_pnp(l_points2d, l_points3d, intrinsic_mat)
        new_inliers = calc_inliers_mask(points3d, points2d,
                                        intrinsic_mat @ view,
                                        MAX_REPROJECTION_ERROR)

        ns = new_inliers.sum()

        inliers = new_inliers
        s = ns

        old_sums.add(ns)

        converged = (ns in old_sums)

        if best_view is None or bs < s:
            best_view = view
            best_inliers = inliers
            bs = s

    return best_view, best_inliers


def find_view_ransac(points2d: np.ndarray,
                     points3d: np.ndarray,
                     intrinsic_mat: np.ndarray) \
    -> Tuple[np.ndarray, np.ndarray]:
    m = len(points2d)
    n = 6
    k = 0

    best_view = None
    best_inliers = None
    bs = 0

    finish = False
    while not finish:
        view, inliers = find_view_until_converged(points2d, points3d, intrinsic_mat)

        s = inliers.sum()

        if best_view is None or bs < s:
            best_view = view
            best_inliers = inliers
            bs = s

        k += 1

        if bs == m:
            finish = True
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                w = 1 - bs / m
                denom = np.log(1 - (1 - w) ** n)
                if abs(denom) > EPS:
                    k_traget = np.log(1 - RANSAC_P) / denom
                else:
                    k_traget = k + 1

            finish = (k_traget <= k)

    return best_view, best_inliers


def find_view(frame: int,
              point_storage: PointStorage,
              intrinsic_mat: np.ndarray) \
    -> Tuple[np.ndarray, int]:
    ids, points2d, points3d = point_storage.frame_points(frame, detected=True)

    view, inliers = find_view_ransac(points2d, points3d, intrinsic_mat)

    point_storage.cut_tracks(ids[inliers == 0], frame)

    return view, inliers.sum()


def triangulate(T: np.ndarray,
                x: np.ndarray,
                intrinsic_mat: np.ndarray) -> np.ndarray:
    n = len(x)

    x = cv2.undistortPoints(
        x.reshape(-1, 1, 2),
        intrinsic_mat,
        np.array([])
    ).reshape(-1, 2)
    x = np.concatenate((x, np.ones((n, 1), dtype=np.float64)), axis=1)

    A = np.cross(np.expand_dims(x, axis=1), T.transpose(0, 2, 1)).transpose(0, 2, 1)
    A = A.reshape(-1, 4)
    A, b = A[:, :3], -A[:, 3]

    p = np.linalg.lstsq(A, b, rcond=None)[0]

    return p


def convert_point_to_views(point3d: np.ndarray, proj_mats: np.ndarray) -> np.ndarray:
    point3d = to_homogeneous(np.expand_dims(point3d, axis=0))[0]
    points3d = proj_mats @ point3d
    return points3d


def project_points_to_2d(points3d: np.ndarray) -> np.ndarray:
    points2d = points3d[:, :2] / points3d[:, 2:]
    return points2d


def compute_point_reprojection_errors(points2d_1: np.ndarray,
                                      points2d_2: np.ndarray) -> np.ndarray:
    points2d_diff = points2d_1 - points2d_2
    return np.linalg.norm(points2d_diff, axis=1)


def calc_point_inliers_mask(point3d: np.ndarray, points2d: np.ndarray,
                            proj_mats: np.ndarray, max_error: float) -> np.ndarray:
    converted_points3d = convert_point_to_views(point3d, proj_mats)
    projected_points2d = project_points_to_2d(converted_points3d)
    errors = compute_point_reprojection_errors(projected_points2d, points2d)
    mask = np.logical_and(errors <= max_error, converted_points3d[:, 2] > 0)
    return mask


def calc_point_triangulation_angle_mask(point3d: np.ndarray,
                                        view_mat_1: np.ndarray,
                                        view_mats: np.ndarray,
                                        min_angle_deg: float) -> np.ndarray:
    camera_center_1 = to_camera_center(view_mat_1)
    camera_centers = to_camera_center(view_mats)

    vec_1 = normalize(np.expand_dims(camera_center_1 - point3d, axis=0))
    vecs = normalize(camera_centers - point3d)

    vec_1 = vec_1.repeat(len(vecs), axis=0)

    coss_abs = np.abs(np.einsum('ij,ij->i', vec_1, vecs))
    angles_mask = coss_abs <= np.cos(np.deg2rad(min_angle_deg))
    return angles_mask


def triangulate_and_check(view_mats: np.ndarray,
                          points2d: np.ndarray,
                          intrinsic_mat: np.ndarray) \
    -> Tuple[Optional[np.ndarray], int]:
    n = len(view_mats)
    if n < 2:
        return None, n

    l, r = 1, n + 1
    while r - l > 1:
        m = (l + r) // 2

        l_view_mats = view_mats[-m:]
        l_points2d = points2d[-m:]

        point3d = triangulate(l_view_mats, l_points2d, intrinsic_mat)

        inliers = calc_point_inliers_mask(point3d, l_points2d,
                                          intrinsic_mat @ l_view_mats,
                                          MAX_REPROJECTION_ERROR)

        if inliers.sum() == len(inliers):
            l = m
        else:
            r = m

    view_mats = view_mats[-l:]
    points2d = points2d[-l:]
    point3d = triangulate(view_mats, points2d, intrinsic_mat)

    if l == 1:
        return None, l

    view_mats = view_mats[-l:]
    inliers = calc_point_triangulation_angle_mask(
        point3d, view_mats[-1], view_mats[:-1],
        MIN_TRIANGULATION_ANGLE_DEG
    )

    if inliers.sum() == 0:
        return None, l
    else:
        return point3d, l


def find_new_points(frame: int,
                    view_mats: np.ndarray,
                    point_storage: PointStorage,
                    intrinsic_mat: np.ndarray) -> int:
    ids, _ = point_storage.frame_points(frame, detected=False)

    triangulated_cnt = 0
    for i in ids:
        frame_ids, points2d = point_storage.point_frames(i)
        points2d = points2d[frame_ids <= frame]
        frame_ids = frame_ids[frame_ids <= frame]

        l_view_mats = view_mats[frame_ids]

        point3d, k = triangulate_and_check(l_view_mats, points2d, intrinsic_mat)

        point_storage.remove_2d(i, frame - k)

        if point3d is None:
            continue

        point_storage.set_3d(np.array([i], dtype=np.int32),
                             np.expand_dims(point3d, axis=0))

        triangulated_cnt += 1

    return triangulated_cnt


def batch_identity(batch_size: int, n: int) -> np.ndarray:
    I = np.zeros((batch_size, n, n), dtype=np.float64)
    i = np.arange(n)
    I[:, i, i] = 1
    return I


def s_1_2(r: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(r, axis=1)
    mask_pi = (np.abs(norms - np.pi) < EPS)

    mask_1_e = (np.abs(r[:, 0]) < EPS)
    mask_2_e = (np.abs(r[:, 1]) < EPS)

    mask_1_l = (r[:, 0] < -EPS)
    mask_2_l = (r[:, 1] < -EPS)
    mask_3_l = (r[:, 2] < -EPS)

    mask = np.logical_and(
        mask_pi,
        np.logical_or(
            np.logical_and(
                np.logical_and(
                    mask_1_e,
                    mask_2_e
                ),
                mask_3_l
            ),
            np.logical_or(
                np.logical_and(
                    mask_1_e,
                    mask_2_l
                ),
                mask_1_l
            )
        )
    )

    r = r.copy()
    r[mask] = -r[mask]

    return r


def rot_to_rodrigues_case_2(R: np.ndarray) -> np.ndarray:
    n = len(R)

    I = batch_identity(n, 3)

    R_ = R + I

    norms = np.linalg.norm(R_, axis=1)
    i = np.argmax(norms, axis=1)

    v = R_[np.arange(n), :, i]

    u = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
    return s_1_2(u * np.pi)


def rot_to_rodrigues_case_3(ro: np.ndarray, s: np.ndarray, c: np.ndarray) -> np.ndarray:
    u = ro / s[:, np.newaxis]
    theta = np.arctan2(s, c)
    return u * theta[:, np.newaxis]


def rot_to_rodrigues(R: np.ndarray) -> np.ndarray:
    A = (R - R.transpose(0, 2, 1)) / 2
    ro = np.stack([A[:, 2, 1], A[:, 0, 2], A[:, 1, 0]], axis=1)

    s = np.linalg.norm(ro, axis=1)
    c = (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2

    mask_s_0 = (abs(s) < EPS)
    mask_c_1 = (abs(c - 1) < EPS)
    mask_c_m1 = (abs(c + 1) < EPS)

    mask_1 = np.logical_and(mask_s_0, mask_c_1)
    mask_2 = np.logical_and(mask_s_0, mask_c_m1)
    mask_3 = np.logical_not(np.logical_or(mask_1, mask_2))

    n = len(R)

    r = np.zeros((n, 3), dtype=np.float64)
    r[mask_2] = rot_to_rodrigues_case_2(R[mask_2])
    r[mask_3] = rot_to_rodrigues_case_3(ro[mask_3], s[mask_3], c[mask_3])

    return r


def antisymmetric(u: np.ndarray) -> np.ndarray:
    n = len(u)
    ux = np.zeros((n, 3, 3), dtype=np.float64)
    ux[:, 0, 1] = -u[:, 2]
    ux[:, 0, 2] = u[:, 1]
    ux[:, 1, 2] = -u[:, 0]
    ux[:, 1, 0] = u[:, 2]
    ux[:, 2, 0] = -u[:, 1]
    ux[:, 2, 1] = u[:, 0]
    return ux


def rodrigues_to_rot_case_2(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    n = len(r)

    u = r / theta[:, np.newaxis]

    I = batch_identity(n, 3)
    R = I * np.cos(theta)[:, np.newaxis, np.newaxis] + \
        (1 - np.cos(theta))[:, np.newaxis, np.newaxis] * (u[:, :, np.newaxis] @ u[:, np.newaxis, :]) + \
        antisymmetric(u) * np.sin(theta)[:, np.newaxis, np.newaxis]

    return R


def rodrigues_to_rot(r: np.ndarray) -> np.ndarray:
    n = len(r)

    theta = np.linalg.norm(r, axis=1)

    mask = (np.abs(theta) > EPS)

    R = batch_identity(n, 3)
    R[mask] = rodrigues_to_rot_case_2(r[mask], theta[mask])

    return R


def bundle_adjustment_fun(params: np.ndarray, 
                          n_cameras: int,
                          n_points: int,
                          camera_indices: np.ndarray,
                          point_indices: np.ndarray,
                          points2d: np.ndarray) -> np.ndarray:
    camera_params = params[:n_cameras * 6].reshape(n_cameras, 6)
    r = camera_params[:, :3]
    t = camera_params[:, 3:]
    R = rodrigues_to_rot(r)

    points3d = params[n_cameras * 6:].reshape(n_points, 3)

    R = R[camera_indices]
    t = t[camera_indices]
    points3d = points3d[point_indices]

    points3d = (R @ points3d[:, :, np.newaxis]).reshape(-1, 3)
    points3d = points3d + t

    projected_points2d = points3d[:, :2] / points3d[:, 2:]

    return (projected_points2d - points2d).reshape(-1)


def bundle_adjustment_sparsity(n_cameras: int,
                               n_points: int,
                               camera_indices: np.ndarray,
                               point_indices: np.ndarray) -> lil_matrix:
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

    return A


def bundle_adjustment(view_mats: np.ndarray,
                      points3d: np.ndarray,
                      camera_indices: np.ndarray,
                      point_indices: np.ndarray,
                      points2d: np.ndarray) \
    -> Tuple[np.ndarray, np.ndarray]:
    R = view_mats[:, :, :3]
    t = view_mats[:, :, 3]
    r = rot_to_rodrigues(R)
    camera_params = np.concatenate((r, t), axis=1)    

    x0 = np.concatenate((camera_params.reshape(-1), points3d.reshape(-1)))

    n_cameras = len(view_mats)
    n_points = len(points3d)

    A = bundle_adjustment_sparsity(
        n_cameras,
        n_points,
        camera_indices,
        point_indices
    )

    res = least_squares(
        bundle_adjustment_fun,
        x0,
        jac_sparsity=A,
        verbose=2,
        x_scale='jac',
        ftol=1e-4,
        method='trf',            
        args=(n_cameras, n_points, camera_indices, point_indices, points2d)
    )

    x = res.x

    camera_params = x[:n_cameras * 6].reshape(n_cameras, 6)
    r = camera_params[:, :3]
    t = camera_params[:, 3:]
    R = rodrigues_to_rot(r)
    view_mats = np.concatenate((R, t[:, :, np.newaxis]), axis=2)

    points3d = x[n_cameras * 6:].reshape(n_points, 3)

    return view_mats, points3d


def optimize(view_mats: np.ndarray,
             point_storage: PointStorage,
             intrinsic_mat: np.ndarray) -> np.ndarray:
    point_ids, points3d = point_storage.point_cloud()
    camera_indices, point_indices, points2d = point_storage.corresponding_view_point(detected=True)

    points2d = cv2.undistortPoints(
        points2d.reshape(-1, 1, 2),
        intrinsic_mat,
        np.array([])
    ).reshape(-1, 2)

    view_mats, points3d = bundle_adjustment(
        view_mats, points3d,
        camera_indices, point_indices, points2d
    )

    point_storage.set_3d(point_ids, points3d)

    return view_mats
    

def track(intrinsic_mat: np.ndarray,
          corner_storage: CornerStorage,
          known_view_1: Optional[Tuple[int, Pose]],
          known_view_2: Optional[Tuple[int, Pose]]) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, CornerStorage]:
    assert (known_view_1 is None) == (known_view_2 is None)

    n_frames = len(corner_storage)
    point_storage = PointStorage(corner_storage)
    view_mats = np.zeros((0, 3, 4), dtype=np.float64)

    if known_view_1 is None:
        (frame_num_1, view_1), (frame_num_2, view_2) = find_init_views(
            point_storage, intrinsic_mat)
    else:
        if known_view_1[0] > known_view_2[0]:
            known_view_1, known_view_2 = known_view_2, known_view_1

        frame_num_1, pose_1 = known_view_1
        frame_num_2, pose_2 = known_view_2

        view_1 = pose_to_view_mat3x4(pose_1)
        view_2 = pose_to_view_mat3x4(pose_2)

    print(f'Initial frames: {frame_num_1} and {frame_num_2}')

    find_init_points(frame_num_1, frame_num_2,
                     view_1, view_2,
                     point_storage,
                     intrinsic_mat)
    point_storage.process_all_events()
    view_mats = np.concatenate((view_mats, np.expand_dims(view_1, axis=0)))

    print(f'{1} of {n_frames}   {point_storage.point_cloud_size()} inliers   '
          f'{point_storage.point_cloud_size()} triangulated   '
          f'{point_storage.point_cloud_size()} in cloud')

    first_point_storage = point_storage.get_prefix(frame_num_1 + 1)
    first_point_storage.reverse()

    for frame in range(1, frame_num_1 + 1):
        view, inliers_cnt = find_view(frame, first_point_storage, intrinsic_mat)
        view_mats = np.concatenate((view_mats, np.expand_dims(view, axis=0)))

        triangulated_cnt = find_new_points(frame, view_mats, first_point_storage, intrinsic_mat)

        print(f'{frame + 1} of {n_frames}   {inliers_cnt} inliers   '
              f'{triangulated_cnt} triangulated   '
              f'{first_point_storage.point_cloud_size()} in cloud')

    first_point_storage.reverse()
    point_storage.set_prefix(first_point_storage, frame_num_1 + 1)
    view_mats = view_mats[::-1]

    for frame in range(frame_num_1 + 1, n_frames):
        view, inliers_cnt = find_view(frame, point_storage, intrinsic_mat)
        view_mats = np.concatenate((view_mats, np.expand_dims(view, axis=0)))

        triangulated_cnt = find_new_points(frame, view_mats, point_storage, intrinsic_mat)

        print(f'{frame + 1} of {n_frames}   {inliers_cnt} inliers   '
              f'{triangulated_cnt} triangulated   '
              f'{point_storage.point_cloud_size()} in cloud')

    point_storage.process_all_events()
    view_mats = optimize(view_mats, point_storage, intrinsic_mat)

    return (view_mats, *point_storage.point_cloud(), point_storage.build_corner_storage())


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    view_mats, point_ids, points3d, corner_storage = track(intrinsic_mat, corner_storage,
                                                            known_view_1, known_view_2)

    point_cloud_builder = PointCloudBuilder(point_ids, points3d)

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
