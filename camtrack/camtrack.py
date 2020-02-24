#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _corners import FrameCorners
from _camtrack import (
    PointCloudBuilder,
    TriangulationParameters,
    create_cli,
    calc_point_cloud_colors,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    pose_to_view_mat3x4,
    calc_inlier_indices,
    build_correspondences,
    triangulate_correspondences,
    rodrigues_and_translation_to_view_mat3x4
)

TRIANGULATION_PARAMETERS = TriangulationParameters(
    max_reprojection_error=10,
    min_triangulation_angle_deg=0,
    min_depth=0
)
PNP_REPROJECTION_ERROR = 10
RANSAC_ITERATIONS = 100


def find_init_points(intrinsic_mat: np.ndarray,
                     corner_storage: np.ndarray,
                     known_view_1: Tuple[int, Pose],
                     known_view_2: Tuple[int, Pose]) \
        -> Tuple[np.ndarray, np.ndarray]:
    frame_num_1, pose_1 = known_view_1
    frame_num_2, pose_2 = known_view_2

    view_1 = pose_to_view_mat3x4(pose_1)
    view_2 = pose_to_view_mat3x4(pose_2)

    corners_1 = corner_storage[frame_num_1]
    corners_2 = corner_storage[frame_num_2]

    correspondences = build_correspondences(corners_1, corners_2)
    points_3d, point_ids, _ = triangulate_correspondences(correspondences,
                                                          view_1, view_2,
                                                          intrinsic_mat,
                                                          TRIANGULATION_PARAMETERS)

    return points_3d, point_ids


def get_detected_point_ids(points_3d: np.ndarray) -> np.ndarray:
    point_ids = filter(lambda i: points_3d[i, 0] is not None, range(len(points_3d)))
    point_ids = np.array(list(point_ids), dtype=int)
    return point_ids


def find_view(points_3d: np.ndarray,
              corners: FrameCorners,
              intrinsic_mat: np.ndarray) -> Tuple[np.ndarray, int]:
    points_3d = points_3d[corners.ids.reshape(-1)]
    
    points_2d = corners.points
    #points_2d = cv2.undistortPoints(
    #    points_2d.reshape(-1, 1, 2),
    #    intrinsic_mat,
    #    np.array([])
    #).reshape(-1, 2)

    detected_point_ids = get_detected_point_ids(points_3d)
    points_3d = points_3d[detected_point_ids].astype(np.float32)
    points_2d = points_2d[detected_point_ids].astype(np.float32)

    _, rvev, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=points_3d,
        imagePoints=points_2d,
        cameraMatrix=intrinsic_mat,
        distCoeffs=None,
        iterationsCount=RANSAC_ITERATIONS,
        reprojectionError=PNP_REPROJECTION_ERROR
    )

    view = rodrigues_and_translation_to_view_mat3x4(rvev, tvec)

    return view, len(inliers)


def find_new_points(view_mats: np.ndarray,
                    detected_point_ids: np.ndarray,
                    intrinsic_mat: List[np.ndarray],
                    corner_storage: np.ndarray,
                    cur_frame_num: int) \
        -> Tuple[np.ndarray, np.ndarray]:
    view_cur = view_mats[cur_frame_num].astype(np.float32)
    corners_cur = corner_storage[cur_frame_num]

    new_points_3d, new_point_ids = np.zeros((0, 3)), np.array([], dtype=int)
    for frame_i in range(cur_frame_num):
        view_i = view_mats[frame_i].astype(np.float32)
        corners_i = corner_storage[frame_i]

        correspondences = build_correspondences(corners_i, corners_cur,
                                                ids_to_remove=detected_point_ids)

        if len(correspondences[0]) == 0:
            continue

        points_3d, point_ids, _ = triangulate_correspondences(correspondences,
                                                              view_i, view_cur,
                                                              intrinsic_mat,
                                                              TRIANGULATION_PARAMETERS)

        new_points_3d = np.concatenate((new_points_3d, points_3d), axis=0)
        new_point_ids = np.concatenate((new_point_ids, point_ids), axis=0)
        detected_point_ids = np.concatenate((detected_point_ids, point_ids), axis=0)

    return new_points_3d, new_point_ids


def track(intrinsic_mat: np.ndarray,
          corner_storage: CornerStorage,
          known_view_1: Tuple[int, Pose],
          known_view_2: Tuple[int, Pose]) \
        -> Tuple[np.ndarray, np.ndarray]:
    assert known_view_1[0] == 0

    n_frames = len(corner_storage)
    view_mats = np.full((n_frames, 3, 4), None)

    n_points = corner_storage.max_corner_id() + 1
    points_3d = np.full((n_points, 3), None)

    new_points_3d, new_point_ids = find_init_points(intrinsic_mat, corner_storage,
                                                    known_view_1, known_view_2)
    view_mats[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
    points_3d[new_point_ids] = new_points_3d

    print(f'1 of {n_frames}   {len(new_point_ids)} triangulated   {len(new_point_ids)} in cloud')

    for frame_num in range(1, n_frames):
        corners = corner_storage[frame_num]
        view, inliers_cnt = find_view(points_3d, corners, intrinsic_mat)
        view_mats[frame_num] = view

        detected_point_ids = get_detected_point_ids(points_3d)
        new_points_3d, new_point_ids = find_new_points(view_mats,
                                                       detected_point_ids,
                                                       intrinsic_mat,
                                                       corner_storage,
                                                       frame_num)
        points_3d[new_point_ids] = new_points_3d

        print(f'{frame_num + 1} of {n_frames}   {inliers_cnt} inliers   '
              f'{len(new_point_ids)} triangulated   '
              f'{len(detected_point_ids) + len(new_point_ids)} in cloud')

    return view_mats, points_3d


def filter_tracked(view_mats: np.ndarray, points_3d: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    for i in range(0, len(view_mats), 1):
        if view_mats[i, 0, 0] is None:
            view_mats[i] = view_mats[i - 1].copy()

    view_mats = view_mats.astype(np.float32)

    point_ids = get_detected_point_ids(points_3d)
    points_3d = points_3d[point_ids].astype(np.float32)

    return view_mats, point_ids, points_3d

def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    view_mats, points_3d = track(intrinsic_mat, corner_storage,
                                 known_view_1, known_view_2)
    view_mats, point_ids, points_3d = filter_tracked(view_mats, points_3d)

    point_cloud_builder = PointCloudBuilder(point_ids, points_3d)

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
