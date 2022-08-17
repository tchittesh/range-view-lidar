import math
from pathlib import Path

import mmcv
import numpy as np


def create_lidar_imgs(data_path,
                      info_path,
                      save_path=None,
                      num_features=4):
    kitti_infos = mmcv.load(info_path)

    min_elevation = 0  # -0.27 rad, -16 deg over train set
    max_elevation = 0  # 0.13 rad, 8 deg over train set
    min_azimuth = 0  # -0.72, -42 deg over train set
    max_azimuth = 0  # 0.71, 41 deg over train set
    for info in mmcv.track_iter_progress(kitti_infos):
        v_path = info['point_cloud']['velodyne_path']
        v_reduced_path = v_path.replace('velodyne', 'velodyne_reduced')
        v_reduced_path = Path(data_path) / v_reduced_path
        reduced_points = np.fromfile(
            str(v_reduced_path), dtype=np.float32,
            count=-1).reshape([-1, num_features])

        reduced_points_x = reduced_points[:, 0]
        reduced_points_y = reduced_points[:, 1]
        reduced_points_z = reduced_points[:, 2]
        flat_distance = (reduced_points_x ** 2 + reduced_points_y ** 2) ** 0.5
        range = (flat_distance ** 2 + reduced_points_z ** 2) ** 0.5
        height = reduced_points_z
        azimuth = np.arctan2(reduced_points_y, reduced_points_x)
        elevation = np.arctan2(reduced_points_z, flat_distance)
        intensity = reduced_points[:, 3]
        occupancy = np.ones_like(range)

        min_azimuth = min(min_azimuth, azimuth.min())
        max_azimuth = max(max_azimuth, azimuth.max())
        min_elevation = min(min_elevation, elevation.min())
        max_elevation = max(max_elevation, elevation.max())

        lidar_img = np.zeros((64, 512, 5))
        azimuth_bin = (
            (-azimuth * 180 / math.pi + 45) * 512 / 90
        ).clip(0, 511).astype(np.int16)
        elevation_bin = (
            (-elevation * 180 / math.pi + 8) * 64 / 24
        ).clip(0, 63).astype(np.int16)
        data = np.stack((range, height, azimuth, intensity, occupancy), axis=1)
        lidar_img[elevation_bin, azimuth_bin] = data

        if save_path is None:
            save_dir = v_reduced_path.parent.parent / \
                (v_reduced_path.parent.stem + '_image')
            if not save_dir.exists():
                save_dir.mkdir()
            save_filename = save_dir / v_reduced_path.name
        else:
            save_filename = str(Path(save_path) / v_reduced_path.name)
        with open(save_filename, 'w') as f:
            lidar_img.tofile(f)

        # print(save_filename)

    # print(min_azimuth)
    # print(max_azimuth)
    # print(min_elevation)
    # print(max_elevation)


if __name__ == '__main__':
    info_paths = [
        '/project_data/ramanan/cthavama/kitti/kitti_infos_train.pkl',
        '/project_data/ramanan/cthavama/kitti/kitti_infos_val.pkl',
        '/project_data/ramanan/cthavama/kitti/kitti_infos_test.pkl',
    ]
    for info_path in info_paths:
        create_lidar_imgs('/project_data/ramanan/cthavama/kitti', info_path)
