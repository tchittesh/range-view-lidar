# LaserNet FCOS3D Work in Progress

Based on [mmdetection3dv0.18.0](https://github.com/open-mmlab/mmdetection3d/tree/v0.18.0) and [Simplified LaserNet](https://github.com/kareemalsawah/Modified_LaserNet_Pytorch).

## Setup

Install mmdetection3d and set up the KITTI dataset as given in the docs in https://github.com/open-mmlab/mmdetection3d/tree/v0.18.0. Then run `python tools/create_lidar_img.py`, changing the `info_paths` in the main function as needed. This should result in the dataset configs in `configs/lasernet/lasernet_kitti.py` to properly load the lidar range view images.

## What Has and Hasn't Been Done

The goal is to write a range view LIDAR 3D detector with the backbone from LaserNet and the dense bounding box regression head from FCOS3D. I've set up the range view images as described above and also adapted the `mmdet3d/models/backbones/deep_layer_aggregation.py` backbone from [Simplified LaserNet](https://github.com/kareemalsawah/Modified_LaserNet_Pytorch). I've started writing the main detector class `mmdet3d/models/detectors/lasernet.py`, but there's much work left to be done. I suspect we'll also need a modified `mmdet3d/models/dense_heads/fcos_mono3d_head.py` class that takes into account the non-projective geometry of range-view images (e.g. change the `_get_bboxes_single` and the `pts2Dto3D` methods).
