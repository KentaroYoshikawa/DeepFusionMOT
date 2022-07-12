import argparse
import os

import numpy as np
import torch
from torchvision.ops.boxes import box_area
from munkres import Munkres

from tracking.track_3d import Track_3D
from tracking.kalman_fileter_3d import KalmanBoxTracker
from detection.detection import Detection_3D_only


parser = argparse.ArgumentParser()
parser.add_argument("--video_id", type=str, default="0000")
parser.add_argument("--data_dir", type=str, default="./datasets/kitti/train/3D_pointrcnn/5fps/")
parser.add_argument("--image_dir", type=str, default="./datasets/kitti/train/image_02_train/")
parser.add_argument("--iou_thresh", type=float, default=0.5)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def _load_3d(video_id, data_dir, image_dir):
    image_filenames = sorted([os.path.join(image_dir, video_id, x) for x in os.listdir(os.path.join(image_dir, video_id)) if is_image_file(x)])
    seq_dets_3D = np.loadtxt(os.path.join(data_dir, f"{video_id}.txt"), delimiter=' ')
    return image_filenames, seq_dets_3D


def box_iou(boxes1, boxes2):
    boxes1 = torch.tensor(boxes1)
    boxes2 = torch.tensor(boxes2)

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou.numpy()


def main(args):
    max_age = 25
    min_hits = 3
    image_filenames, seq_dets_3D = _load_3d(args.video_id, args.data_dir, args.image_dir)
    last_dets_3D_camera = None
    last_dets_3Dto2D_image = None
    trackers = []
    track_id_3d = 0
    h = Munkres()
    for frame, _ in enumerate(image_filenames):
        dets_3D_camera = seq_dets_3D[seq_dets_3D[:, 0] == frame, 7:14]  # 3D bounding box(h,w,l,x,y,z,theta)
        # ori_array = seq_dets_3D[seq_dets_3D[:, 0] == frame, -1].reshape((-1, 1))
        # other_array = seq_dets_3D[seq_dets_3D[:, 0] == frame, 1:7]
        # additional_info = np.concatenate((ori_array, other_array), axis=1)
        dets_3Dto2D_image = seq_dets_3D[seq_dets_3D[:, 0] == frame, 2:6]
        if frame % 2 == 1:
            continue
        if last_dets_3D_camera is None and last_dets_3Dto2D_image is None:
            for det_3D in dets_3D_camera:
                detection_3d = Detection_3D_only(det_3D, additional_info=None)
                kf_3d = KalmanBoxTracker(detection_3d.bbox)
                pose = np.concatenate(kf_3d.kf.x[:7], axis=0)
                trackers.append(
                    Track_3D(
                        pose=pose,
                        kf_3d=kf_3d,
                        track_id_3d=track_id_3d,
                        n_init=min_hits,
                        max_age=max_age,
                        additional_info=None,
                    )
                )
                track_id_3d += 2
            last_dets_3D_camera = dets_3D_camera
            last_dets_3Dto2D_image = dets_3Dto2D_image
            continue

        ious = box_iou(last_dets_3Dto2D_image, dets_3Dto2D_image)
        if ious.shape[0] > ious.shape[1]:
            matching = h.compute(ious.T)
            matching = [(j, i) for i, j in matching]
        else:
            matching = h.compute(ious)

        new_trackers = []
        for matched in matching:
            last_idx, idx = matched
            if ious[last_idx, idx] <= args.iou_thresh:
                # delete unused trackier
                # create new tracker
                detection_3d = Detection_3D_only(dets_3D_camera[idx], additional_info=None)
                kf_3d = KalmanBoxTracker(detection_3d.bbox)
                pose = np.concatenate(kf_3d.kf.x[:7], axis=0)
                new_trackers.append(
                    Track_3D(
                        pose=pose,
                        kf_3d=kf_3d,
                        track_id_3d=track_id_3d,
                        n_init=min_hits,
                        max_age=max_age,
                        additional_info=None
                    )
                )
                track_id_3d += 2
                continue
            detection_3d = Detection_3D_only(dets_3D_camera[idx],  additional_info=None)
            trackers[last_idx].update_3d(detection_3d)
            new_trackers.append(trackers[last_idx])

        last_dets_3D_camera = dets_3D_camera
        last_dets_3Dto2D_image = dets_3Dto2D_image


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
