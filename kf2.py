import argparse
import copy
import os
import os.path as op

import numpy as np
import torch
from munkres import Munkres
from torchvision.ops.boxes import box_area

from detection.detection import Detection_2D, Detection_3D_only
from tracking.kalman_fileter_3d import KalmanBoxTracker
from tracking.kalman_filter_2d import KalmanFilter
from tracking.track_2d import Track_2D
from tracking.track_3d import Track_3D

parser = argparse.ArgumentParser()
parser.add_argument("--video_id", type=str, default="0000")
parser.add_argument("--data_dir", type=str, default="./datasets/kitti/train/3D_pointrcnn/5fps")
parser.add_argument("--image_dir", type=str, default="./datasets/kitti/train/image_02_train")
parser.add_argument("--iou_thresh", type=float, default=0.5)


def is_image_file(filename):
    return any(
        filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
    )


def _load_3d(video_id, data_dir, image_dir):
    image_filenames = sorted(
        [
            os.path.join(image_dir, video_id, x)
            for x in os.listdir(os.path.join(image_dir, video_id))
            if is_image_file(x)
        ]
    )
    seq_dets_3D = np.loadtxt(os.path.join(data_dir, f"{video_id}.txt"), delimiter=" ")
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


def write_to_file(filename, frame_id, cls_id, bbox2d, score, bbox3d, alpha):
    """
    ** write file **
    sample_id, class_id,
    bbox[0], bbox[1], bbox[2], bbox[3], score,
    bbox3d[0], bbox3d[1], bbox3d[2], bbox3d[3], bbox3d[4], bbox3d[5], bbox3d[6], alpha
    """
    lines = []
    for i in range(len(cls_id)):
        bbox2d_i = ",".join([f"{box:.4f}" for box in bbox2d[i]])
        bbox3d_i = ",".join([f"{box:.4f}" for box in bbox3d[i]])
        line = f"{frame_id},{int(cls_id[i])},{bbox2d_i},{score[i]:.4f},{bbox3d_i},{alpha[i]:.4f}"
        print(line)
        lines.append(line + "\n")
    with open(filename, "a") as fp:
        fp.writelines(lines)


def main(args):
    max_age = 25
    min_hits = 3
    output_dir = op.join(op.dirname(args.data_dir), op.basename(args.data_dir) + "_interpolated")
    if not op.exists(output_dir):
        os.mkdir(output_dir)
    output_file = op.join(output_dir, args.video_id + ".txt")
    if op.exists(output_file):
        os.remove(output_file)

    image_filenames, seq_dets_3D = _load_3d(args.video_id, args.data_dir, args.image_dir)
    last_dets_3D_camera = None
    last_dets_3Dto2D_image = None
    last_dets_3D_speed = None
    last_dets_3Dto2D_speed = None
    trackers_3d = []
    trackers_2d = []
    cls_ids = []
    track_id_3d = 0
    track_id_2d = 0
    kf_2d = KalmanFilter()
    h = Munkres()
    delta_t = 0.2  # 5fps

    for frame, _ in enumerate(image_filenames):
        cls_id = seq_dets_3D[seq_dets_3D[:, 0] == frame, 1]
        score = seq_dets_3D[seq_dets_3D[:, 0] == frame, 6]
        alpha = seq_dets_3D[seq_dets_3D[:, 0] == frame, -1]
        dets_3D_camera = seq_dets_3D[seq_dets_3D[:, 0] == frame, 7:14]  # 3D bounding box(h,w,l,x,y,z,theta)
        # ori_array = seq_dets_3D[seq_dets_3D[:, 0] == frame, -1].reshape((-1, 1))
        # other_array = seq_dets_3D[seq_dets_3D[:, 0] == frame, 1:7]
        # additional_info = np.concatenate((ori_array, other_array), axis=1)
        dets_3Dto2D_image = seq_dets_3D[seq_dets_3D[:, 0] == frame, 2:6]
        if frame % 2 == 1:
            for trk_3d, trk_2d, cls_id in zip(trackers_3d, trackers_2d, cls_ids):
                trk_3d.predict_3d(trk_3d.kf_3d)
                bbox3d = trk_3d.pose
                beta = np.arctan2(bbox3d[6], bbox3d[3])
                alpha = -np.sign(beta) * np.pi / 2 + beta + bbox3d[6]

                trk_2d.predict_2d(kf_2d)
                bbox2d = trk_2d.x1y1x2y2()
                write_to_file(output_file, frame, [cls_id], [bbox2d], [1.0], [bbox3d], [alpha])
            continue

        # first iteration
        if last_dets_3D_camera is None and last_dets_3Dto2D_image is None:
            write_to_file(output_file, frame, cls_id, dets_3Dto2D_image, score, dets_3D_camera, alpha)
            for det_3D in dets_3D_camera:
                detection_3d = Detection_3D_only(det_3D, additional_info=None)
                kf_3d = KalmanBoxTracker(detection_3d.bbox)
                pose = np.concatenate(kf_3d.kf.x[:7], axis=0)
                trackers_3d.append(
                    Track_3D(
                        pose=pose,
                        kf_3d=kf_3d,
                        track_id_3d=track_id_3d,
                        n_init=min_hits,
                        max_age=max_age,
                        additional_info=None,
                    )
                )
                trackers_3d[-1].update_3d(detection_3d)
                track_id_3d += 2
            for det_2D in dets_3Dto2D_image:
                detection_2d = Detection_2D(det_2D)
                mean, covariance = kf_2d.initiate(detection_2d.to_xyah())
                trackers_2d.append(
                    Track_2D(
                        mean,
                        covariance,
                        track_id_2d,
                        min_hits,
                        max_age,
                    )
                )
                trackers_2d[-1].update_2d(kf_2d, detection_2d)
                track_id_2d += 2
            cls_ids = cls_id
            last_dets_3D_camera = dets_3D_camera
            last_dets_3Dto2D_image = dets_3Dto2D_image
            last_dets_3D_speed = [np.zeros(7) for _ in dets_3D_camera]
            last_dets_3Dto2D_speed = [np.zeros(4) for _ in dets_3Dto2D_image]
            continue

        write_to_file(output_file, frame, cls_id, dets_3Dto2D_image, score, dets_3D_camera, alpha)

        ious = box_iou(last_dets_3Dto2D_image, dets_3Dto2D_image)
        _ious = copy.deepcopy(ious)
        ious = 1.0 - ious
        if ious.shape[1] == 0:
            continue
        if ious.shape[0] > ious.shape[1]:
            matching = h.compute(ious.T)
            matching = [(j, i) for i, j in matching]
        else:
            matching = h.compute(ious)

        ious = _ious
        new_trackers_3d, new_trackers_2d, new_cls_ids = [], [], []
        matched_dets_indexes = []
        dets_3D_speed = [np.zeros(7) for _ in dets_3D_camera]
        dets_3Dto2D_speed = [np.zeros(4) for _ in dets_3Dto2D_image]
        for matched in matching:
            last_idx, idx = matched
            matched_dets_indexes.append(idx)

            v2_3D = last_dets_3D_speed[last_idx]
            v2_3Dto2D = last_dets_3Dto2D_speed[last_idx]
            v4_3D = (dets_3D_camera[idx] - last_dets_3D_camera[last_idx]) / delta_t
            v4_3Dto2D = (dets_3Dto2D_image[idx] - last_dets_3Dto2D_image[last_idx]) / delta_t
            a_3D = (v4_3D - v2_3D) / delta_t
            a_3Dto2D = (v4_3Dto2D - v2_3Dto2D) / delta_t
            _v2_3D = (
                dets_3D_camera[idx] - last_dets_3D_camera[last_idx] - a_3D * (delta_t) ** 2 / 2
            ) / delta_t
            _v2_3Dto2D = (
                dets_3Dto2D_image[idx] - last_dets_3Dto2D_image[last_idx] - a_3Dto2D * (delta_t) ** 2 / 2
            ) / delta_t
            x3_3D = last_dets_3D_camera[last_idx] + _v2_3D * (delta_t / 2) + a_3D * (delta_t / 2) ** 2 / 2
            x3_3Dto2D = (
                last_dets_3Dto2D_image[last_idx]
                + _v2_3Dto2D * (delta_t / 2)
                + a_3Dto2D * (delta_t / 2) ** 2 / 2
            )
            dets_3D_speed[idx] = v4_3D
            dets_3Dto2D_speed[idx] = v4_3Dto2D

            # update intermediate
            detection_3d = Detection_3D_only(x3_3D, additional_info=None)
            detection_2d = Detection_2D(x3_3Dto2D)
            trackers_3d[last_idx].update_3d(detection_3d)
            trackers_2d[last_idx].update_2d(kf_2d, detection_2d)
            # dummy prediction with intermediate
            trackers_3d[last_idx].predict_3d(trackers_3d[last_idx].kf_3d)
            trackers_2d[last_idx].predict_2d(kf_2d)

            # update current
            detection_3d = Detection_3D_only(dets_3D_camera[idx], additional_info=None)
            detection_2d = Detection_2D(dets_3Dto2D_image[idx])
            trackers_3d[last_idx].update_3d(detection_3d)
            trackers_2d[last_idx].update_2d(kf_2d, detection_2d)
            # dummy prediction with current
            new_trackers_3d.append(trackers_3d[last_idx])
            new_trackers_2d.append(trackers_2d[last_idx])
            new_cls_ids.append(cls_id[idx])

        last_dets_3D_speed = dets_3D_speed
        last_dets_3Dto2D_speed = dets_3Dto2D_speed

        # append new tracker 3d
        for i, det in enumerate(dets_3D_camera):
            if i in matched_dets_indexes:
                continue
            detection_3d = Detection_3D_only(det, additional_info=None)
            kf_3d = KalmanBoxTracker(detection_3d.bbox)
            pose = np.concatenate(kf_3d.kf.x[:7], axis=0)
            new_trackers_3d.append(
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

        # append new tracker 2d
        for i, det in enumerate(dets_3Dto2D_image):
            if i in matched_dets_indexes:
                continue
            detection_2d = Detection_2D(det)
            mean, covariance = kf_2d.initiate(detection_2d.to_xyah())
            new_trackers_2d.append(
                Track_2D(
                    mean,
                    covariance,
                    track_id_2d,
                    min_hits,
                    max_age,
                )
            )
            track_id_2d += 2

        for i, c in enumerate(cls_id):
            if i in matched_dets_indexes:
                continue
            new_cls_ids.append(c)

        trackers_3d = new_trackers_3d
        trackers_2d = new_trackers_2d
        cls_ids = new_cls_ids
        last_dets_3D_camera = dets_3D_camera
        last_dets_3Dto2D_image = dets_3Dto2D_image


if __name__ == "__main__":
    args = parser.parse_args()
    for i in range(0, 21):
        # DEBUG:
        # for i in range(0, 1):
        args.video_id = f"{i:04d}"
        main(args)
