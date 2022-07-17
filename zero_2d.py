import os
import numpy as np
import argparse
from os.path import join 


parser = argparse.ArgumentParser()
parser.add_argument("--video_id" , type=str, default="0000")
parser.add_argument("--data_dir" , type=str, default="./datasets/kitti/train/3D_pointrcnn/10fps_Car_val")


def _load_3d(video_id, data_dir):
    seq_dets_3D = np.loadtxt(os.path.join(data_dir, f"{video_id}.txt"), delimiter=',')
    return seq_dets_3D


def main(args):

    
    output_dir = os.path.join(os.path.dirname(args.data_dir), os.path.basename(args.data_dir)+"_zero2D")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_file = os.path.join(output_dir, args.video_id+"txt")
    if os.path.exists(output_file):
        os.remove(output_file)
    

    seq_dets_3D = _load_3d(args.video_id, args.data_dir)
    #print(seq_dets_3D[2][3])
     
    results = []
    for i in range(len(seq_dets_3D)):
        seq_dets_3D_ = f"{int(seq_dets_3D[i][0])},{int(seq_dets_3D[i][1])},{0},{0},{0},{0},{float(seq_dets_3D[i][6])},{float(seq_dets_3D[i][7])},{float(seq_dets_3D[i][8])},{float(seq_dets_3D[i][9])},{float(seq_dets_3D[i][10])},{float(seq_dets_3D[i][11])},{float(seq_dets_3D[i][12])},{float(seq_dets_3D[i][13])},{float(seq_dets_3D[i][14])}"
        #print(seq_dets_3D_)
        results.append(seq_dets_3D_+"\n")
        #print(result)

    #print(results)

    with open (output_file , 'a') as f:
        f.writelines(results)
    


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)