# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import random
import time
import cv2
import os

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from glob import glob

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        # default="../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
        default="../configs/e2e_mask_rcnn_R_50_FPN_wolf.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=224,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_file = os.path.join("..", cfg.OUTPUT_DIR, "last_checkpoint")
    # print("save_file: " ,save_file)

    try:
        with open(save_file, "r") as f:
            last_saved = f.read()
            last_saved = last_saved.strip()
            last_saved = os.path.join("..", last_saved)
    except IOError:
        # if file doesn't exist, maybe because it has just been
        # deleted by a separate process
        last_saved = None

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
        weight_loading=last_saved
    )

    train_images = glob("../../dataset/robot_images/*.jpg")
    test_images  = glob("../../dataset/tripod_images/*.jpg")

    start_time = time.time()
    rand_img   = random.choice(train_images)
    print('images: {}'.format(rand_img))
    img        = cv2.imread(rand_img)
        
    composite = coco_demo.run_on_opencv_image(img)
    print("Time: {:.2f} s / img".format(time.time() - start_time))
    cv2.imshow("COCO detections", composite)

    cv2.waitKey(0)

if __name__ == "__main__":
    main()