#python tools/train_net.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_wolf.yaml" SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)" TEST.IMS_PER_BATCH 1 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000
python tools/train_net.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_wolf.yaml"
