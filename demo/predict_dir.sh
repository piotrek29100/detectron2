python demo/demo.py \
--config-file configs/DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml \
--input "./../test_images/**" \
--output "./../0429999_test_images_results" \
--confidence-threshold 0.5 \
--opts MODEL.WEIGHTS "./output/20200607T1810/model_0429999.pth" MODEL.DEVICE cpu
#--opts MODEL.WEIGHTS "./output/20200505T1026/model_0124999.pth" MODEL.DEVICE cpu