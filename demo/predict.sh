python demo/demo.py \
--config-file configs/DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml \
--input "./../test_images/8_wzor_Protokolu_odbioru_robot_wykonawcy.pdf_0.jpg" "./../test_images/8_wzor_Protokolu_odbioru_robot_wykonawcy.pdf_1.jpg" "./../test_images/8_wzor_Protokolu_odbioru_robot_wykonawcy.pdf_2.jpg" "./../test_images/8_wzor_Protokolu_odbioru_robot_wykonawcy.pdf_3.jpg" "./../test_images/8_wzor_Protokolu_odbioru_robot_wykonawcy.pdf_4.jpg" \
--output "./../test_images_results" \
--confidence-threshold 0.5 \
--opts MODEL.WEIGHTS "./output/20200505T1026/model_0074999.pth" MODEL.DEVICE cpu