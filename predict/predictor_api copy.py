# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import numpy as np

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predict.predictor import VisualizationDemo
from detectron2.data.catalog import Metadata


class PredictorApi:
    def __init__(self, config_path, model_weights_path, model_device="cpu"):
        mp.set_start_method("spawn", force=True)
        self.logger = setup_logger()

        cfg = self._setup_cfg(
            config_path,
            [
                "MODEL.WEIGHTS",
                model_weights_path,
                "MODEL.DEVICE",
                model_device,
            ],
            0.5,
        )

        metadata = Metadata(
            evaluator_type="coco",
            name="PubLayNet",
            thing_classes=["text", "title", "list", "table", "figure"],
        )
        self.demo = VisualizationDemo(cfg, metadata, parallel=True)

    def _setup_cfg(self, config_file, opts, confidence_threshold):
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(opts)
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
            confidence_threshold
        )
        cfg.freeze()
        return cfg

    def predict(self, path, output_path):
        img = read_image(path, format="BGR")
        start_time = time.time()
        predictions, visualized_output = self.demo.run_on_image(img)
        self.logger.info(
            "{}: detected {} instances in {:.2f}s".format(
                path, len(predictions["instances"]), time.time() - start_time
            )
        )
        visualized_output.save(output_path)

        _classes = predictions["instances"].pred_classes.cpu().numpy()
        _boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()

        return np.concatenate([_classes.reshape(-1, 1), _boxes], axis=1)

    def predict_batch(self, paths):
        self.logger.info(f"Images number: {len(paths)}")
        images = [read_image(path, format="BGR") for path in paths]
        start_time = time.time()

        self.logger.info("Time {:.2f}s".format(time.time() - start_time))

        _classes = predictions["instances"].pred_classes.cpu().numpy()
        _boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()

        return np.concatenate([_classes.reshape(-1, 1), _boxes], axis=1)
