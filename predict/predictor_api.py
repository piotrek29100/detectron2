# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import numpy as np
import torch
from collections import deque

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import AsyncPredictor
from detectron2.data.catalog import Metadata


class PredictorApi:
    def __init__(self, config_path, model_weights_path, model_device="cpu"):
        mp.set_start_method("spawn", force=True)
        self.logger = setup_logger("detectron.out")

        cfg = self._setup_cfg(
            config_path,
            ["MODEL.WEIGHTS", model_weights_path, "MODEL.DEVICE", model_device,],
            0.5,
        )

        self.metadata = Metadata(
            evaluator_type="coco",
            name="PubLayNet",
            thing_classes=["text", "title", "list", "table", "figure"],
        )
        num_gpu = torch.cuda.device_count()
        self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)

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

    def predict_batch(self, paths):
        self.logger.info(f"Images number: {len(paths)}")
        images = [read_image(path, format="BGR") for path in paths]
        start_time = time.time()
        buffer_size = 5

        for i, image in enumerate(images):
            self.predictor.put(image)

            if i >= buffer_size:
                yield self._get_prediction(start_time)

        while len(self.predictor) > 0:
            yield self._get_prediction(start_time)

    def _get_prediction(self, start_time):
        predictions = self.predictor.get()

        self.logger.info("Time {:.2f}s".format(time.time() - start_time))

        _classes = predictions["instances"].pred_classes.cpu().numpy()
        _boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
        _scores = predictions["instances"].scores.cpu().numpy()

        return np.concatenate(
            [_classes.reshape(-1, 1), _boxes, _scores.reshape(-1, 1)], axis=1
        )
