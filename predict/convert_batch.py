# python detectron2/predict/convert_batch.py --bucket-name file-search-corpus --file-prefix "dev-pdfs" --skip 500
from predictor_api import PredictorApi
import cloud_storage_utils as cloud_storage_utils
import os
import shutil
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import logging
import argparse
import json


def get_args():
    """Define the arguments with the default values."""

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--bucket-name", required=True, type=str)
    args_parser.add_argument("--file-prefix", required=True, type=str)
    args_parser.add_argument("--skip", required=False, type=int, default=0)
    args_parser.add_argument("--override", required=False, action="store_true")
    return args_parser.parse_args()


def configure_logging(output_path):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output_path, "w", "utf-8")
    formatter = logging.Formatter(
        "%(asctime)s: %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def main():
    args = get_args()
    torch.multiprocessing.freeze_support()

    configure_logging("logs.out")

    api = PredictorApi(
        "detectron2/configs/DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml",
        "detectron2/output/20200607T1810/model_0429999.pth",
        "cuda",
    )

    storage_client = cloud_storage_utils.get_storage_client()
    bucket = storage_client.get_bucket(args.bucket_name)
    all_blobs = list(
        storage_client.list_blobs(args.bucket_name, prefix=args.file_prefix)
    )
    blobs = [b for b in all_blobs if b.name.endswith(".jpg")]
    all_blobs_names = [b.name for b in all_blobs]

    if not args.override:
        blobs = [
            blob
            for blob in blobs
            if not _get_detectron_blob_name(args.file_prefix, blob.name)
            in all_blobs_names
        ]

    blobs = blobs[args.skip :]

    print(f"Blobs number: {len(blobs)}")

    images_per_batch = 100
    batches = int(np.ceil(len(blobs) / images_per_batch))
    batches = list(range(batches))
    image_width = 1200

    classes = ["text", "title", "list", "table", "figure"]

    for batch in batches:
        logging.info(f"Batch {batch}")

        shutil.rmtree("tmp", ignore_errors=True)
        Path("tmp").mkdir(parents=True, exist_ok=True)

        paths = []
        image_size_list = []
        for i in range(
            batch * images_per_batch,
            min((batch + 1) * images_per_batch, len(blobs)),
        ):
            blob = blobs[i]
            path = os.path.join("tmp", blob.name.split("/")[-1])
            blob.download_to_filename(path)
            image = Image.open(path)
            scale = image.size[0] / image_width
            new_image = image.resize(
                (
                    int(np.ceil(image.size[0] / scale)),
                    int(np.ceil(image.size[1] / scale)),
                )
            )
            new_image.save(path)
            paths.append(path)
            image_size_list.append([new_image.size[0], new_image.size[1]])

        results = list(api.predict_batch(paths))
        segments_paths = [path + ".detectron" for path in paths]

        for result, segments_path, path, image_size in zip(
            results, segments_paths, paths, image_size_list
        ):
            logging.info(f"[{i}] Blob: {blob.name}")
            with open(segments_path, "w", encoding="utf-8") as f:
                data = [
                    {
                        "class": classes[int(r[0])],
                        "x1": r[1] / image_size[0],
                        "y1": r[2] / image_size[1],
                        "x2": r[3] / image_size[0],
                        "y2": r[4] / image_size[1],
                        "score": r[5],
                    }
                    for r in result
                ]

                json.dump(data, f, ensure_ascii=False)

            blob = bucket.blob(
                _get_detectron_blob_name(args.file_prefix, path)
            )
            blob.upload_from_filename(path + ".detectron")

        shutil.rmtree("tmp", ignore_errors=True)


def _get_detectron_blob_name(file_prefix, image_path):
    return f"{file_prefix}/{image_path.split('/')[-1][:-4]}.detectron"


if __name__ == "__main__":
    main()
