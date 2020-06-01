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

def get_args():
    """Define the arguments with the default values."""

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--bucket-name", required=True, type=str)
    args_parser.add_argument("--file-prefix", required=True, type=str)
    args_parser.add_argument("--skip", required=False, type=int, default=0)
    return args_parser.parse_args()


def configure_logging(output_path):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output_path, "w", "utf-8")
    formatter = logging.Formatter("%(asctime)s: %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def main():
    args = get_args()
    torch.multiprocessing.freeze_support()

    configure_logging("logs.out")

    api = PredictorApi(
        "detectron2/configs/DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml",
        "detectron2/output/20200505T1026/model_0124999.pth",
        "cuda",
    )

    storage_client = cloud_storage_utils.get_storage_client()
    bucket = storage_client.get_bucket(args.bucket_name)
    blobs = list(storage_client.list_blobs(args.bucket_name, prefix=args.file_prefix))
    blobs = [
        b for b in blobs if b.name.endswith(".jpg") and not ".segmented." in b.name
    ]
    blobs = blobs[args.skip:]

    print(f"Blobs number: {len(blobs)}")

    images_per_batch = 100
    batches = int(np.ceil(len(blobs) / images_per_batch))
    batches = list(range(batches))
    image_width = 1200

    for batch in batches:
        logging.info(f"Batch {batch}")

        shutil.rmtree("tmp", ignore_errors=True)
        Path("tmp").mkdir(parents=True, exist_ok=True)

        paths = []
        image_size_list = []
        for i in range(
            batch * images_per_batch, min((batch + 1) * images_per_batch, len(blobs))
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
        csv_paths = [path + ".segmented.csv" for path in paths]

        for result, csv_path, path, image_size in zip(
            results, csv_paths, paths, image_size_list
        ):
            logging.info(f"[{i}] Blob: {blob.name}")
            with open(csv_path, "w") as f:
                f.write("class,x1,y1,x2,y2\n")
                for r in result:
                    r[1] = r[1] / image_size[0]
                    r[2] = r[2] / image_size[1]
                    r[3] = r[3] / image_size[0]
                    r[4] = r[4] / image_size[1]
                    f.write(",".join([str(x) for x in r]) + "\n")

            blob = bucket.blob(
                f"{args.file_prefix}/{path.split('/')[-1][:-4]}.segmented.csv"
            )
            blob.upload_from_filename(path + ".segmented.csv")

        shutil.rmtree("tmp", ignore_errors=True)


if __name__ == "__main__":
    main()
