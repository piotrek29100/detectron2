from predict.predictor_api import PredictorApi
import predict.cloud_storage_utils as cloud_storage_utils
import os
import shutil
from pathlib import Path

bucket_name = "file-search-corpus"
file_prefix = "google-file-search-pdfs"

api = PredictorApi(
    "detectron2/configs/DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml",
    "detectron2/output/20200505T1026/model_0124999.pth",
    "cuda"
)

storage_client = cloud_storage_utils.get_storage_client()
bucket = storage_client.get_bucket(bucket_name)
blobs = list(storage_client.list_blobs(bucket_name, prefix=file_prefix))
blobs = [b for b in blobs if b.name.endswith(".jpg") and not ".segmented." in b.name]

print(f"Blobs number: {len(blobs)}")

blob = blobs[0]
Path("tmp").mkdir(parents=True, exist_ok=True)
path = os.path.join("tmp", blob.name.split("/")[-1])
blob.download_to_filename(path)

results = api.predict(path, path + ".segmented.jpg")
csv_path = path + ".segmented.csv"

with open(csv_path, "w") as f:
    f.write("class,x1,y1,x2,y2\n")
    for r in results:
        f.write(",".join([str(x) for x in r]) + "\n")

blob = bucket.blob(blob.name.split("/")[-1] + ".segmented.jpg")
blob.upload_from_filename(path + ".segmented.jpg")

blob = bucket.blob(blob.name.split("/")[-1] + ".segmented.csv")
blob.upload_from_filename(path + ".segmented.csv")

shutil.rmtree("tmp", ignore_errors=True)
