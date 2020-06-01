from google.cloud import storage
import os

def get_storage_client(service_account_credentials_path=None):
    if service_account_credentials_path is None:
        service_account_credentials_path = get_credentials_path()
    # Explicitly use service account credentials by specifying the private key file.
    return storage.Client.from_service_account_json(service_account_credentials_path)

def get_credentials_path():
    try:
        service_account_credentials_path = next(
            f for f in os.listdir() if ".json" in f and "key" in f
        )
    except:
        raise Exception(
            "Specify path to google cloud storage credentials "
            + "by passing --service-account-credentials-path "
            + "argument or add json file with key in name e.g. "
            + "key.json to the root directory"
        )
    return service_account_credentials_path
