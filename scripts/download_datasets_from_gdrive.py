import os
import sys
import zipfile
import gdown

DATASETS_DIR = "datasets"

DATASET_IDS = {
    "MOT17": "1ZreYUD3-UZhmj6yaouFI_lbhX3WHafN9",
    "MOT20": "187bYy2a1wkUcOegbvzOfitGSF99DL9ug",
    "PersonPath22": "FILE_ID_FOR_PERSONPATH22",
    "VIRAT-S": "1YkYjzp89tDiByfqFUT37baLWnVqMHpU4",
    "KITTI": "1qwfNvF6dEqkC_0BwseFK8-BYL1OZ52M_"
}


def download_and_extract(dataset_name):
    file_id = DATASET_IDS.get(dataset_name)

    if not file_id:
        print(f"Dataset name '{dataset_name}' not recognized. Skipping...")
        return

    if not os.path.exists(DATASETS_DIR):
        os.makedirs(DATASETS_DIR)

    os.chdir(DATASETS_DIR)

    print(f"Downloading {dataset_name} dataset...")
    gdown.download(
        f"https://drive.google.com/uc?id={file_id}", f"{dataset_name}.zip", quiet=False)

    zip_path = f"{dataset_name}.zip"
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()
        os.remove(zip_path)
        print(f"Extraction complete. {zip_path} has been removed.")
    else:
        print(f"Failed to download {zip_path}")

    os.chdir("..")


def main():
    if len(sys.argv) < 2:
        print("Usage: python download_datasets_from_gdrive.py <dataset_name1> <dataset_name2> ...")
        print("Please provide the name(s) of the dataset(s) (e.g., MOT17, MOT20, PersonPath22, VIRAT-S, KITTI).")
        sys.exit(1)

    for dataset_name in sys.argv[1:]:
        download_and_extract(dataset_name)


if __name__ == "__main__":
    main()
