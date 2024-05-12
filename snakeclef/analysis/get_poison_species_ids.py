import pandas as pd
from pathlib import Path

META_DIR = Path(__file__).parent / "metadata"
from paths import TRAIN_DATA_DIR

poison_status = pd.read_csv(META_DIR / "venomous_status_list.csv")

# all files in META_DIR if Metadata in the filename
metadata_files = [f for f in META_DIR.iterdir() if "Metadata" in f.name]
# read and combine all metadata files
metadata = pd.concat([pd.read_csv(f) for f in metadata_files])

train_metadata = META_DIR / "SnakeCLEF2023-TrainMetadata-iNat.csv"
train_metadata = pd.read_csv(train_metadata)

# assert that class_id and species in metadata have a 1:1 mapping
assert len(metadata["class_id"].unique()) == len(metadata["binomial_name"].unique())

venom_ids = poison_status[poison_status["MIVS"] == 1]["class_id"].sort_values()

# print a list that can be copy-pased of the venom_ids
print(venom_ids.to_list())

print(len(metadata["class_id"].unique()))

# check that each image exists in the training data at TRAIN_DATA_DIR / image_path
found_files = 0
missing_files = 0
for image_path in metadata["image_path"]:
    try:
        assert (TRAIN_DATA_DIR / image_path).exists()
        assert (TRAIN_DATA_DIR / image_path).is_file()
        found_files += 1
    except AssertionError:
        print(f"Missing file: {image_path}")
        missing_files += 1

print(f"Found {found_files} files")
print(f"Missing {missing_files} files")
# percentage missing:
print(f"Missing {missing_files / (found_files + missing_files) * 100:.2f}% of files")



