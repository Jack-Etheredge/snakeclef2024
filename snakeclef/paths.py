from pathlib import Path

CHECKPOINT_DIR = Path("__file__").parent.absolute() / "model_checkpoints"
DATA_DIR = Path("~/datasets/snakeclef_resized").expanduser().absolute()
TRAIN_DATA_DIR = DATA_DIR / "train"
VAL_DATA_DIR = DATA_DIR / "val"
METADATA_DIR = Path("__file__").parent.absolute() / "metadata"


def setup_project():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    setup_project()
