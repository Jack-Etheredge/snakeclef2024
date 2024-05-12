import json

import pandas as pd
import numpy as np
import os

from hydra import compose, initialize
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from tqdm import tqdm
from PIL import Image
import torch
from pathlib import Path
from PIL import ImageFile

from closedset_model import build_model
from competition_metrics import evaluate
from datasets import get_valid_transform
from paths import METADATA_DIR, VAL_DATA_DIR
from utils import copy_config, get_device

np.set_printoptions(precision=5)
ImageFile.LOAD_TRUNCATED_IMAGES = True

class PytorchWorker:
    """Run inference using PyTorch."""

    def __init__(self, model_path: str, number_of_categories: int = 1784, model_id="efficientnet_b0", device="cpu"):

        ########################################
        # must be set before calling _load_model
        self.number_of_categories = number_of_categories
        self.model_id = model_id
        self.device = device
        ########################################

        self.transforms = TRANSFORMS
        # most other attributes must be set before calling _load_model, so call last
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        print("Setting up Pytorch Model")
        # model = models.efficientnet_b0()
        # model.classifier[1] = nn.Linear(in_features=1280, out_features=self.number_of_categories)
        model = build_model(
            model_id=self.model_id,
            pretrained=False,
            fine_tune=False,
            num_classes=self.number_of_categories,
            # this is all that matters. everything else will be overwritten by checkpoint state
            dropout_rate=0.5,
        ).to(self.device)
        model_ckpt = torch.load(model_path, map_location=self.device)
        model.load_state_dict(model_ckpt['model_state_dict'])

        return model.to(self.device).eval()

    def predict_image(self, image: np.ndarray) -> list():
        """Run inference using ONNX runtime.

        :param image: Input image as numpy array.
        :return: A list with logits and confidences.
        """

        logits = self.model(self.transforms(image).unsqueeze(0).to(self.device))

        return logits.tolist()


def make_submission(test_metadata, model_path, output_csv_path, images_root_path, cfg):
    """Make submission file"""

    device = get_device()

    model_id = cfg["evaluate"]["model_id"]
    model = PytorchWorker(model_path, model_id=model_id, device=device)

    # this allows use on both validation and test
    test_metadata.rename(columns={"observationID": "observation_id"}, inplace=True)
    test_metadata = test_metadata.drop_duplicates("observation_id", keep="first")

    predictions = []
    image_paths = test_metadata["image_path"]
    with torch.no_grad():
        for image_path in tqdm(image_paths):
            try:
                image_path = os.path.join(images_root_path, image_path)
                test_image = Image.open(image_path).convert("RGB")
                logits = model.predict_image(test_image)
                predictions.append(np.argmax(logits))
            except Exception as e:
                print(f"issue with image {image_path}: {e}")
                predictions.append(-1)

    test_metadata.loc[:, "class_id"] = predictions
    keep_cols = ["observation_id", "class_id"]
    test_metadata[keep_cols].to_csv(output_csv_path, index=None)


def evaluate_experiment(cfg, experiment_id):
    experiment_dir = Path("model_checkpoints") / experiment_id
    model_file = "model.pth"
    model_path = str(experiment_dir / model_file)
    predictions_output_csv_path = str(experiment_dir / "submission.csv")

    metadata_file_path = METADATA_DIR / "SnakeCLEF2023-ValMetadata.csv"
    test_metadata = pd.read_csv(metadata_file_path)

    make_submission(
        test_metadata=test_metadata,
        model_path=model_path,
        images_root_path=VAL_DATA_DIR,
        output_csv_path=predictions_output_csv_path,
        cfg=cfg,
    )

    scores_output_path = str(experiment_dir / f"competition_metrics_scores.json")

    # Generate metrics from predictions
    test_metadata.rename(columns={"observationID": "observation_id"}, inplace=True)
    test_metadata.drop_duplicates("observation_id", keep="first", inplace=True)
    y_true = test_metadata["class_id"].values
    submission_df = pd.read_csv(predictions_output_csv_path)
    submission_df.drop_duplicates("observation_id", keep="first", inplace=True)

    homebrewed_scores = calc_homebrewed_scores(y_true, np.copy(submission_df["class_id"].values))
    save_metrics(homebrewed_scores, metadata_file_path, predictions_output_csv_path,
                 scores_output_path)


def save_metrics(homebrewed_scores, metadata_file_path, predictions_with_unknown_output_csv_path, scores_output_path):
    # add additional competition metricspredictions_with_unknown_output_csv_path
    competition_metrics_scores = evaluate(
        test_annotation_file=metadata_file_path,
        user_submission_file=predictions_with_unknown_output_csv_path,
        phase_codename="prediction-based",
    )
    # deduplicate and flatten results
    competition_metrics_scores = competition_metrics_scores["submission_result"]
    competition_metrics_scores.update(homebrewed_scores)
    with open(scores_output_path, "w") as f:
        json.dump(competition_metrics_scores, f)


def calc_homebrewed_scores(y_true, y_pred):
    homebrewed_scores = dict()

    balanced_accuracy_known_classes = balanced_accuracy_score(y_true, y_pred)
    homebrewed_scores['balanced_accuracy_known_classes'] = balanced_accuracy_known_classes
    print("balanced accuracy on known classes:", balanced_accuracy_known_classes)
    accuracy_known_classes = accuracy_score(y_true, y_pred)
    homebrewed_scores['accuracy_known_classes'] = accuracy_known_classes
    print("unbalanced accuracy on known classes:", accuracy_known_classes)

    return homebrewed_scores


if __name__ == "__main__":

    with initialize(version_base=None, config_path="conf", job_name="evaluate"):
        cfg = compose(config_name="config")
    print(OmegaConf.to_yaml(cfg))

    experiment_id = cfg["evaluate"]["experiment_id"]
    model_id = cfg["evaluate"]["model_id"]
    image_size = cfg["evaluate"]["image_size"]

    TRANSFORMS = get_valid_transform(image_size=image_size, pretrained=True)

    copy_config("evaluate", experiment_id)

    print(f"evaluating experiment {experiment_id}")
    evaluate_experiment(cfg=cfg, experiment_id=experiment_id)
