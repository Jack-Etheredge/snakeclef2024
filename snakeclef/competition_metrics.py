"""
https://huggingface.co/spaces/BVRA/SnakeCLEF2024
https://github.com/BohemianVRA/FGVC-Competitions/blob/feat/baselineTrainingForSnakeCLEF2024/2023/SnakeCLEF2023/evaluate.py
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

COLUMNS = ["observation_id", "class_id"]
# venomous_lvl = pd.read_csv(
#     "http://ptak.felk.cvut.cz/plants/plants/SnakeCLEF2023/venomous_status_list.csv"
# )
# VENOMOUS_SPECIES = venomous_lvl[venomous_lvl["MIVS"] == 1].class_id.unique()
VENOMOUS_SPECIES = {0, 1, 2, 3, 4, 5, 6, 27, 29, 30, 32, 33, 34, 110, 112, 113, 114, 150, 151, 152, 154,
                                   159, 160, 162, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217,
                                   218, 219, 221, 222, 223, 224, 225, 226, 228, 229, 232, 233, 234, 235, 236, 237, 238,
                                   240, 241, 242, 243, 244, 245, 247, 248, 249, 256, 257, 258, 259, 260, 261, 262, 279,
                                   280, 285, 301, 302, 307, 308, 309, 310, 402, 403, 405, 406, 407, 409, 410, 411, 412,
                                   413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 427, 428, 429, 430,
                                   431, 433, 434, 437, 439, 440, 442, 443, 462, 463, 464, 465, 470, 510, 511, 512, 513,
                                   562, 595, 596, 597, 598, 599, 600, 601, 739, 740, 741, 742, 743, 783, 804, 805, 806,
                                   828, 829, 830, 855, 856, 857, 858, 999, 1033, 1034, 1035, 1036, 1037, 1040, 1041,
                                   1042, 1044, 1046, 1048, 1049, 1050, 1052, 1053, 1056, 1057, 1059, 1060, 1061, 1063,
                                   1065, 1066, 1068, 1071, 1072, 1073, 1076, 1079, 1081, 1082, 1087, 1088, 1090, 1091,
                                   1093, 1094, 1106, 1107, 1108, 1109, 1110, 1112, 1113, 1114, 1115, 1116, 1117, 1118,
                                   1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1158, 1190, 1191,
                                   1192, 1229, 1230, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1327, 1328, 1329,
                                   1330, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1382, 1383, 1389, 1390, 1391, 1392,
                                   1393, 1395, 1396, 1452, 1528, 1529, 1530, 1650, 1651, 1652, 1653, 1662, 1663, 1665,
                                   1666, 1668, 1669, 1670, 1671, 1674, 1679, 1680, 1682, 1684, 1685, 1686, 1687, 1688,
                                   1689, 1703, 1713, 1714, 1734, 1735, 1736, 1737, 1739, 1740, 1741, 1742, 1743, 1744,
                                   1745, 1746, 1747, 1749}


def psc(merged_df, normalize=True):
    test_venomous_species = merged_df[
        (merged_df.class_id_gt.isin(VENOMOUS_SPECIES)) & (merged_df.class_id_gt != -1)
    ]
    test_harmless_species = merged_df[
        (~merged_df.class_id_gt.isin(VENOMOUS_SPECIES)) & (merged_df.class_id_gt != -1)
    ]

    # compute the confusion of venomous vs. harmless
    # target is harmless, prediction is harmless
    t_0_p_0 = merged_df[
        (~merged_df.class_id_gt.isin(VENOMOUS_SPECIES))
        & (~merged_df.class_id_pred.isin(VENOMOUS_SPECIES))
        & (merged_df.class_id_gt != merged_df.class_id_pred)
    ]

    # target is harmless, prediction is venomous
    t_0_p_1 = merged_df[
        (~merged_df.class_id_gt.isin(VENOMOUS_SPECIES))
        & (merged_df.class_id_pred.isin(VENOMOUS_SPECIES))
        & (merged_df.class_id_gt != merged_df.class_id_pred)
    ]

    # target is venomous, prediction is harmless
    t_1_p_0 = merged_df[
        (merged_df.class_id_gt.isin(VENOMOUS_SPECIES))
        & (~merged_df.class_id_pred.isin(VENOMOUS_SPECIES))
        & (merged_df.class_id_gt != merged_df.class_id_pred)
    ]

    # target is venomous, prediction is venomous
    t_1_p_1 = merged_df[
        (merged_df.class_id_gt.isin(VENOMOUS_SPECIES))
        & (merged_df.class_id_pred.isin(VENOMOUS_SPECIES))
        & (merged_df.class_id_gt != merged_df.class_id_pred)
    ]

    if normalize:
        t_0_p_0 = np.round(len(t_0_p_0) / len(test_harmless_species) * 100, 2)
        t_0_p_1 = np.round(len(t_0_p_1) / len(test_harmless_species) * 100, 2)
        t_1_p_0 = np.round(len(t_1_p_0) / len(test_venomous_species) * 100, 2)
        t_1_p_1 = np.round(len(t_1_p_1) / len(test_venomous_species) * 100, 2)
    else:
        t_0_p_0 = len(t_0_p_0)
        t_0_p_1 = len(t_0_p_1)
        t_1_p_0 = len(t_1_p_0)
        t_1_p_1 = len(t_1_p_1)

    return t_0_p_0, t_0_p_1, t_1_p_0, t_1_p_1


def track1_metric(results, w_f1=1, w_t_0_p_0=1, w_t_0_p_1=2, w_t_1_p_0=5, w_t_1_p_1=2):
    """
    Computes the track metric.

    Args:
        results:
        w_f1: Weight for macro F1 measure.
        w_t_0_p_0: Weight for confusion of harmless snake for another harmless one.
        w_t_0_p_1: Weight for confusion of harmless snake for a venomous one.
        w_t_1_p_0: Weight for confusion of venomous snake for a harmless one.
        w_t_1_p_1: Weight for confusion of venomous snake for another venomous one.

    Returns:
        The weighted metric.

    """
    return np.round(
        (
            (w_f1 * results["F1 Score"])
            + (w_t_0_p_0 * (100 - results["PSC"][0]))
            + (w_t_0_p_1 * (100 - results["PSC"][1]))
            + (w_t_1_p_0 * (100 - results["PSC"][2]))
            + (w_t_1_p_1 * (100 - results["PSC"][3]))
        )
        / (w_f1 + w_t_0_p_0 + w_t_0_p_1 + w_t_1_p_0 + w_t_1_p_1),
        2,
    )


def track2_metric(results, w_t_0_p_0=1, w_t_0_p_1=2, w_t_1_p_0=5, w_t_1_p_1=2):
    """
    Computes the track 2 metric.

    Args:
        results:
        w_t_0_p_0: Weight for confusion of harmless snake for another harmless one.
        w_t_0_p_1: Weight for confusion of harmless snake for a venomous one.
        w_t_1_p_0: Weight for confusion of venomous snake for a harmless one.
        w_t_1_p_1: Weight for confusion of venomous snake for another venomous one.

    Returns:
        The weighted metric.

    """
    return (
        w_t_0_p_0 * results["PSC_total"][0]
        + w_t_0_p_1 * results["PSC_total"][1]
        + w_t_1_p_0 * results["PSC_total"][2]
        + w_t_1_p_1 * results["PSC_total"][3]
    )


def evaluate_csv(test_annotation_file: str, user_submission_file: str) -> List[dict]:
    # load gt annotations
    gt_df = pd.read_csv(test_annotation_file, sep=",")
    for col in COLUMNS:
        assert col in gt_df, f"Test annotation file is missing column '{col}'."
    # keep only observation-based predictions
    gt_df = gt_df.drop_duplicates("observation_id")

    # load user predictions
    try:
        is_tsv = user_submission_file.endswith(".tsv")
        user_pred_df = pd.read_csv(user_submission_file, sep="\t" if is_tsv else ",")
    except Exception:
        print("Could not read file submitted by the user.")
        raise ValueError("Could not read file submitted by the user.")

    # validate user predictions
    missing_cols = []
    for col in COLUMNS:
        if col not in user_pred_df:
            missing_cols.append(col)

        if len(missing_cols) > 0:
            missing_cols_str = ", ".join(missing_cols)
            print(f"File submitted by the user is missing column(s) '{missing_cols_str}'.")
            raise ValueError(f"File submitted by the user is missing column '{missing_cols_str}'.")

    if len(gt_df) != len(user_pred_df):
        print(f"File submitted by the user should have {len(gt_df)} records.")
        raise ValueError(f"File submitted by the user should have {len(gt_df)} records.")
    missing_obs = gt_df.loc[
        ~gt_df["observation_id"].isin(user_pred_df["observation_id"]),
        "observation_id",
    ]
    if len(missing_obs) > 0:
        if len(missing_obs) > 3:
            missing_obs_str = ", ".join(missing_obs.iloc[:3].astype(str)) + ", ..."
        else:
            missing_obs_str = ", ".join(missing_obs.astype(str))
        print(f"File submitted by the user is missing observations: {missing_obs_str}")
        raise ValueError(f"File submitted by the user is missing observations: {missing_obs_str}")

    # merge dataframes
    merged_df = pd.merge(
        gt_df,
        user_pred_df,
        how="outer",
        on="observation_id",
        validate="one_to_one",
        suffixes=("_gt", "_pred"),
    )

    # evaluate accuracy_score and f1_score
    result = [
        {
            "test_split": {
                "F1 Score": np.round(
                    f1_score(
                        merged_df["class_id_gt"],
                        merged_df["class_id_pred"],
                        average="macro",
                    )
                    * 100,
                    2,
                ),
                "Accuracy": np.round(
                    accuracy_score(merged_df["class_id_gt"], merged_df["class_id_pred"]) * 100,
                    2,
                ),
                "PSC": psc(merged_df),
                "PSC_total": psc(merged_df, normalize=False),
            }
        }
    ]

    result[0]["test_split"]["Track1 Metric"] = track1_metric(result[0]["test_split"])
    result[0]["test_split"]["Track2 Metric"] = track2_metric(result[0]["test_split"])

    print(f"Evaluated scores: {result[0]['test_split']}")

    return result


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """
    print("Starting Evaluation.....")
    out = {}
    if phase_codename == "prediction-based":
        print("Evaluating for Prediction-based Phase")
        out["result"] = evaluate_csv(test_annotation_file, user_submission_file)

        # To display the results in the result file
        out["submission_result"] = out["result"][0]["test_split"]
        print("Completed evaluation")
    return out


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-annotation-file",
        help="Path to test_annotation_file on the server.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--user-submission-file",
        help="Path to a file created by predict script.",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    result = evaluate(
        test_annotation_file=args.test_annotation_file,
        user_submission_file=args.user_submission_file,
        phase_codename="prediction-based",
    )
    with open("scores.json", "w") as f:
        json.dump(result, f)
