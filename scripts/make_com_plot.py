from argparse import ArgumentParser
from pathlib import Path

from src.caloch_eval import evaluate


def main():

    parser = ArgumentParser()
    parser.add_argument("--prediction-file", type=Path, help="Name of the prediction file (should be in the predictions folder of the trainer's default root dir)")
    parser.add_argument("--reference-file", type=Path, help="Path to the reference file (should be in the same format as the prediction file)")
    parser.add_argument("--eval-dataset", type=str, default="1-pion", help="Name of the evaluation dataset (should be the same as the one used for training)")
    parser.add_argument("--plot-dir", type=Path, default="plots", help="Directory where the plots will be saved")
    args = parser.parse_args()

    if not args.prediction_file.is_file():
        raise FileNotFoundError(f"Prediction file {args.prediction_file} does not exist.")
    if not args.reference_file.is_file():
        raise FileNotFoundError(f"Reference file {args.reference_file} does not exist.")

    evaluate.main(
        (
            f"-i {args.prediction_file} "
            f"-r {args.reference_file} "
            f"-m all -d {args.eval_dataset} "
            f"--output_dir {args.plot_dir} --cut 1.515e-3 --mode no-cls"
        ).split()
    )