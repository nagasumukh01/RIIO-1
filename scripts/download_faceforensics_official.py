#!/usr/bin/env python
"""Official FaceForensics++ approval downloader supplied by the user."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
import urllib.request
from os.path import join

from tqdm import tqdm

FILELIST_URL = "misc/filelist.json"
DEEPFAKES_DETECTION_URL = "misc/deepfake_detection_filenames.json"
DEEPFAKES_MODEL_NAMES = ["decoder_A.h5", "decoder_B.h5", "encoder.h5"]
DATASETS = {
    "original": "original_sequences/youtube",
    "DeepFakeDetection_original": "original_sequences/actors",
    "Deepfakes": "manipulated_sequences/Deepfakes",
    "DeepFakeDetection": "manipulated_sequences/DeepFakeDetection",
    "Face2Face": "manipulated_sequences/Face2Face",
    "FaceShifter": "manipulated_sequences/FaceShifter",
    "FaceSwap": "manipulated_sequences/FaceSwap",
    "NeuralTextures": "manipulated_sequences/NeuralTextures",
}
ALL_DATASETS = ["original", "DeepFakeDetection_original", "Deepfakes", "DeepFakeDetection", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]


def download_file(url: str, output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.isfile(output_file):
        tqdm.write(f"Skipping existing file {output_file}")
        return
    handle, temporary = tempfile.mkstemp(dir=os.path.dirname(output_file))
    os.close(handle)
    try:
        urllib.request.urlretrieve(url, temporary)
        os.replace(temporary, output_file)
    except Exception:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def download_files(filenames: list[str], base_url: str, output_path: str) -> None:
    for filename in tqdm(filenames):
        download_file(base_url + filename, join(output_path, filename))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download an approved FaceForensics++ subset")
    parser.add_argument("output_path")
    parser.add_argument("-d", "--dataset", default="all", choices=list(DATASETS) + ["all"])
    parser.add_argument("-c", "--compression", default="c23", choices=["raw", "c23", "c40"])
    parser.add_argument("-n", "--num_videos", type=int, default=10)
    parser.add_argument("--server", default="EU2", choices=["EU", "EU2", "CA"])
    args = parser.parse_args()
    servers = {"EU": "http://canis.vc.in.tum.de:8100/", "EU2": "http://kaldir.vc.in.tum.de/faceforensics/", "CA": "http://falas.cmpt.sfu.ca:8100/"}
    base_url = servers[args.server] + "v3/"
    tos_url = servers[args.server] + "webpage/FaceForensics_TOS.pdf"
    print("FaceForensics++ terms of use:", tos_url)
    print("You confirmed acceptance before running this downloader.")
    datasets = [args.dataset] if args.dataset != "all" else ALL_DATASETS
    for dataset in datasets:
        dataset_path = DATASETS[dataset]
        if "DeepFakeDetection" in dataset_path or "actors" in dataset_path:
            filepaths = json.loads(urllib.request.urlopen(base_url + DEEPFAKES_DETECTION_URL).read().decode("utf-8"))
            filelist = filepaths["actors" if "actors" in dataset_path else "DeepFakesDetection"]
        else:
            file_pairs = json.loads(urllib.request.urlopen(base_url + FILELIST_URL).read().decode("utf-8"))
            filelist = []
            if "original" in dataset_path:
                for pair in file_pairs:
                    filelist.extend(pair)
            else:
                for pair in file_pairs:
                    filelist.extend(["_".join(pair), "_".join(pair[::-1])])
        filelist = filelist[:args.num_videos]
        filelist = [filename + ".mp4" for filename in filelist]
        output = join(args.output_path, dataset_path, args.compression, "videos")
        print(f"Downloading {len(filelist)} videos for {dataset} to {output}")
        download_files(filelist, base_url + dataset_path + f"/{args.compression}/videos/", output)


if __name__ == "__main__":
    main()
