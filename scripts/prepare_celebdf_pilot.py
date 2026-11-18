
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.preprocess_faces import get_face_detector, extract_from_video
from src.utils.config import get_data_config, ensure_dir, PROJECT_ROOT
from tqdm import tqdm

SEED = 42
FRAMES = 4
FACE_SIZE = 224
MIN_FACE = 64

DATASET = PROJECT_ROOT / "datasets" / "Celeb-DF-v2"
TEST_LIST = DATASET / "List_of_testing_videos.txt"

FACES_ROOT = PROJECT_ROOT / "data" / "faces" / "celebdf"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "celebdf"


def read_official_test():
    entries = []
    with open(TEST_LIST, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue

            official_label = int(parts[0])
            relative_path = parts[1].replace("/", "\\")
            video_path = DATASET / relative_path

            if not video_path.is_file():
                raise FileNotFoundError(video_path)

            # Official Celeb-DF: 1 = REAL, 0 = FAKE
            label = 0 if official_label == 1 else 1

            entries.append({
                "path": video_path,
                "label": label,
                "method": "real" if label == 0 else "fake",
                "video_id": "celebdf_test_" +
                            relative_path.replace("\\", "_").replace(".mp4", "")
            })
    return entries


def collect_remaining(test_entries):
    test_paths = {str(x["path"].resolve()) for x in test_entries}

    real, fake = [], []

    for folder in ("Celeb-real", "YouTube-real"):
        for path in (DATASET / folder).glob("*.mp4"):
            if str(path.resolve()) not in test_paths:
                real.append({
                    "path": path,
                    "label": 0,
                    "method": "real",
                    "video_id": f"celebdf_{folder}_{path.stem}"
                })

    for path in (DATASET / "Celeb-synthesis").glob("*.mp4"):
        if str(path.resolve()) not in test_paths:
            fake.append({
                "path": path,
                "label": 1,
                "method": "fake",
                "video_id": f"celebdf_Celeb-synthesis_{path.stem}"
            })

    return real, fake


def choose(items, count, rng):
    if len(items) < count:
        raise RuntimeError(
            f"Need {count} videos, but only found {len(items)}"
        )
    return rng.sample(items, count)


def main():
    if not TEST_LIST.is_file():
        raise FileNotFoundError(TEST_LIST)

    # Never overwrite existing face crops or manifests.
    for split in ("train", "val", "test"):
        face_dir = FACES_ROOT / split
        manifest = PROCESSED_ROOT / f"{split}_manifest.json"

        if manifest.exists():
            raise FileExistsError(
                f"Existing manifest: {manifest}. "
                "Move it safely before rerunning."
            )

        if face_dir.exists() and any(face_dir.iterdir()):
            raise FileExistsError(
                f"Existing face data: {face_dir}. "
                "Not overwriting it."
            )

    rng = random.Random(SEED)

    official_test = read_official_test()
    test_real = choose(
        [x for x in official_test if x["label"] == 0], 50, rng
    )
    test_fake = choose(
        [x for x in official_test if x["label"] == 1], 50, rng
    )

    remaining_real, remaining_fake = collect_remaining(official_test)

    train_real = choose(remaining_real, 40, rng)
    train_fake = choose(remaining_fake, 40, rng)

    used_real = {str(x["path"]) for x in train_real}
    used_fake = {str(x["path"]) for x in train_fake}

    val_real_pool = [x for x in remaining_real if str(x["path"]) not in used_real]
    val_fake_pool = [x for x in remaining_fake if str(x["path"]) not in used_fake]

    val_real = choose(val_real_pool, 10, rng)
    val_fake = choose(val_fake_pool, 10, rng)

    splits = {
        "train": train_real + train_fake,
        "val": val_real + val_fake,
        "test": test_real + test_fake,
    }

    for items in splits.values():
        rng.shuffle(items)

    cfg = get_data_config()
    face_size = cfg["sampling"].get("face_size", FACE_SIZE)
    min_face = cfg["sampling"].get("min_face_size", MIN_FACE)

    detector_type, detector = get_face_detector()
    print(f"\nFace detector: {detector_type}")

    ensure_dir(FACES_ROOT)
    ensure_dir(PROCESSED_ROOT)

    for split, items in splits.items():
        output_samples = []

        for item in tqdm(items, desc=f"Extracting {split}"):
            out_dir = FACES_ROOT / split / item["video_id"]

            saved = extract_from_video(
                item["path"],
                out_dir,
                detector_type,
                detector,
                max_frames=FRAMES,
                face_size=face_size,
                min_face=min_face,
            )

            if saved > 0:
                output_samples.append({
                    "video_id": item["video_id"],
                    "path": str(out_dir.relative_to(PROJECT_ROOT)),
                    "label": item["label"],
                    "num_frames": saved,
                    "method": item["method"],
                    "source_video": str(item["path"]),
                })

        manifest_path = PROCESSED_ROOT / f"{split}_manifest.json"

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "dataset": "celebdf",
                "split": split,
                "samples": output_samples
            }, f, indent=2)

        real_count = sum(x["label"] == 0 for x in output_samples)
        fake_count = sum(x["label"] == 1 for x in output_samples)

        print(
            f"\n{split}: requested={len(items)}, "
            f"processed={len(output_samples)}, "
            f"real={real_count}, fake={fake_count}"
        )
        print(f"Manifest: {manifest_path}")

    print("\nPilot preprocessing finished.")
    print("Check the processed counts before training.")


if __name__ == "__main__":
    main()
