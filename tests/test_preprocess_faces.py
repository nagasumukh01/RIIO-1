from __future__ import annotations

from collections import Counter

from scripts.preprocess_faces import split_videos


def test_split_videos_keeps_groups_separate_and_balances_subset():
    videos = []
    for index in range(100):
        videos.append({
            "video_id": f"real_{index}",
            "group_id": f"source_{index}",
            "label": 0,
        })
        videos.append({
            "video_id": f"fake_{index}",
            "group_id": f"source_{index}",
            "label": 1,
        })

    splits = split_videos(videos, [0.7, 0.15, 0.15], seed=42, max_videos=200)
    expected_sizes = {"train": 140, "val": 30, "test": 30}
    seen_groups = {}

    for split_name, rows in splits.items():
        assert len(rows) == expected_sizes[split_name]
        assert Counter(row["label"] for row in rows) == {0: len(rows) // 2, 1: len(rows) // 2}
        for row in rows:
            group_id = row["group_id"]
            assert group_id not in seen_groups or seen_groups[group_id] == split_name
            seen_groups[group_id] = split_name
