import csv
import decord
import numpy as np
import pandas as pd
from PIL import Image
from transformers import SwinModel, ViTImageProcessor
from pathlib import Path
from utils.tracker import progress_bar
decord.bridge.set_bridge('torch')


# Replace the following dirs with your AudioSet and AudioCaps data path here.
# Assume all .mp4 files are included in {audiocaps_dir} and {audioset_dir}
current_dir = Path(".")
output_dir = current_dir.joinpath("data", "tse_simulated")
audioset_audiocaps_dir = current_dir.joinpath("data", "tse_anchors")
anchors_path = current_dir.joinpath("annotations", "audioset/all.txt")
folders = ["train", "valid", "test", "unseen"]

# Using tiny model to test. In our paper, the model is the large one.
# model_name = "microsoft/swin-tiny-patch4-window7-224"
model_name = "microsoft/swin-large-patch4-window7-224"
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
model = SwinModel.from_pretrained(model_name)
model.cuda()


def compute_feat(video_path, anchor):
    vr = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()
    duration = len(vr) / fps

    if anchor - 1.0 < 0:
        anchor = 1.01
    if anchor + 1.0 > duration:
        anchor = duration - 1.01

    start_t = int((anchor - 1.0) * fps)
    end_t = int((anchor + 1.0) * fps)

    frames = vr.get_batch(list(range(start_t, end_t))).numpy()
    frames = [Image.fromarray(f) for f in frames[-60:]]

    inputs = feature_extractor(images=frames, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}

    outputs = model(**inputs)
    rtv = outputs.pooler_output.squeeze(0).detach().cpu().numpy()

    return rtv


if __name__ == "__main__":
    id2video = {
        video.stem: str(video)
        for video in audioset_audiocaps_dir.glob("**/*.mp4")
    }

    anchors = {}
    with open(anchors_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split()
            assert len(line) == 2
            mp4_id, anchor = line
            anchor = float(anchor)
            if mp4_id in anchors:
                assert anchor == anchors[mp4_id]
            anchors[mp4_id] = anchor

    for folder in folders:
        try:
            utts = pd.read_csv(
                output_dir.joinpath(folder, "annotation.csv"),
                usecols=["id", "s1"],
                index_col="id",
            ).to_dict()["s1"]
        except FileNotFoundError:
            continue

        output_dir = output_dir.joinpath(folder, "visual_clue")
        output_dir.mkdir(parents=True, exist_ok=True)
        annotation_csv = output_dir.joinpath("annotation.csv")

        with (
            progress_bar,
            open(annotation_csv, mode='w', encoding="utf-8", newline='') as csv_file
        ):
            writer = csv.writer(csv_file)
            writer.writerow(['id', 'visual_clue_path'])
            task_id = progress_bar.add_task(
                "feature", filename=f"Visual clue for {folder} set", total=len(utts)
            )

            for idx, key in enumerate(utts.keys()):
                vid = key.split('_mix_')[0]
                clue_path = output_dir.joinpath(f"{key}.npy")
                clue_emb = compute_feat(id2video[vid], anchors[vid])

                np.save(clue_path, clue_emb)
                writer.writerow([idx, clue_path])
                progress_bar.update(task_id, advance=1)

                # Debug
                if idx > 5:
                    break
