import csv
import json
import numpy as np
import pandas as pd
from transformers import pipeline
from pathlib import Path
from utils.tracker import progress_bar


current_dir = Path(".")
output_dir = current_dir.joinpath("data", "tse_simulated")
caption_path = current_dir.joinpath("annotations", "audioset/caps.json")
folders = ["train", "valid", "test", "unseen"]

model = pipeline(
    "feature-extraction",
    model="distilroberta-base",
    tokenizer="distilroberta-base",
    device=0,
)


def compute_feat(text):
    feature = np.asarray(model(text))[0][1:-1]
    return feature


if __name__ == "__main__":
    # This file was created with a pretrained audio caption model for anchor audios:
    # https://github.com/wsntxxn/AudioCaption
    caps = json.load(open(caption_path, encoding="utf-8"))

    for folder in folders:
        try:
            utts = pd.read_csv(
                output_dir.joinpath(folder, "annotation.csv"),
                usecols=["id", "s1"],
                index_col="id",
            ).to_dict()["s1"]
        except FileNotFoundError:
            continue

        output_dir = output_dir.joinpath(folder, "text_clue")
        output_dir.mkdir(parents=True, exist_ok=True)
        annotation_csv = output_dir.joinpath("annotation.csv")

        with (
            progress_bar,
            open(annotation_csv, mode='w', encoding="utf-8", newline='') as csv_file
        ):
            writer = csv.writer(csv_file)
            writer.writerow(['id', 'text_clue_path'])
            task_id = progress_bar.add_task(
                "feature", filename=f"Text clue for {folder} set", total=len(utts)
            )

            for idx, key in enumerate(utts.keys()):
                vid = key.split('_mix_')[0]
                clue_path = output_dir.joinpath(f"{key}.npy")
                clue_emb = compute_feat(caps[vid])

                np.save(clue_path, clue_emb)
                writer.writerow([idx, clue_path])
                progress_bar.update(task_id, advance=1)

                # Debug
                if idx > 5:
                    break
