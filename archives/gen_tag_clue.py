import csv
import hashlib
import os
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from audioset_tagging_cnn.pytorch.models import *
from audioset_tagging_cnn.pytorch.pytorch_utils import move_data_to_device
from audioset_tagging_cnn.utils.config import classes_num
from pathlib import Path
from customs.tracker import progress_bar
device = torch.device("cuda")

current_dir = Path(".")
output_dir = current_dir.joinpath("data", "tse_simulated")
folders = ["train", "valid", "test", "unseen"]


def download_model(model_name, url, md5):
    # Download ckpt from audioset_tagging_cnn(https://github.com/qiuqiangkong/audioset_tagging_cnn)
    if (
        not os.path.exists(model_name)
        or hashlib.md5(open(model_name, "rb").read()).hexdigest() != md5
    ):
        os.system(f"wget -O {model_name} {url}")
    else:
        print(f"Model {model_name} already exists, skip.")


def load_model(checkpoint_path):
    Model = eval("Cnn14_16k")
    model = Model(
        sample_rate=16000,
        window_size=512,
        hop_size=160,
        mel_bins=64,
        fmin=50,
        fmax=8000,
        classes_num=classes_num,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    if "cuda" in str(device):
        model.to(device)
        model = torch.nn.DataParallel(model)
        print(f"GPU number: {torch.cuda.device_count()}")
    else:
        print("Using CPU.")
    return model


download_model(
    "Cnn14_16k_mAP=0.438.pth",
    "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1",
    "362fc5ff18f1d6ad2f6d464b45893f2c",
)
model = load_model("./Cnn14_16k_mAP=0.438.pth")


def compute_feat(audio_path):
    waveform, _ = sf.read(audio_path)
    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)
    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]

    one_hot = np.zeros(clipwise_output.shape, dtype='float32')
    one_hot[np.argmax(clipwise_output)] = 1

    return one_hot


if __name__ == "__main__":
    for folder in folders:
        try:
            utts = pd.read_csv(
                output_dir.joinpath(folder, "annotation.csv"),
                usecols=["id", "s1"],
                index_col="id",
            ).to_dict()["s1"]
        except FileNotFoundError:
            continue

        output_dir = output_dir.joinpath(folder, "tag_clue")
        output_dir.mkdir(parents=True, exist_ok=True)
        annotation_csv = output_dir.joinpath("annotation.csv")

        with (
            progress_bar,
            open(annotation_csv, mode='w', encoding="utf-8", newline='') as csv_file
        ):
            writer = csv.writer(csv_file)
            writer.writerow(['id', 'tag_clue_path'])
            task_id = progress_bar.add_task(
                "feature", filename=f"Tag clue for {folder} set", total=len(utts)
            )

            for idx, key in enumerate(utts.keys()):
                vid = key.split('_mix_')[0]
                clue_path = output_dir.joinpath(f"{key}.npy")
                clue_emb = compute_feat(utts[key])

                np.save(clue_path, clue_emb)
                writer.writerow([idx, clue_path])
                progress_bar.update(task_id, advance=1)

                # Debug
                if idx > 5:
                    break
