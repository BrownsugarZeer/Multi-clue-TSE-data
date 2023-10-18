import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod

# text clue
import json
from transformers import pipeline

# visual clue
import decord
from PIL import Image
from transformers import SwinModel, ViTImageProcessor

# tag clue
import hashlib
import os
import soundfile as sf
import torch
from audioset_tagging_cnn.pytorch.models import *
from audioset_tagging_cnn.pytorch.pytorch_utils import move_data_to_device
from audioset_tagging_cnn.utils.config import classes_num


class BaseClue(ABC):
    """Base class for all clues."""

    name: str = ""
    description: str = ""

    @abstractmethod
    def compute_feat(self, input, **kwargs):
        """Compute feature for a given audio/video/text."""


class TextClue(BaseClue, ABC):
    name: str = "text_clue"
    description: str = "Natural language description about the target sound."

    def __init__(self, caption_path: Path):
        # This file was created with a pretrained audio caption model for anchor audios:
        # https://github.com/wsntxxn/AudioCaption
        self.caps = json.load(open(caption_path, encoding="utf-8"))
        self.model = pipeline(
            "feature-extraction",
            model="distilroberta-base",
            tokenizer="distilroberta-base",
            device=0,
        )

    def compute_feat(self, sound_id: str, **kwargs):
        return np.asarray(self.model(self.caps[sound_id]))[0][1:-1]


class VisualClue(BaseClue, ABC):
    name: str = "visual_clue"
    description: str = "A video clip related to the target sound."

    def __init__(self, audioset_audiocaps_dir: Path, anchors_path: Path):
        decord.bridge.set_bridge("torch")

        self.id2video = {
            video.stem: str(video)
            for video in audioset_audiocaps_dir.glob("**/*.mp4")
        }

        self.anchors = {}
        with open(anchors_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip().split()
                assert len(line) == 2
                mp4_id, anchor = line
                anchor = float(anchor)
                if mp4_id in self.anchors:
                    assert anchor == self.anchors[mp4_id]
                self.anchors[mp4_id] = anchor

        # Using tiny model to test. In our paper, the model is the large one.
        model_name = "microsoft/swin-tiny-patch4-window7-224"
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        self.model = SwinModel.from_pretrained(model_name)
        self.model.cuda()

    def compute_feat(self, sound_id: str, **kwargs):
        anchor = self.anchors[sound_id]
        vr = decord.VideoReader(self.id2video[sound_id])
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

        inputs = self.feature_extractor(images=frames, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}

        outputs = self.model(**inputs)
        rtv = outputs.pooler_output.squeeze(0).detach().cpu().numpy()

        return rtv


class TagClue(BaseClue, ABC):
    name: str = "tag_clue"
    description: str = "A simple linear layer takes a one-hot sound event tag as input."

    def __init__(self):
        self.device = torch.device("cuda")
        self.model = self._load_model(
            "Cnn14_16k_mAP=0.438.pth",
            "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1",
            "362fc5ff18f1d6ad2f6d464b45893f2c",
        )

    def _load_model(self, model_name, url, md5):
        # Download ckpt from audioset_tagging_cnn(https://github.com/qiuqiangkong/audioset_tagging_cnn)
        if (
            not os.path.exists(model_name)
            or hashlib.md5(open(model_name, "rb").read()).hexdigest() != md5
        ):
            os.system(f"wget -O {model_name} {url}")
        else:
            print(f"Model {model_name} already exists, skip.")

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

        checkpoint = torch.load(model_name, map_location=self.device)
        model.load_state_dict(checkpoint['model'])

        if "cuda" in str(self.device):
            model.to(self.device)
            model = torch.nn.DataParallel(model)
            print(f"GPU number: {torch.cuda.device_count()}")
        else:
            print("Using CPU.")
        return model

    def compute_feat(self, sound_path: str, **kwargs):
        waveform, _ = sf.read(sound_path)
        waveform = waveform[None, :]    # (1, audio_length)
        waveform = move_data_to_device(waveform, self.device)

        with torch.no_grad():
            self.model.eval()
            batch_output_dict = self.model(waveform, None)
        clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]

        one_hot = np.zeros(clipwise_output.shape, dtype='float32')
        one_hot[np.argmax(clipwise_output)] = 1

        return one_hot
