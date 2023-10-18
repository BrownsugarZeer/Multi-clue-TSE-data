# (W.I.P)Prepare the Multi-clue-TSE dataset

This is the data simulation scirpt for paper "Target Sound Extraction (TSE) with Variable Cross-modality Clues".

## How to use

1. Clone this project: `git clone https://github.com/BrownsugarZeer/Multi-clue-TSE-data.git`
2. Clone audioset_tagging_cnn: `git clone https://github.com/qiuqiangkong/audioset_tagging_cnn.git`
3. Install pytorch(depends on your env setup): `pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113`
4. Install requirements: `pip install -r requirements.txt`
5. Installing [FFmpeg](#installing-ffmpeg)
6. Download the AudioSet and AudioCaps dataset: `python prepare_tse_anchor.py`
7. Run simulation script: `python data_simulation.py`
8. Prepare tag clues: `python gen_tag_clue.py`, the one-hot tag will be created in `output/[train|val|test|unseen]/tag_onehot/`
9. Prepare text clues: `python gen_text_clue.py`
10. Prepare visual clues: `python gen_visual_clue.py`

## Supported clues

- [x] Tag clue
- [x] Video clue
- [x] Text clue

## Installing FFmpeg

Before using `ffmpeg-python`, FFmpeg must be installed and accessible via the `$PATH` environment variable.

There are a variety of ways to install FFmpeg, such as the [official download links](https://ffmpeg.org/download.html), or using your package manager of choice (e.g. `sudo apt install ffmpeg` on Debian/Ubuntu, `brew install ffmpeg` on OS X, etc.). The infomation about [install on windows 10](https://annkuoq.github.io/blog/2019-12-17-install-ffmpeg/) can provide some additional details.

Regardless of how FFmpeg is installed, you can check if your environment path is set correctly by running the `ffmpeg` command from the terminal, in which case the version information should appear, as in the following example (truncated for brevity):

```bash
$ ffmpeg
ffmpeg version 4.2.4-1ubuntu0.1 Copyright (c) 2000-2020 the FFmpeg developers
  built with gcc 9 (Ubuntu 9.3.0-10ubuntu2)
```

> **Note**: The actual version information displayed here may vary from one system to another; but if a message such as `ffmpeg: command not found` appears instead of the version information, FFmpeg is not properly installed.

## Download the AudioSet and AudioCaps dataset

### AudioSet

The [AudioSet](https://research.google.com/audioset/download.html) dataset is a large-scale collection of human-labeled 10-second sound clips drawn from YouTube videos. To collect all our data we worked with human annotators who verified the presence of sounds they heard within YouTube segments. To nominate segments for annotation, we relied on YouTube metadata and content-based search.

The sound events in the dataset consist of a subset of the AudioSet [ontology](https://github.com/audioset/ontology). You can learn more about the dataset construction in our [ICASSP 2017 paper](https://research.google/pubs/pub45857/). Explore the dataset annotations by sound class below.


AudioSet provides data as YouTube IDs or 128-dimensional feature vectors. However, these formats have limitations. The feature vectors can't be converted back into original audio, and YouTube IDs only point to the audio's online location, not containing the audio data itself. Therefore, utilizing this data for certain training models can be challenging(that is a reason why we automatically download).

| Total | Available | Failed |
| ----- | --------- | ------ |
| 80477 | 79004     | 1473   |

- Failed means that `Video unavailable`, `Video has been removed`, `Video is private` and so on.

### AudioCaps

[AudioCaps](https://audiocaps.github.io/) is a subset of Audioset and is already included in `"annotations/audioset/all.txt`, so there is no need to download it again.

## Citations

```text

@inproceedings{liTargetSoundExtraction2023a,
  title = {Target {{Sound Extraction}} with {{Variable Cross-Modality Clues}}},
  booktitle = {{{ICASSP}} 2023 - 2023 {{IEEE International Conference}} on {{Acoustics}}, {{Speech}} and {{Signal Processing}} ({{ICASSP}})},
  author = {Li, Chenda and Qian, Yao and Chen, Zhuo and Wang, Dongmei and Yoshioka, Takuya and Liu, Shujie and Qian, Yanmin and Zeng, Michael},
  year = {2023},
  month = jun,
  pages = {1--5},
  doi = {10.1109/ICASSP49357.2023.10095266},
}
```
