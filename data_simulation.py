import csv
import librosa
import numpy
import soundfile as sf
from pathlib import Path
from utils.tracker import progress_bar, logger

logger(filename="data_simulation_error_log.txt")

# Replace the following dirs with your AudioSet and AudioCaps data path here.
# Assume all .mp4 or .wav files are included in {audiocaps_dir} and {audioset_dir}
current_dir = Path(".")
audioset_audiocaps_dir = current_dir.joinpath("data", "tse_anchors")
anchors_path = current_dir.joinpath("annotations", "audioset/all.txt")
output_dir = current_dir.joinpath("data", "tse_simulated")
output_dir.mkdir(parents=True, exist_ok=True)

for path in [audioset_audiocaps_dir, anchors_path]:
    if not path.exists():
        raise FileNotFoundError("Please make sure you have all data downloaded.")

CLIP_LEN = 2.0
HCLIP_LEN = CLIP_LEN / 2

folders = {
    # "train": current_dir.joinpath("annotations", "audioset/train.txt"),
    "valid": current_dir.joinpath("annotations", "audioset/val.txt"),
    # "test": current_dir.joinpath("annotations", "audioset/test.txt"),
    # "unseen": current_dir.joinpath("annotations", "audioset/unseen.txt"),
}


def process_sound(audio_path, anchor):
    sound, sr = librosa.load(audio_path, sr=16000)
    start = int((anchor * sr)) - int(sr * HCLIP_LEN)
    end = int((anchor * sr)) + int(sr * HCLIP_LEN)

    if start < 0:
        start = 0
        end = 0 + int(CLIP_LEN * sr)
    elif end > len(sound):
        end = len(sound)
        start = end - int(CLIP_LEN * sr)

    sound = sound[start:end]
    eng = (sound ** 2).sum() + 1e-6
    sound = sound / numpy.sqrt(eng)
    return sound, sr


if __name__ == "__main__":
    anchors = {}
    with open(anchors_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split()
            assert len(line) == 2
            wav_id, anchor = line
            anchor = float(anchor)
            if wav_id in anchors:
                assert anchor == anchors[wav_id]
            anchors[wav_id] = anchor

    for folder, set_path in folders.items():
        annotation_csv = output_dir.joinpath(folder, "annotation.csv")

        for _ in ["wavs", "wavs/s1", "wavs/s2", "wavs/mix"]:
            output_dir.joinpath(folder, _).mkdir(parents=True, exist_ok=True)

        with (
            progress_bar,
            open(set_path, encoding="utf-8") as f,
            open(annotation_csv, mode='w', encoding="utf-8", newline='') as csv_file,
        ):
            writer = csv.writer(csv_file)
            writer.writerow(["id", "s1", "s2", "mix"])
            list_file = f.readlines()

            task_id = progress_bar.add_task(
                "simulate", filename=f"Data for {folder} set", total=len(list_file)
            )

            for line in list_file:
                wav_id_1, wav_id_2, gain_in_db = line.strip().split()

                s1_path = audioset_audiocaps_dir.joinpath(f"{wav_id_1}.wav")
                s2_path = audioset_audiocaps_dir.joinpath(f"{wav_id_2}.wav")

                if not s1_path.exists() or not s2_path.exists():
                    continue

                s1, sr = process_sound(s1_path, anchors[wav_id_1])
                s2, _ = process_sound(s2_path, anchors[wav_id_2])

                gain_in_db = float(gain_in_db)
                gain = 10 ** (gain_in_db / 20.0)
                s1 = s1 * gain

                mix = s1 + s2

                clip_max = abs(mix).max() + 1e-6
                mix = mix / clip_max * 0.9
                s1 = s1 / clip_max * 0.9
                s2 = s2 / clip_max * 0.9

                uid = wav_id_1 + "_mix_" + wav_id_2

                s1_path = output_dir.joinpath(folder, f"wavs/s1/{uid}.wav")
                s2_path = output_dir.joinpath(folder, f"wavs/s2/{uid}.wav")
                mix_path = output_dir.joinpath(folder, f"wavs/mix/{uid}.wav")

                sf.write(s1_path, s1, sr)
                sf.write(s2_path, s2, sr)
                sf.write(mix_path, mix, sr)
                writer.writerow([uid, s1_path, s2_path, mix_path])

                progress_bar.update(task_id, advance=1)
