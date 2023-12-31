import logging
import csv
import pandas as pd
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from speechbrain.utils.data_utils import download_file
from time import strftime, gmtime, sleep
from typing import List, Optional
from customs.tracker import progress_bar, logger

logger(filename="prepare_audioset_error_log.txt")


def prepare_audioset(
    save_folder,
    classes: Optional[List[str]] = None,
    limit: Optional[int] = None,
    num_worker: int = 1,
    skip_prep: bool = False,
):
    """
    Downloads the Audioset dataset.

    Arguments
    ---------
    save_folder: str
        Path to save the downloaded dataset.
    classes: List[str], optional
        List of classes to download, if None, download all classes.
    limit: int, optional
        Maximum number of files to download, if None, download all files.
    num_worker: int
        Number of workers to download the files, default is 1.
    skip_prep: bool
        If True, skip data preparation.

    Example
    -------
    >>> save_folder = '/path/to/audioset'
    >>> prepare_audioset(save_folder, limit=10)
    """

    if skip_prep:
        return

    current_dir = Path(__file__).parent.resolve()

    if not Path(save_folder).is_dir():
        for csv_name in [
            "eval_segments.csv",
            "balanced_train_segments.csv",
            "unbalanced_train_segments.csv",
            "class_labels_indices.csv",
        ]:
            download_file(
                f"http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/{csv_name}",
                current_dir.joinpath(save_folder, csv_name),
            )

    if limit == 0:
        return

    # Get all the labels
    class_csv = current_dir.joinpath(save_folder, "class_labels_indices.csv")
    with open(class_csv, encoding="utf-8") as label_file:
        reader = csv.DictReader(label_file)
        (
            _,
            mid,
            display_name,
        ) = reader.fieldnames

        label_ids = []
        if classes is not None:
            label_ids = [
                row[mid]
                for row in reader
                for class_name in classes
                if class_name.lower() == row[display_name].lower()
            ]
        else:
            label_ids = [row[mid] for row in reader]

    # Create a csv file for each corresponding class
    balanced_csv = current_dir.joinpath(save_folder, "balanced_train_segments.csv")
    unbalanced_csv = current_dir.joinpath(save_folder, "unbalanced_train_segments.csv")

    _dfs = []
    for csv_file in [balanced_csv, unbalanced_csv]:
        df = pd.read_csv(
            csv_file,
            sep=", ",
            skiprows=3,
            header=None,
            names=["YTID", "start_seconds", "end_seconds", "positive_labels"],
            engine="python",
            index_col="positive_labels",
        )
        _dfs.append(df[df.index.str.contains("|".join(label_ids))])
    dfs = pd.concat(_dfs)
    del _dfs

    if limit is not None:
        dfs = dfs.head(limit)

    # Download the audio files
    dataset_dir = current_dir.joinpath(save_folder, "audioset")
    if not dataset_dir.is_dir():
        dataset_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Directory {dataset_dir} already exists.")

    n_clips = dfs["YTID"].unique().size
    print(f"{n_clips} files are found.")

    def on_download_video(clip):
        fname = f"{clip.YTID}_{int(clip.start_seconds)}_{int(clip.end_seconds)}"
        start_time = strftime("%H:%M:%S", gmtime(clip.start_seconds))
        end_time = strftime("%H:%M:%S", gmtime(clip.end_seconds))
        url = f"https://www.youtube.com/watch?v={clip.YTID}"

        with subprocess.Popen(
            "yt-dlp"
            ' -f "bv*[ext=mp4]+ba[ext=m4a][asr=16000]/b[ext=mp4] / bv*+ba/b"'
            ' --output "{output_path}"'
            " --download-sections *{start_time}-{end_time}"
            " --quiet"
            " -x --audio-format {format} --audio-quality {quality}"
            " --keep-video"
            ' --postprocessor-args "ffmpeg:-ar 16000"'
            " {url}".format(
                output_path=dataset_dir.joinpath(fname + ".%(ext)s"),
                start_time=start_time,
                end_time=end_time,
                url=url,
                format="wav",
                quality=0,  # 0 (best) and 10 (worst)
            ),
            shell=True,
        ) as process:
            try:
                exit_code = process.wait(timeout=30)
                if exit_code != 0:
                    logging.error("Failed to download [%s](%s)", fname, url)
                    return
            except subprocess.TimeoutExpired:
                logging.error("Timeout [%s](%s)", fname, url)
                process.terminate()  # give up
                return

    with progress_bar:
        task_id = progress_bar.add_task(
            "download", filename="Audioset Dataset", total=n_clips
        )
        with ThreadPoolExecutor(max_workers=num_worker) as exector:
            for clip in dfs.itertuples():
                exector.submit(on_download_video, clip)
                progress_bar.update(task_id, advance=1)
                sleep(random.uniform(0.5, 1.5))


if __name__ == "__main__":
    prepare_audioset(
        save_folder="temp",
        classes=["Shout"],
        limit=1,
    )
