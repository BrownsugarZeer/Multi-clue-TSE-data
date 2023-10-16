import logging
import pandas as pd
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import strftime, gmtime, sleep
from typing import Optional
from utils import progress_bar, logger

logger(filename="yt_dlp_error_log.txt")


def prepare_multi_clue_tse(
    save_folder,
    limit: Optional[int] = None,
    num_worker: int = 1,
    skip_prep: bool = False,
):
    """
    Downloads the Multi-clue TSE dataset.

    Arguments
    ---------
    save_folder: str
        Path to save the downloaded dataset.
    limit: int, optional
        Maximum number of files to download, if None, download all files.
    num_worker: int
        Number of workers to download the files, default is 1.
    skip_prep: bool
        If True, skip data preparation.

    Example
    -------
    >>> save_folder = '/path/to/audioset'
    >>> prepare_multi_clue_tse(save_folder, limit=10)
    """

    if skip_prep:
        return

    current_dir = Path(__file__).parent.resolve()

    # Parse dataset
    df = pd.read_csv(
        current_dir.joinpath("archives", "anchors/all.txt"),
        sep=" ",
        header=None,
        names=["wav_id", "anchor"],
    )

    df = df["wav_id"].str.rsplit("_", n=2, expand=True)
    df.columns = ["wav_id", "start_seconds", "end_seconds"]
    df = df.astype({"start_seconds": float, "end_seconds": float})

    if limit is not None:
        df = df.head(limit)

    # Download the audio files in multiple threads
    dataset_dir = current_dir.joinpath(save_folder, "tse_anchors")
    if not dataset_dir.is_dir():
        dataset_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Directory {dataset_dir} already exists.")

    n_clips = df['wav_id'].unique().size
    print(f"{n_clips} files are found.")

    def on_download_video(clip):
        fname = f"{clip.wav_id}_{int(clip.start_seconds)}_{int(clip.end_seconds)}"
        start_time = strftime("%H:%M:%S", gmtime(clip.start_seconds))
        end_time = strftime("%H:%M:%S", gmtime(clip.end_seconds))
        url = f"https://www.youtube.com/watch?v={clip.wav_id}"

        with subprocess.Popen(
            "yt-dlp"
            " -f \"bv*[ext=mp4]+ba[ext=m4a][asr=16000]/b[ext=mp4] / bv*+ba/b\""
            " --output \"{output_path}\""
            " --download-sections *{start_time}-{end_time}"
            " --quiet"
            " -x --audio-format {format} --audio-quality {quality}"
            " --keep-video"
            " --postprocessor-args \"ffmpeg:-ar 16000\""
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
            "download", filename="Multi-clue TSE Dataset", total=n_clips
        )
        with ThreadPoolExecutor(max_workers=num_worker) as exector:
            for clip in df.itertuples():
                exector.submit(on_download_video, clip)
                progress_bar.update(task_id, advance=1)
                sleep(random.uniform(0.5, 1.5))


if __name__ == "__main__":
    prepare_multi_clue_tse(
        save_folder="data",
        num_worker=10,
        limit=10,  # Debug
    )
