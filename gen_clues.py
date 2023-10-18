import copy
import numpy as np
import pandas as pd
from pathlib import Path
from customs.tracker import progress_bar
from customs.clues import TextClue, VisualClue, TagClue

current_dir = Path(".")
output_dir = current_dir.joinpath("data", "tse_simulated")
clues = [
    TextClue(
        current_dir.joinpath("annotations", "audioset/caps.json")
    ),
    VisualClue(
        audioset_audiocaps_dir=current_dir.joinpath("data", "tse_anchors"),
        anchors_path=current_dir.joinpath("annotations", "audioset/all.txt"),
    ),
    TagClue(),
]
fields = {"ID": [], **{clue.name: [] for clue in clues}}
folders = {
    folder: copy.deepcopy(fields)
    for folder in ["train", "valid", "test", "unseen"]
}

if __name__ == "__main__":
    for folder in folders.keys():
        try:
            utts = pd.read_csv(
                output_dir.joinpath(folder, "annotation.csv"),
                usecols=["id", "s1"],
                index_col="id",
            ).to_dict()["s1"]
        except FileNotFoundError:
            continue

        with progress_bar:
            task_id = progress_bar.add_task(
                "feature", filename=f"In {folder} set", total=len(utts)
            )

            for clue in clues:
                output_dir.joinpath(folder, clue.name).mkdir(parents=True, exist_ok=True)

            for idx, key in enumerate(utts.keys()):
                vid = key.split('_mix_')[0]

                for clue in clues:
                    clue_path = output_dir.joinpath(folder, f"{clue.name}/{key}.npy")

                    if clue.name in ["text_clue", "visual_clue"]:
                        clue_emb = clue.compute_feat(vid)
                    elif clue.name in ["tag_clue"]:
                        clue_emb = clue.compute_feat(utts[key])
                    else:
                        raise NotImplementedError
                    np.save(clue_path, clue_emb)

                    folders[folder][clue.name].append(clue_path)
                folders[folder]["ID"].append(key)

                progress_bar.update(task_id, advance=1)

            # Debug
            # if idx > 5:
            #     break

    for split in folders:
        new_filename = current_dir.joinpath("annotations", f"{split}.csv")
        new_df = pd.DataFrame(folders[split])
        new_df.to_csv(new_filename, index=False)
