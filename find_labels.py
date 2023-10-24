import pandas as pd

# Read the training set
df_all = pd.read_csv(
    "annotations/audioset/train.txt",
    sep="  ",
    usecols=[0, 1],
    engine="python",
    names=["wav_id", "positive_labels"],
)

df_all["wav_id"] = df_all["wav_id"].apply(lambda x: x.rsplit("_", maxsplit=2)[0])
df_all = (
    pd.concat([df_all["wav_id"], df_all["positive_labels"]])
    .drop_duplicates()
    .reset_index(drop=True)
    .to_frame(name="wav_id")
)
print(df_all.head(), df_all.shape)

# Read the class file from https://research.google.com/audioset/download.html
df_labels = pd.read_csv("tests/temp/class_labels_indices.csv")
print(df_labels.head(), df_labels.shape)

segments = {"balanced_train_segments": {}, "unbalanced_train_segments": {}}
for segment in segments:
    # Read the segments file from https://research.google.com/audioset/download.html
    df_segment = pd.read_csv(
        f"tests/temp/{segment}.csv",
        skiprows=3,
        header=None,
        usecols=[0, 3],
        delimiter=", ",
        engine="python",
        names=["YTID", "positive_labels"],
    )
    print(df_segment.head(), df_segment.shape)

    # Merge df_all and df_segment based on the common columns
    label_df = pd.merge(df_all, df_segment, left_on="wav_id", right_on="YTID", how="inner")

    def map_labels_to_class(label_str, df_labels):
        labels = label_str.split(",")
        indices = []
        display_names = []
        for label in labels:
            label = label.strip("\"")
            matched_row = df_labels[df_labels["mid"] == label]
            if not matched_row.empty:
                indices.append(str(matched_row["index"].values[0]))
                display_names.append(matched_row["display_name"].values[0])
        return ",".join(indices), ",".join(display_names)

    label_df["index"], label_df["display_name"] = zip(*label_df["positive_labels"].apply(lambda x: map_labels_to_class(x, df_labels)))
    label_df.drop(columns=["YTID"], inplace=True)
    print(label_df.head(), label_df.shape)

    # Split `index` by comma and convert to list of integers and then convert to set
    # note that we only take the first label(main class) in the list.
    label_df["index"] = label_df["index"].apply(lambda x: x.split(","))
    segments[segment] = set(int(sublist[0]) for sublist in label_df["index"].tolist())


total_labels = segments["balanced_train_segments"] | segments["unbalanced_train_segments"]
print(f"Total lable are {len(total_labels)}(same as paper mentioned).")
print(total_labels)

unused_labels = total_labels ^ set(range(527))
print(f"Unused lable are {len(unused_labels)}.")
print(unused_labels)
