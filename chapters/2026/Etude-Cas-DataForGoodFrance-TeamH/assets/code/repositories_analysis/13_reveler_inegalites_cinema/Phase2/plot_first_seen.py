import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_prepare(sources_csv, modes_csv):
    df_sources = pd.read_csv(sources_csv)
    df_sources["label"] = df_sources["source"]
    df_sources["category"] = "Source"

    df_modes = pd.read_csv(modes_csv)
    df_modes["label"] = df_modes["access_mode"]
    df_modes["category"] = "Mode d’accès"

    df = pd.concat(
        [
            df_sources[["label", "first_commit_date", "category"]],
            df_modes[["label", "first_commit_date", "category"]],
        ],
        ignore_index=True
    )

    df["first_commit_date"] = pd.to_datetime(df["first_commit_date"])
    return df


def plot_timeline(df, output_path):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))

    sns.stripplot(
        data=df,
        x="first_commit_date",
        y="label",
        hue="category",
        dodge=True,
        size=8
    )

    plt.xlabel("Date de première apparition")
    plt.ylabel("")
    plt.title("Chronologie d’introduction des sources de données et des modes d’accès")

    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    df = load_and_prepare(
        "results/first_seen_sources.csv",
        "results/first_seen_modes.csv"
    )
    plot_timeline(df, "results/first_seen_timeline.png")
