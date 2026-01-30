import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

INPUT = Path("results/data_processing_history.csv")
OUTPUT = Path("results/data_transformation_files_evolution.png")

ACTIVITIES = {
    "data_processing",
    "exploration",
    "ml_integration",
    "db_structuration"
}

START_DATE = "2024-01-01"


def plot_data_transformation_evolution():

    df = pd.read_csv(INPUT)
    df["date"] = pd.to_datetime(df["date"])

    # garder uniquement les activités pertinentes
    df = df[df["activity"].isin(ACTIVITIES)]
    df = df.sort_values(["date", "commit"])

    active_files = set()
    history = []

    for (commit, date), group in df.groupby(["commit", "date"], sort=False):
        for _, row in group.iterrows():
            path = row["file_path"]

            if row["change_type"] == "ADD":
                active_files.add(path)

            elif row["change_type"] == "DELETE":
                active_files.discard(path)

        history.append({
            "date": date,
            "count": len(active_files)
        })

    hist = pd.DataFrame(history).set_index("date")

    # série continue dans le temps
    hist = hist.resample("D").last().ffill().fillna(0)
    hist = hist[hist.index >= START_DATE]

    # --- Plot
    plt.figure(figsize=(12, 6))

    plt.plot(
        hist.index,
        hist["count"],
        drawstyle="steps-post",
        linewidth=2.5,
        label="Nombre de fichiers"
    )

    plt.fill_between(
        hist.index,
        hist["count"],
        step="post",
        alpha=0.15
    )

    plt.title(
        "Évolution du nombre d’artefacts de transformation de données",
        fontsize=15,
        pad=15
    )

    plt.xlabel("Date")
    plt.ylabel("Nombre de fichiers actifs")

    plt.grid(True, linestyle="--", alpha=0.4)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)

    # --- Annotation synthèse
    max_val = hist["count"].max()
    final_val = hist["count"].iloc[-1]
    duration = (hist.index[-1] - hist.index[0]).days

    plt.annotate(
        f"Max : {max_val} fichiers\n"
        f"Final : {final_val} fichiers\n"
        f"Période : {duration} jours",
        xy=(hist.index[-1], final_val),
        xytext=(-130, 40),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w", alpha=0.9),
        arrowprops=dict(arrowstyle="->", alpha=0.6)
    )

    plt.legend()
    plt.tight_layout()

    OUTPUT.parent.mkdir(exist_ok=True)
    plt.savefig(OUTPUT, dpi=200)
    plt.show()

    print(f"Saved: {OUTPUT.resolve()}")


if __name__ == "__main__":
    plot_data_transformation_evolution()
