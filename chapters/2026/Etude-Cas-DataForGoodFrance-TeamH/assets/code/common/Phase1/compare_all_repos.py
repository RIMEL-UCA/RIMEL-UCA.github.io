import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Raw input data
# -------------------------
projects = {
    "SHIFTPORTAL": {
        "data_file_ratio": 1.0,
        "data_raw_ratio": 0.196,
        "data_code_ratio": 0.208,
        "data_formats_count": 3,
        "notebooks_ratio": 0.0
    },
    "ECLAIREUR_PUBLIC": {
        "data_file_ratio": 0.191,
        "data_raw_ratio": 0.056,
        "data_code_ratio": 0.105,
        "data_formats_count": 4,
        "notebooks_ratio": 0.0
    },
    "POLLUTION_EAU": {
        "data_file_ratio": 0.567,
        "data_raw_ratio": 0.296,
        "data_code_ratio": 0.12,
        "data_formats_count": 4,
        "notebooks_ratio": 0.141
    },
    "INEGALITES_CINEMA": {
        "data_file_ratio": 0.335,
        "data_raw_ratio": 0.058,
        "data_code_ratio": 0.382,
        "data_formats_count": 5,
        "notebooks_ratio": 0.033
    }
}

# -------------------------
# Normalize data_formats_count
# -------------------------
max_formats = max(
    p["data_formats_count"] for p in projects.values()
)

for p in projects.values():
    p["data_formats_count"] = p["data_formats_count"] / max_formats

# -------------------------
# Radar setup
# -------------------------
labels = list(next(iter(projects.values())).keys())
num_vars = len(labels)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# -------------------------
# Plot
# -------------------------
for name, metrics in projects.items():
    values = list(metrics.values())
    values += values[:1]

    ax.plot(angles, values, linewidth=2, label=name)
    ax.fill(angles, values, alpha=0.15)

# -------------------------
# Formatting
# -------------------------
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_ylim(0, 1)

ax.set_title("Comparative Data Structure Profile (Normalized Radar)", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.show()
