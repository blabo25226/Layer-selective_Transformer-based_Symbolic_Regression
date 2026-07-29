import argparse
import matplotlib.pyplot as plt
import numpy as np
import json

# Adding parser arguments
parser = argparse.ArgumentParser(description="Generate and save radar chart for model F1 scores.")
parser.add_argument("--input", type=str, required=True, help="Path to the input JSON file containing data.")
parser.add_argument("--output", type=str, required=True, help="Path to save the radar chart image.")
args = parser.parse_args()

# Load data from JSON file
with open(args.input, "r") as f:
    data = json.load(f)

# Extract dataset names and models
datasets = list(data.keys())
models = ["RandomForest_1", "RandomForest_100", "LogisticRegression", "Boolformer"]

# Extract F1 scores for each model and dataset
f1_scores = {model: [data[ds][model]["f1_score"] for ds in datasets] for model in models}

# Calculate average F1 scores for legends
avg_f1_scores = {model: np.mean(scores) for model, scores in f1_scores.items()}

# Sort datasets by Boolformer's F1 score in ascending order
sorted_datasets = sorted(datasets, key=lambda ds: data[ds]["Boolformer"]["f1_score"])

# Extract sorted F1 scores for each model
sorted_f1_scores = {model: [max(0, data[ds][model]["f1_score"]) for ds in sorted_datasets] for model in models}

# Recalculate the radar chart angles based on sorted datasets
num_vars = len(sorted_datasets)
sorted_angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
sorted_angles = sorted_angles[::-1]  # Reverse angles to go clockwise
sorted_angles += sorted_angles[:1]  # Close the radar

# Adjust the angles for the start angle
start_angle = 90
start_angle_rad = np.deg2rad(start_angle)  # Convert start angle to radians
sorted_angles = [(angle + start_angle_rad) % (2 * np.pi) for angle in sorted_angles]


# Update the radar chart with sorted datasets
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

for model in models:
    values = sorted_f1_scores[model] + sorted_f1_scores[model][:1]  # Close the plot
    ax.plot(sorted_angles, values, label=f"{model} (avg F1: {avg_f1_scores[model]:.2f})")
    ax.fill(sorted_angles, values, alpha=0.1)

# Add sorted dataset names to the plot
ax.set_yticks(np.arange(0, 1.1, 0.2))
ax.set_xticks(sorted_angles[:-1])
ax.set_xticklabels(sorted_datasets, fontsize=10)

# Add a legend in the center of the graph
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='center', fontsize=13, bbox_to_anchor=(0.5, 0.5), frameon=True)

# Save plot to file
plt.tight_layout()
plt.savefig(args.output, dpi=300)
plt.show()
