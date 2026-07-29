import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from argparse import ArgumentParser

# Argument Parser Setup
parser = ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Input JSON file path.")
parser.add_argument("--output", type=str, required=True, help="Output directory for saving plots.")
parser.add_argument("--num_points", type=int, required=True, help="Number of points to average for curves.")
args = parser.parse_args()

input_file = args.input
output_dir = args.output
num_points = args.num_points

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the JSON file
with open(input_file, "r") as f:
    data = json.load(f)

# Group data by number of active variables
grouped_active_vars = defaultdict(list)
for entry in data:
    nb_active = entry["tgt_dim"]  # 'tgt_dim' corresponds to active variables
    grouped_active_vars[nb_active].append(entry)

# Function to compute metric vs x-axis values with optional averaging
def compute_metric_vs_x(grouped_data, x_key, metric_key, average=False, num_points=10):
    x_vals = defaultdict(list)
    for nb_active, entries in grouped_data.items():
        for entry in entries:
            x_vals[entry[x_key]].append(entry[metric_key])
    
    if not average:  # No averaging for # active variables
        x_vals_sorted = sorted(x_vals.items())
        x, y = zip(*x_vals_sorted)
        x = list(x)
        y = [sum(scores) / len(scores) for scores in y]  # Average metric
        return x, y

    # Averaging for the other variables
    all_x = np.array([k for k in x_vals.keys()])
    min_x, max_x = min(all_x), max(all_x)
    interval_size = (max_x - min_x) / num_points

    averaged_x, averaged_y = [], []
    for i in range(num_points):
        interval_min = min_x + i * interval_size
        interval_max = interval_min + interval_size
        interval_y_values = [
            y for x, ys in x_vals.items() for y in ys if interval_min <= x < interval_max
        ]
        if interval_y_values:  # If there are points in the interval
            averaged_x.append((interval_min + interval_max) / 2)
            averaged_y.append(sum(interval_y_values) / len(interval_y_values))
    return averaged_x, averaged_y

# Plotting function
def plot_graph(x_key, x_label, metric_key, output_name, average=False, y_axis_label=False):
    plt.figure(figsize=(3, 9/4))
    for nb_active, entries in grouped_active_vars.items():
        x, y = compute_metric_vs_x(
            {nb_active: entries}, x_key, metric_key, average, num_points=num_points
        )
        plt.plot(x, y, marker="o")  # No legend for the curves
    if y_axis_label:
        plt.ylabel("Accuracy" if metric_key == "best_score" else "Perfect Recovery", fontsize=12)
    plt.xlabel(x_label, fontsize=12)
    plt.grid(True)
    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Graph saved as {output_path}")

# Generate graphs for all metrics including '_rdm' versions
metrics = [
    ("noisy_score", "Accuracy (Noisy)"),
    ("unnoised_score", "Accuracy"),
    ("perfect_recover", "Perfect Recovery"),
    ("score_rdm", "Accuracy (Randomized)"),
    ("perfect_recover_rdm", "Perfect Recovery (Randomized)"),
    ("f1_score", "F1 Score"), 
    ("f1_score_rdm", "F1 Score (Randomized)"),
    ("noisy_f1_score", "F1 Score (Noisy)")
]
x_keys = [
    ("tgt_dim", "# active variables", False, True),  # No averaging, show y-axis label
    ("nb_inactives", "# inactive variables", True, False),  # Averaging, no y-axis label
    ("nb_pts", "# input points", True, False),  # Averaging, no y-axis label
    ("prob_flip", "Flip Probability", True, False),  # Averaging, no y-axis label
]

for metric_key, metric_name in metrics:
    for x_key, x_label, average, y_axis_label in x_keys:
        output_name = f"{metric_key}_vs_{x_key}.png"
        plot_graph(x_key, x_label, metric_key, output_name, average=average, y_axis_label=y_axis_label)

def main():
    # Argument Parser Setup
    parser = ArgumentParser()
    # ... rest of your existing code ...

if __name__ == "__main__":
    main()
