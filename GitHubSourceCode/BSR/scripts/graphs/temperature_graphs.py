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
parser.add_argument("--num_points", type=int, default=10, help="Number of points to average for curves.")
args = parser.parse_args()

input_file = args.input
output_dir = args.output
num_points = args.num_points

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the JSON file
with open(input_file, "r") as f:
    data = json.load(f)

# Ensure data is a list
if not isinstance(data, list):
    raise TypeError("Expected a list of dictionaries in the JSON file.")

# Group data by temperature
grouped_by_temp = defaultdict(list)
for entry in data:
    temperature = entry["temperature"]
    grouped_by_temp[temperature].append(entry)

# Function to compute metric averages with optional averaging
def compute_metric_vs_temperature(grouped_data, metric_key, num_points):
    temp_vals = defaultdict(list)
    for temp, entries in grouped_data.items():
        for entry in entries:
            temp_vals[temp].append(entry[metric_key])
    
    temp_vals_sorted = sorted(temp_vals.items())
    x, y = zip(*temp_vals_sorted)
    x = list(x)
    y = [sum(scores) / len(scores) for scores in y]  # Average metric
    
    # Reduce number of points if necessary
    if len(x) > num_points:
        indices = np.linspace(0, len(x) - 1, num_points, dtype=int)
        x = [x[i] for i in indices]
        y = [y[i] for i in indices]
    
    return x, y

# Plot function
def plot_metric(metric_key, metric_label):
    plt.figure(figsize=(6, 4))
    x, y = compute_metric_vs_temperature(grouped_by_temp, metric_key, num_points)
    plt.plot(x, y, marker="o", label=metric_label)
    plt.xlabel("Temperature", fontsize=12)
    plt.ylabel(metric_label, fontsize=12)
    plt.title(f"{metric_label} vs Temperature")
    plt.grid(True)
    plt.legend()
    output_path = os.path.join(output_dir, f"temperature_vs_{metric_key}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Graph saved as {output_path}")

# Generate graphs for the specified metrics
metrics = [
    ("unnoised_score", "Unnoised Score"),
    ("noisy_score", "Noisy Score"),
    ("score_rdm", "Random Score"),
    ("f1_score", "F1 Score"),
    ("noisy_f1_score", "Noisy F1 Score"),
    ("f1_score_rdm", "F1 RDM Score"),
    ("perfect_recover", "Perfect Recovery"),
    ("perfect_recover_rdm", "Perfect Recovery (Random)"),
    ("nb_invalids", "Number of Invalids"),
    ("tgt_dim", "Dimension"),
]

for metric_key, metric_label in metrics:
    plot_metric(metric_key, metric_label)

def main():
    # Argument Parser Setup
    parser = ArgumentParser()
    # ... rest of your existing code ...

if __name__ == "__main__":
    main()