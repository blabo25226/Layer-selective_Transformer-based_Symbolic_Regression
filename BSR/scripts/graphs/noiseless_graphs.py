import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from argparse import ArgumentParser

# Argument Parser Setup
parser = ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Input JSON file path.")
parser.add_argument("--output", type=str, required=True, help="Output directory for saving plots.")
parser.add_argument("--num_points", type=int, required=True, help="Number of points to average for curves.")
parser.add_argument("--ft_size", type=int, default=14, help="Font size.")
args = parser.parse_args()

input_file = args.input
output = args.output
num_points = args.num_points
size = args.ft_size

# Ensure the output directory exists
os.makedirs(output, exist_ok=True)

# Load the JSON file
with open(input_file, "r") as f:
    data = json.load(f)

# Ensure all scores are adjusted to max(0, best_score)
for entry in data:
    entry["best_score"] = max(0, entry["best_score"])

### Helper Function: Average Along X-Axis
def average_along_x(x_vals, y_vals, num_points):
    min_x, max_x = min(x_vals), max(x_vals)
    interval_size = (max_x - min_x) / num_points
    averaged_x, averaged_y = [], []
    for i in range(num_points):
        interval_min = min_x + i * interval_size
        interval_max = interval_min + interval_size
        interval_indices = [j for j, x in enumerate(x_vals) if interval_min <= x < interval_max]
        if interval_indices:
            averaged_x.append((interval_min + interval_max) / 2)
            averaged_y.append(np.mean([y_vals[j] for j in interval_indices]))
    return averaged_x, averaged_y

### Process data for Binary Operators vs Perfect Recovery and Scores
grouped_bin_op_data = defaultdict(list)
grouped_bin_op_scores = defaultdict(list)

for entry in data:
    nb_bin_op = entry["nb_bin_op"]
    perfect_recover = entry["perfect_recover"]
    score = entry["best_score"]
    grouped_bin_op_data[nb_bin_op].append(perfect_recover)
    grouped_bin_op_scores[nb_bin_op].append(score)

# Flatten data for averaging
nb_bin_op_vals = sorted(grouped_bin_op_data.keys())
perfect_recovery_flat = [
    (op, val) for op in nb_bin_op_vals for val in grouped_bin_op_data[op]
]
score_flat = [
    (op, val) for op in nb_bin_op_vals for val in grouped_bin_op_scores[op]
]

# Average along x-axis for both Perfect Recovery and Scores
x_vals_recovery, y_vals_recovery = zip(*perfect_recovery_flat)
x_vals_score, y_vals_score = zip(*score_flat)
avg_bin_op_vals_recovery, avg_recovery = average_along_x(x_vals_recovery, y_vals_recovery, num_points)
avg_bin_op_vals_score, avg_score = average_along_x(x_vals_score, y_vals_score, num_points)

# Plot Binary Operators vs Perfect Recovery and Scores
plt.figure(figsize=(15/5, 9/4))  # (5, 3) * 3/5
plt.plot(avg_bin_op_vals_recovery, avg_recovery, marker="o", label="Perfect Recovery")
plt.plot(avg_bin_op_vals_score, avg_score, marker="x", label="Accuracy")
plt.xlabel("# binary operators", fontsize=size)
plt.ylabel("Mean Value", fontsize=size)
plt.grid(True)
plt.legend(fontsize=int(size * 0.8))
output_bin_op = os.path.join(output, "binary_operators_vs_metrics.png")
plt.savefig(output_bin_op, dpi=300, bbox_inches="tight")
plt.show()
print(f"Graph saved as {output_bin_op}")

### Process data for Number of Formulas vs Binary Operators
bin_op_count = Counter(entry["nb_bin_op"] for entry in data)  # Count formulas per nb_bin_op
nb_bin_op_vals_count = sorted(bin_op_count.keys())
formula_counts = [bin_op_count[op] for op in nb_bin_op_vals_count]

# Plot Number of Formulas vs Binary Operators
plt.figure(figsize=(3, 9/4))
plt.bar(nb_bin_op_vals_count, formula_counts, width=1., label="Number of Formulas")
plt.xlabel("Number of binary operators", fontsize=size)
plt.grid(axis="y")
# plt.legend(fontsize=size)
output_bin_op_count = os.path.join(output, "formulas_vs_binary_operators.png")
plt.savefig(output_bin_op_count, dpi=300, bbox_inches="tight")
plt.show()
print(f"Graph saved as {output_bin_op_count}")

### Process data for Number of Formulas vs Total Operators (nb_bin_op + nb_unary_op)
total_op_count = Counter((entry["nb_bin_op"] + entry["nb_unary_op"]) for entry in data)  # Count formulas per total operators
total_op_vals = sorted(total_op_count.keys())
formula_counts_total_op = [total_op_count[op] for op in total_op_vals]

# Plot Number of Formulas vs Total Operators
plt.figure(figsize=(3, 9/4))
plt.bar(total_op_vals, formula_counts_total_op, width=1., label="Number of Formulas")
plt.xlabel("Total number of operators", fontsize=size)
plt.grid(axis="y")
# plt.legend(fontsize=size)
output_total_op_count = os.path.join(output, "formulas_vs_total_operators.png")
plt.savefig(output_total_op_count, dpi=300, bbox_inches="tight")
plt.show()
print(f"Graph saved as {output_total_op_count}")

### Process data for Target Dimension vs Perfect Recovery and Scores
grouped_tgt_dim_data = defaultdict(list)
grouped_tgt_dim_scores = defaultdict(list)

for entry in data:
    tgt_dim = entry["tgt_dim"]
    perfect_recover = entry["perfect_recover"]
    score = entry["best_score"]
    grouped_tgt_dim_data[tgt_dim].append(perfect_recover)
    grouped_tgt_dim_scores[tgt_dim].append(score)

tgt_dim_vals = sorted(grouped_tgt_dim_data.keys())
mean_perfect_recovery_tgt_dim = [
    sum(grouped_tgt_dim_data[dim]) / len(grouped_tgt_dim_data[dim]) for dim in tgt_dim_vals
]
mean_scores_tgt_dim = [
    sum(grouped_tgt_dim_scores[dim]) / len(grouped_tgt_dim_scores[dim]) for dim in tgt_dim_vals
]

# Plot Target Dimension vs Perfect Recovery and Scores
plt.figure(figsize=(3, 9/4))
plt.plot(tgt_dim_vals, mean_perfect_recovery_tgt_dim, marker="o", label="Perfect Recovery")
plt.plot(tgt_dim_vals, mean_scores_tgt_dim, marker="x", label="Accuracy")
plt.xlabel("# active variables", fontsize=size)
plt.grid(True)
# plt.legend(fontsize=int(size * 0.8))
output_tgt_dim = os.path.join(output, "active_var_vs_metrics.png")
plt.savefig(output_tgt_dim, dpi=300, bbox_inches="tight")
plt.show()
print(f"Graph saved as {output_tgt_dim}")

def main():
    # Argument Parser Setup
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for saving plots.")
    parser.add_argument("--num_points", type=int, required=True, help="Number of points to average for curves.")
    parser.add_argument("--ft_size", type=int, default=14, help="Font size.")
    args = parser.parse_args()

    input_file = args.input
    output = args.output
    num_points = args.num_points
    size = args.ft_size

    # Ensure the output directory exists
    os.makedirs(output, exist_ok=True)

    # Load the JSON file
    with open(input_file, "r") as f:
        data = json.load(f)

    # Ensure all scores are adjusted to max(0, best_score)
    for entry in data:
        entry["best_score"] = max(0, entry["best_score"])

    ### Helper Function: Average Along X-Axis
    def average_along_x(x_vals, y_vals, num_points):
        min_x, max_x = min(x_vals), max(x_vals)
        interval_size = (max_x - min_x) / num_points
        averaged_x, averaged_y = [], []
        for i in range(num_points):
            interval_min = min_x + i * interval_size
            interval_max = interval_min + interval_size
            interval_indices = [j for j, x in enumerate(x_vals) if interval_min <= x < interval_max]
            if interval_indices:
                averaged_x.append((interval_min + interval_max) / 2)
                averaged_y.append(np.mean([y_vals[j] for j in interval_indices]))
        return averaged_x, averaged_y

    ### Process data for Binary Operators vs Perfect Recovery and Scores
    grouped_bin_op_data = defaultdict(list)
    grouped_bin_op_scores = defaultdict(list)

    for entry in data:
        nb_bin_op = entry["nb_bin_op"]
        perfect_recover = entry["perfect_recover"]
        score = entry["best_score"]
        grouped_bin_op_data[nb_bin_op].append(perfect_recover)
        grouped_bin_op_scores[nb_bin_op].append(score)

    # Flatten data for averaging
    nb_bin_op_vals = sorted(grouped_bin_op_data.keys())
    perfect_recovery_flat = [
        (op, val) for op in nb_bin_op_vals for val in grouped_bin_op_data[op]
    ]
    score_flat = [
        (op, val) for op in nb_bin_op_vals for val in grouped_bin_op_scores[op]
    ]

    # Average along x-axis for both Perfect Recovery and Scores
    x_vals_recovery, y_vals_recovery = zip(*perfect_recovery_flat)
    x_vals_score, y_vals_score = zip(*score_flat)
    avg_bin_op_vals_recovery, avg_recovery = average_along_x(x_vals_recovery, y_vals_recovery, num_points)
    avg_bin_op_vals_score, avg_score = average_along_x(x_vals_score, y_vals_score, num_points)

    # Plot Binary Operators vs Perfect Recovery and Scores
    plt.figure(figsize=(15/5, 9/4))  # (5, 3) * 3/5
    plt.plot(avg_bin_op_vals_recovery, avg_recovery, marker="o", label="Perfect Recovery")
    plt.plot(avg_bin_op_vals_score, avg_score, marker="x", label="Accuracy")
    plt.xlabel("# binary operators", fontsize=size)
    plt.ylabel("Mean Value", fontsize=size)
    plt.grid(True)
    plt.legend(fontsize=int(size * 0.8))
    output_bin_op = os.path.join(output, "binary_operators_vs_metrics.png")
    plt.savefig(output_bin_op, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Graph saved as {output_bin_op}")

    ### Process data for Number of Formulas vs Binary Operators
    bin_op_count = Counter(entry["nb_bin_op"] for entry in data)  # Count formulas per nb_bin_op
    nb_bin_op_vals_count = sorted(bin_op_count.keys())
    formula_counts = [bin_op_count[op] for op in nb_bin_op_vals_count]

    # Plot Number of Formulas vs Binary Operators
    plt.figure(figsize=(3, 9/4))
    plt.bar(nb_bin_op_vals_count, formula_counts, width=1., label="Number of Formulas")
    plt.xlabel("Number of binary operators", fontsize=size)
    plt.grid(axis="y")
    # plt.legend(fontsize=size)
    output_bin_op_count = os.path.join(output, "formulas_vs_binary_operators.png")
    plt.savefig(output_bin_op_count, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Graph saved as {output_bin_op_count}")

    ### Process data for Number of Formulas vs Total Operators (nb_bin_op + nb_unary_op)
    total_op_count = Counter((entry["nb_bin_op"] + entry["nb_unary_op"]) for entry in data)  # Count formulas per total operators
    total_op_vals = sorted(total_op_count.keys())
    formula_counts_total_op = [total_op_count[op] for op in total_op_vals]

    # Plot Number of Formulas vs Total Operators
    plt.figure(figsize=(3, 9/4))
    plt.bar(total_op_vals, formula_counts_total_op, width=1., label="Number of Formulas")
    plt.xlabel("Total number of operators", fontsize=size)
    plt.grid(axis="y")
    # plt.legend(fontsize=size)
    output_total_op_count = os.path.join(output, "formulas_vs_total_operators.png")
    plt.savefig(output_total_op_count, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Graph saved as {output_total_op_count}")

    ### Process data for Target Dimension vs Perfect Recovery and Scores
    grouped_tgt_dim_data = defaultdict(list)
    grouped_tgt_dim_scores = defaultdict(list)

    for entry in data:
        tgt_dim = entry["tgt_dim"]
        perfect_recover = entry["perfect_recover"]
        score = entry["best_score"]
        grouped_tgt_dim_data[tgt_dim].append(perfect_recover)
        grouped_tgt_dim_scores[tgt_dim].append(score)

    tgt_dim_vals = sorted(grouped_tgt_dim_data.keys())
    mean_perfect_recovery_tgt_dim = [
        sum(grouped_tgt_dim_data[dim]) / len(grouped_tgt_dim_data[dim]) for dim in tgt_dim_vals
    ]
    mean_scores_tgt_dim = [
        sum(grouped_tgt_dim_scores[dim]) / len(grouped_tgt_dim_scores[dim]) for dim in tgt_dim_vals
    ]

    # Plot Target Dimension vs Perfect Recovery and Scores
    plt.figure(figsize=(3, 9/4))
    plt.plot(tgt_dim_vals, mean_perfect_recovery_tgt_dim, marker="o", label="Perfect Recovery")
    plt.plot(tgt_dim_vals, mean_scores_tgt_dim, marker="x", label="Accuracy")
    plt.xlabel("# active variables", fontsize=size)
    plt.grid(True)
    # plt.legend(fontsize=int(size * 0.8))
    output_tgt_dim = os.path.join(output, "active_var_vs_metrics.png")
    plt.savefig(output_tgt_dim, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Graph saved as {output_tgt_dim}")

if __name__ == "__main__":
    main()


