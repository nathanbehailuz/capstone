"""Minimal, reproducible plots from Sileo's structured artifacts."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def plot_run(run_dir):
    run_dir = Path(run_dir)
    metrics = read_jsonl(run_dir / "training_metrics.jsonl")
    explanations = read_jsonl(run_dir / "explanations.jsonl")
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    if not metrics:
        return

    epochs = [row["epoch"] for row in metrics]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(epochs, [row["reward"] for row in metrics])
    axes[0, 0].set_title("Raw action reward")
    axes[0, 1].plot(epochs, [row.get("approx_kl", 0) for row in metrics])
    axes[0, 1].plot(epochs, [row.get("clip_frac", 0) for row in metrics])
    axes[0, 1].set_title("KL and clip fraction")
    axes[1, 0].scatter(
        [row["size"] for row in explanations],
        [row["selected_confidence"] for row in explanations],
        s=10,
    )
    axes[1, 0].set_title("Fidelity versus size")
    axes[1, 1].plot(
        epochs, [row.get("stop_rate", 0) for row in metrics], label="STOP"
    )
    axes[1, 1].plot(
        epochs,
        [row.get("exhaustion_rate", 0) for row in metrics],
        label="exhausted",
    )
    axes[1, 1].legend()
    axes[1, 1].set_title("Termination rates")
    figure.tight_layout()
    figure.savefig(plot_dir / "sileo_v1_validation.png", dpi=160)

    for row in explanations[:4]:
        graph = nx.Graph()
        graph.add_nodes_from(row["selected_nodes"])
        graph.add_edges_from(tuple(edge) for edge in row.get("selected_edges", []))
        figure, axis = plt.subplots(figsize=(4, 4))
        layout = nx.spring_layout(graph, seed=0)
        nx.draw_networkx(graph, layout, ax=axis, with_labels=True, node_size=300)
        axis.set_title(f"instance {row['instance_id']}")
        axis.axis("off")
        figure.tight_layout()
        figure.savefig(
            plot_dir / f"explanation_{row['instance_id']}.png", dpi=160
        )
        plt.close(figure)
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    plot_run(parser.parse_args().run_dir)
