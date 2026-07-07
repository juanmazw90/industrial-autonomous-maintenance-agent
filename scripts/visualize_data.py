#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=1.26",
#   "pandas>=2.0",
#   "matplotlib>=3.8",
#   "seaborn>=0.13",
#   "pyarrow>=15.0",
# ]
# ///
"""
EDA plots for the synthetic AMIA dataset.

Usage:
    uv run scripts/visualize_data.py
    uv run scripts/visualize_data.py --input data/synthetic --output data/synthetic/plots
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PALETTE = {
    "normal": "#4CAF50",
    "bearing_wear": "#FF9800",
    "misalignment": "#2196F3",
    "electrical_failure": "#9C27B0",
    "overheating": "#F44336",
    "cavitation": "#00BCD4",
    "failure": "#212121",
}


def plot_machine_timeline(df: pd.DataFrame, machine_id: str, output_dir: Path) -> None:
    """Time-series of all sensors for one machine, colored by failure mode."""
    machine_df = df[df["machine_id"] == machine_id].copy()
    machine_df["timestamp"] = pd.to_datetime(machine_df["timestamp"])
    machine_df = machine_df.sort_values("timestamp")

    sensors = [
        ("vibration_rms", "Vibration RMS (mm/s)"),
        ("temperature_bearing", "Bearing Temp (°C)"),
        ("current_phase_a", "Current Phase A (A)"),
        ("speed_rpm", "Speed (RPM)"),
    ]

    fig, axes = plt.subplots(len(sensors), 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f"{machine_id} — Full Sensor Timeline", fontsize=14, fontweight="bold")

    for ax, (col, label) in zip(axes, sensors):
        for mode, color in PALETTE.items():
            mask = machine_df["failure_mode"] == mode
            if mask.any():
                ax.scatter(
                    machine_df.loc[mask, "timestamp"],
                    machine_df.loc[mask, col],
                    c=color,
                    s=1.5,
                    alpha=0.6,
                    label=mode,
                )
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()

    # Shared legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=m)
        for m, c in PALETTE.items()
        if m in machine_df["failure_mode"].values
    ]
    axes[0].legend(handles=handles, loc="upper right", fontsize=8, ncol=3)

    out = output_dir / f"{machine_id}_timeline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_failure_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    """Box plots of sensor distributions by failure mode."""
    # Exclude the 'failure' row itself for cleaner mode comparisons
    plot_df = df[df["failure_mode"] != "failure"].copy()

    sensors = [
        ("vibration_rms", "Vibration RMS (mm/s)"),
        ("temperature_bearing", "Bearing Temp (°C)"),
        ("current_phase_a", "Current Phase A (A)"),
        ("speed_rpm", "Speed (RPM)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Sensor Distributions by Failure Mode", fontsize=13, fontweight="bold")

    mode_order = ["normal", "bearing_wear", "misalignment", "electrical_failure", "overheating", "cavitation"]
    present_modes = [m for m in mode_order if m in plot_df["failure_mode"].unique()]

    for ax, (col, label) in zip(axes.flat, sensors):
        subset = plot_df[plot_df["failure_mode"].isin(present_modes)]
        sns.boxplot(
            data=subset,
            x="failure_mode",
            y=col,
            hue="failure_mode",
            order=present_modes,
            palette={m: PALETTE[m] for m in present_modes},
            ax=ax,
            fliersize=1.5,
            legend=False,
        )
        ax.set_xlabel("")
        ax.set_ylabel(label, fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    out = output_dir / "failure_mode_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_rul_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Histogram of RUL values for rows where RUL is labeled."""
    labeled = df[df["rul_hours"].notna()].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Remaining Useful Life (RUL) Distribution", fontsize=13, fontweight="bold")

    axes[0].hist(labeled["rul_hours"], bins=50, color="#1976D2", edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("RUL (hours)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("All Failure Modes")
    axes[0].grid(True, alpha=0.3)

    for mode in labeled["failure_mode"].unique():
        if mode == "failure":
            continue
        subset = labeled[labeled["failure_mode"] == mode]["rul_hours"]
        axes[1].hist(subset, bins=30, alpha=0.55, label=mode, color=PALETTE.get(mode, "gray"))

    axes[1].set_xlabel("RUL (hours)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("By Failure Mode")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    out = output_dir / "rul_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_bearing_wear_detail(df: pd.DataFrame, output_dir: Path) -> None:
    """Zoom into a single bearing wear event to show degradation curve."""
    bw_events = df[df["failure_mode"] == "bearing_wear"]["machine_id"].value_counts()
    if bw_events.empty:
        return
    machine_id = bw_events.index[0]
    machine_df = df[
        (df["machine_id"] == machine_id)
        & (df["failure_mode"].isin(["bearing_wear", "failure"]))
    ].copy()

    # Take first continuous bearing_wear block
    machine_df["timestamp"] = pd.to_datetime(machine_df["timestamp"])
    machine_df = machine_df.sort_values("timestamp").reset_index(drop=True)

    if len(machine_df) < 10:
        return

    # Add 100h of normal context before the degradation start
    first_ts = machine_df["timestamp"].iloc[0]
    context = df[
        (df["machine_id"] == machine_id)
        & (df["failure_mode"] == "normal")
        & (pd.to_datetime(df["timestamp"]) >= first_ts - pd.Timedelta(hours=100))
        & (pd.to_datetime(df["timestamp"]) < first_ts)
    ].copy()
    context["timestamp"] = pd.to_datetime(context["timestamp"])
    plot_data = pd.concat([context, machine_df]).sort_values("timestamp")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"Bearing Wear Degradation — {machine_id}", fontsize=13, fontweight="bold")

    sensors = [
        ("vibration_rms", "Vibration RMS (mm/s)", "#FF9800"),
        ("temperature_bearing", "Bearing Temp (°C)", "#F44336"),
        ("current_phase_a", "Current Phase A (A)", "#9C27B0"),
    ]
    for ax, (col, label, color) in zip(axes, sensors):
        ax.plot(plot_data["timestamp"], plot_data[col], lw=1.0, color=color)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)
        # Shade degradation region
        deg_start = machine_df["timestamp"].iloc[0]
        ax.axvspan(deg_start, plot_data["timestamp"].iloc[-1], alpha=0.08, color="red")
        ax.axvline(deg_start, color="red", lw=1.2, linestyle="--", alpha=0.6, label="Degradation start")

    axes[0].legend(fontsize=8)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate()

    out = output_dir / "bearing_wear_detail.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EDA plots for AMIA synthetic dataset")
    parser.add_argument("--input", default="data/synthetic", help="Directory with parquet file")
    parser.add_argument("--output", default="data/synthetic/plots", help="Output directory for plots")
    args = parser.parse_args()

    input_path = Path(args.input) / "sensor_readings.parquet"
    if not input_path.exists():
        print(f"Dataset not found at {input_path}")
        print("Run 'uv run scripts/generate_data.py' first.")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"  {len(df):,} rows, {df['machine_id'].nunique()} machines\n")

    print("Generating plots...")
    for machine_id in sorted(df["machine_id"].unique()):
        plot_machine_timeline(df, machine_id, output_dir)

    plot_failure_distributions(df, output_dir)
    plot_rul_distribution(df, output_dir)
    plot_bearing_wear_detail(df, output_dir)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
