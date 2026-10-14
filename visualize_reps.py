"""简单的离线图表；保存PNG，不需要摄像头或图形界面。"""

from pathlib import Path


def make_rep_figure(result, curves):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    progress = [row["progress_percent"] for row in curves]
    for side, color in (("left", "tab:blue"), ("right", "tab:orange")):
        axes[0].plot(progress, [row[f"{side}_angle"] for row in curves], label=side.title(), color=color)
        normalized = [row[f"{side}_normalized"] for row in curves]
        if all(value is not None for value in normalized):
            axes[1].plot(progress, normalized, label=side.title(), color=color)
    bottom = 100 * (result["bottom_time"] - result["start_time"]) / (result["end_time"] - result["start_time"])
    for axis in axes:
        axis.axvline(bottom, color="grey", linestyle="--", label="Combined bottom")
        axis.set_xlabel("Rep progress (%)")
        axis.set_xlim(0, 100)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Smoothed elbow angle (degrees)")
    axes[1].set_ylabel("Normalized angle (0 = min, 1 = max)")
    axes[1].set_ylim(-0.05, 1.05)
    score = result["quality_score"]
    if score is None:
        subtitle = f"Insufficient data: {result['data_warning']}"
    else:
        subtitle = (f"Sync: {result['synchronization_score']:.1f} | "
                    f"Pull: {result['pull_synchronization_score']:.1f} | "
                    f"Return: {result['return_synchronization_score']:.1f} | Quality: {score:.1f}\n"
                    f"{result['asymmetry_result']}: {result['asymmetry_reasons'] or 'no threshold exceeded'}")
    figure.suptitle(f"Rep {result['rep']} — {subtitle}", fontsize=10)
    figure.tight_layout()
    return figure


def save_visualizations(results, curves, motion, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    times = motion["times"]
    axes[0].plot(times, motion["left_smooth"], label="Left 3D")
    axes[0].plot(times, motion["right_smooth"], label="Right 3D")
    axes[0].set_ylabel("Elbow angle (degrees)")
    axes[1].plot(times, motion["combined"], label="Combined motion")
    axes[1].axhline(0.7, linestyle="--", color="grey", label="Top threshold")
    axes[1].axhline(0.3, linestyle=":", color="grey", label="Bottom threshold")
    axes[1].set_ylabel("Session normalized motion")
    axes[1].set_xlabel("Video time (s)")
    for row in results:
        color = "tab:green" if row["asymmetry_result"] == "symmetric" else "tab:red"
        for axis in axes:
            axis.axvspan(row["start_time"], row["end_time"], color=color, alpha=0.08)
        axes[1].text(row["bottom_time"], 0.04, str(row["rep"]), ha="center", fontsize=8)
    for axis in axes:
        axis.legend(loc="upper right")
        axis.grid(alpha=0.2)
    figure.suptitle("PoseCoach — rep segmentation (red shading = review suggested)")
    figure.tight_layout()
    figure.savefig(output_dir / "angle_timeline.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 5))
    reps = [row["rep"] for row in results]
    for field, label in (("quality_score", "Quality"), ("synchronization_score", "Curve synchronization"),
                         ("rom_balance_score", "ROM balance"), ("bottom_timing_score", "Bottom timing")):
        axis.plot(reps, [float("nan") if row[field] is None else row[field] for row in results],
                  marker="o", label=label, linewidth=1.3)
    for row in results:
        if row["asymmetry_result"] == "asymmetric":
            axis.text(row["rep"], 103, "!", ha="center", color="tab:red")
        elif row["data_status"] != "ok":
            axis.text(row["rep"], 50, "N/A", ha="center", color="grey")
    axis.set(ylim=(0, 110), xlabel="Rep", ylabel="Score (0–100)",
             title="Rep scores — ! = asymmetry flag; demo heuristics, not validated grading")
    axis.set_xticks(reps)
    axis.grid(alpha=0.2)
    axis.legend(loc="lower left")
    if not results:
        axis.text(0.5, 0.5, "No complete reps detected", transform=axis.transAxes, ha="center")
    figure.tight_layout()
    figure.savefig(output_dir / "rep_scores.png", dpi=150)
    plt.close(figure)

    curve_groups = {}
    for row in curves:
        curve_groups.setdefault(row["rep"], []).append(row)
    rep_dir = output_dir / "reps"
    rep_dir.mkdir(exist_ok=True)
    # 重跑时清理本工具旧的单次动作图，避免残留上一次较多的rep。
    for old_plot in rep_dir.glob("rep_[0-9][0-9][0-9].png"):
        old_plot.unlink()
    for result in results:
        figure = make_rep_figure(result, curve_groups[result["rep"]])
        figure.savefig(rep_dir / f"rep_{result['rep']:03}.png", dpi=150)
        plt.close(figure)
    print(f"图表已保存: {output_dir}")
