"""展示指定rep的统一进度曲线，使用与评分完全相同的数据。"""

import argparse
from pathlib import Path

from analyze_reps_3d import ROOT, analyze_data, load_angle_csv


def main():
    parser = argparse.ArgumentParser(description="Compare normalized left/right rep curves")
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--input", type=Path, default=ROOT / "lat_pulldown_2d_3d.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "rep_comparison.png")
    parser.add_argument("--show", action="store_true", help="Also show a window on your own computer")
    args = parser.parse_args()
    try:
        results, curves, _ = analyze_data(*load_angle_csv(args.input))
        if not 1 <= args.rep <= len(results):
            raise ValueError(f"Rep must be between 1 and {len(results)}; requested {args.rep}")
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from visualize_reps import make_rep_figure
        figure = make_rep_figure(results[args.rep - 1], [row for row in curves if row["rep"] == args.rep])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=150)
        print(f"Rep {args.rep} comparison saved: {args.output}")
        if args.show:
            plt.show()
        plt.close(figure)
    except (OSError, ValueError) as error:
        parser.exit(1, f"PoseCoach: {error}\n")


if __name__ == "__main__":
    main()
