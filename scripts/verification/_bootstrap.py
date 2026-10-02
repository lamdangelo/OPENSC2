"""Common bootstrap for the verification figure scripts: headless backend,
sys.path for the flat source_code imports (matching the editable install) and
for the shared test-side builders, and the output directory."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    REPOSITORY_ROOT / "source_code",
    REPOSITORY_ROOT / "tests" / "verification",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "output"


def save_figure(figure, stem):
    """Save a figure as PNG (quick look) and EPS (report), repo convention."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIRECTORY / f"{stem}.png", dpi=140)
    figure.savefig(OUTPUT_DIRECTORY / f"{stem}.eps", format="eps")
    print(f"saved {OUTPUT_DIRECTORY / stem}.png/.eps")
