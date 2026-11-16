"""Shared configuration: paths, seeds and run switches (edit here or override in the notebook)."""
from pathlib import Path

SEED = 42
ROOT = Path.cwd()
DATA_RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"
DB_PATH = PROCESSED / "porto.db"
for _p in (DATA_RAW, PROCESSED, MODELS, REPORTS, FIGURES, REPORTS / "sql"):
    _p.mkdir(parents=True, exist_ok=True)

# ---- run switches ---------------------------------------------------------
ZIP_NAME = "porto-seguro-safe-driver-prediction.zip"  # the Kaggle download you upload in Colab
MANUAL_ZIP_PATH = None  # optional explicit path to the zip, e.g. Path("/content/drive/MyDrive/porto.zip")
TEST_SIZE = 0.20        # stratified hold-out share
CV_FOLDS = 3            # folds used for tuning / ablation inside the training part
N_TRIALS_LGBM = 20      # Optuna trials (more = better, slower)
N_TRIALS_XGB = 12
RUN_EXTERNAL = True     # Part 3: external dataset (freMTPL2freq from OpenML)
MAKE_SUBMISSION = True  # Part 3: refit on all data and write submission.csv (needs test.csv)
RUN_INTERACTIVE = True  # Part 3: True opens the live console menu (uses input())


def save_fig(name: str, dpi: int = 150) -> None:
    """Save the current matplotlib figure into figures/."""
    import matplotlib.pyplot as plt
    plt.savefig(FIGURES / f"{name}.png", dpi=dpi, bbox_inches="tight")
