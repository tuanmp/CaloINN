from pathlib import Path

import matplotlib.pyplot as plt

SAVE_DIR = Path(__file__).resolve().parent.parent.parent / "attachements"


def save_and_show(fig, filename):
    fig.tight_layout()
    fig.savefig(f"{SAVE_DIR}/{filename}", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {filename}")
