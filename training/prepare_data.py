"""Create the demo dataset versions (v1, v2). Run before `dvc add`. Deterministic."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


def build(version: str) -> pd.DataFrame:
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.rename(columns=lambda c: c.replace(" ", "_"))
    if version == "v1":
        return df
    if version == "v2":  # more data: +250 jittered resamples (simulates newly collected data)
        rng = np.random.default_rng(7)
        extra = df.sample(250, replace=True, random_state=7).copy()
        feats = extra.columns.drop("target")
        extra[feats] = extra[feats] * rng.normal(1.0, 0.03, size=(len(extra), len(feats)))
        return pd.concat([df, extra], ignore_index=True)
    raise SystemExit(f"unknown version {version}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1", choices=["v1", "v2"])
    ap.add_argument("--out", default="data/dataset.csv")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    build(a.version).to_csv(a.out, index=False)
    print(f"wrote {a.out} ({a.version})")
