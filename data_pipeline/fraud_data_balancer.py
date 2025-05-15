# data_pipeline/fraud_data_balancer.py

import logging
from typing import Optional, List

import pandas as pd
import yaml
import os

from models.gan_synthesizer import GANSynthesizer

# --- Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Load GAN config ---
GAN_CONFIG_PATH = "configs/gan_config.yaml"
if not os.path.exists(GAN_CONFIG_PATH):
    logger.error(f"Missing GAN config: {GAN_CONFIG_PATH}")
    _GAN_CFG = {}
else:
    with open(GAN_CONFIG_PATH, "r") as cfg_file:
        _GAN_CFG = yaml.safe_load(cfg_file)


def balance_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "default_flag",
    target_ratio: float = 0.5,
    random_state: Optional[int] = 42
) -> pd.DataFrame:
    """
    Balance binary dataset using a GAN to synthesize minority-class samples.

    Parameters:
        df (pd.DataFrame): Dataset including both features and labels
        feature_cols (List[str]): Feature column names for GAN input
        label_col (str): Binary label column (minority = 1)
        target_ratio (float): Desired minority class ratio
        random_state (int): Reproducibility seed

    Returns:
        pd.DataFrame: Original + synthetic samples
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame.")

    df_major = df[df[label_col] == 0].reset_index(drop=True)
    df_minor = df[df[label_col] == 1].reset_index(drop=True)

    n_major, n_minor = len(df_major), len(df_minor)
    total = n_major + n_minor
    current_ratio = n_minor / max(total, 1)

    logger.info(f"📊 Current minority ratio: {current_ratio:.3f} ({n_minor}/{total})")

    if current_ratio >= target_ratio:
        logger.info("✔️ Target ratio already satisfied. No GAN needed.")
        return df

    if df_minor.empty or not feature_cols:
        logger.warning("⚠️ No minority samples or empty feature list; skipping GAN.")
        return df

    desired_n_minor = int(target_ratio * total / (1 - target_ratio))
    n_synth = desired_n_minor - n_minor

    logger.info(f"🔁 Generating {n_synth} synthetic samples using GAN...")

    gan_params = _GAN_CFG.get("gan_params", {})
    gan = GANSynthesizer(**gan_params, random_state=random_state)

    try:
        X_train = df_minor[feature_cols]
        gan.fit(X_train)

        synth_X = gan.sample(n_synth)
        synth_df = pd.DataFrame(synth_X, columns=feature_cols)
        synth_df[label_col] = 1

        df_final = pd.concat([df, synth_df], ignore_index=True)

        new_ratio = len(df_final[df_final[label_col] == 1]) / len(df_final)
        logger.info(f"✅ Final minority ratio: {new_ratio:.3f} ({len(df_final[df_final[label_col] == 1])}/{len(df_final)})")

        return df_final

    except Exception as e:
        logger.error(f"❌ GAN balancing failed: {e}")
        return df


# --- CLI / test block ---
if __name__ == "__main__":
    from data_pipeline.generate_training_dataset import generate_training_dataset

    logger.info("🧪 Running test balance pipeline...")

    raw_data = generate_training_dataset(limit=500)
    if isinstance(raw_data, list):
        raw_df = pd.DataFrame(raw_data)
    else:
        raise TypeError("generate_training_dataset should return a list of dicts.")

    if "default_flag" not in raw_df.columns:
        raw_df["default_flag"] = raw_df["fraud_label"]  # fallback if needed

    feature_cols = [
        col for col in raw_df.columns
        if col not in {"pinfl", "default_flag", "fraud_label", "risk_label", "loan_limit_label", "lgd_label"}
    ]

    balansed_df = balance_dataset(raw_df, feature_cols, label_col="default_flag", target_ratio=0.5)
    print("✅ Final balanced dataset shape:", balanced_df.shape)
