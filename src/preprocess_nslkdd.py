"""
preprocess_nslkdd.py

Step 3: clean and prepare the REAL NSL-KDD data (Kaggle package the user
downloaded and uploaded). This replaces the synthetic-data version of
preprocess.py from the earlier prototype phase.

What this does, in order:
  1. Attach the standard 41 NSL-KDD column names (the raw files have no header).
  2. Map the 39 specific attack names down to 4 recognized families
     (DoS, Probe, R2L, U2R) + Normal -> 5-class target.
  3. One-hot encode the 3 categorical columns (protocol_type, service, flag),
     aligning train/test columns so a category seen in one but not the
     other doesn't break anything.
  4. Scale numeric columns (StandardScaler, fit on train only).
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Paths are relative to the project root (the folder that contains src/ and data/).
# This works no matter where you put the project folder on your laptop.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "KDDTrain+.txt")
TEST_PATH = os.path.join(PROJECT_ROOT, "data", "KDDTest+.txt")

COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

# Standard NSL-KDD attack-family mapping (verified against this dataset's own
# label lists -- 11 DoS + 6 Probe + 15 R2L + 7 U2R = 39 attack types total,
# matching what's actually present across train+test).
ATTACK_FAMILY_MAP = {
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "mailbomb": "DoS", "processtable": "DoS", "udpstorm": "DoS",
    "apache2": "DoS", "worm": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L (remote-to-local: breaking in from outside)
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L",
    # U2R (user-to-root: escalating privileges once already inside)
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
    # Normal
    "normal": "Normal",
}


def load_raw():
    train = pd.read_csv(TRAIN_PATH, names=COLUMN_NAMES)
    test = pd.read_csv(TEST_PATH, names=COLUMN_NAMES)
    return train, test


def map_to_family(df):
    df = df.copy()
    unmapped = set(df["label"].unique()) - set(ATTACK_FAMILY_MAP.keys())
    if unmapped:
        raise ValueError(f"Found labels with no family mapping: {unmapped}")
    df["attack_family"] = df["label"].map(ATTACK_FAMILY_MAP)
    return df


def preprocess_nslkdd():
    train, test = load_raw()
    train = map_to_family(train)
    test = map_to_family(test)

    # Drop columns we don't feed to the model: original fine-grained label,
    # the difficulty score (not a real traffic feature), and num_outbound_cmds
    # (constant 0 in this dataset -- carries no signal).
    drop_cols = ["label", "difficulty", "num_outbound_cmds"]
    train_X_raw = train.drop(columns=drop_cols + ["attack_family"])
    test_X_raw = test.drop(columns=drop_cols + ["attack_family"])

    # One-hot encode categoricals, then align columns (in case train/test
    # ever see different categories -- didn't happen here, but this makes
    # the code safe regardless).
    train_X_encoded = pd.get_dummies(train_X_raw, columns=CATEGORICAL_COLUMNS)
    test_X_encoded = pd.get_dummies(test_X_raw, columns=CATEGORICAL_COLUMNS)
    train_X_encoded, test_X_encoded = train_X_encoded.align(
        test_X_encoded, join="left", axis=1, fill_value=0
    )

    # Scale numeric (non-dummy) columns; fit on train only to avoid leakage.
    original_numeric_cols = [c for c in train_X_raw.columns if c not in CATEGORICAL_COLUMNS]
    scaler = StandardScaler()
    train_X_encoded[original_numeric_cols] = scaler.fit_transform(train_X_encoded[original_numeric_cols])
    test_X_encoded[original_numeric_cols] = scaler.transform(test_X_encoded[original_numeric_cols])

    # Encode the 5-class target
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train["attack_family"])
    y_test = label_encoder.transform(test["attack_family"])

    artifacts = {
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_columns": list(train_X_encoded.columns),
    }

    return train_X_encoded, test_X_encoded, y_train, y_test, artifacts


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, artifacts = preprocess_nslkdd()

    print(f"Train shape: {X_train.shape}   Test shape: {X_test.shape}")
    print(f"Classes: {list(artifacts['label_encoder'].classes_)}")
    print()
    print("Train class distribution:")
    train_counts = pd.Series(y_train).value_counts().sort_index()
    for idx, count in train_counts.items():
        print(f"  {artifacts['label_encoder'].classes_[idx]:8s}: {count:6d}  ({count/len(y_train)*100:.1f}%)")
    print()
    print("Test class distribution:")
    test_counts = pd.Series(y_test).value_counts().sort_index()
    for idx, count in test_counts.items():
        print(f"  {artifacts['label_encoder'].classes_[idx]:8s}: {count:6d}  ({count/len(y_test)*100:.1f}%)")
    print()
    print("Sample of encoded features (first 3 rows, first 8 columns):")
    print(X_train.iloc[:3, :8])
