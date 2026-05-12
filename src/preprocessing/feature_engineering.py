import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# ============================================================
# 1. Paths
# ============================================================

TRAIN_PATH = "data/raw/KDDTrain+.txt"
TEST_PATH = "data/raw/KDDTest+.txt"

PROCESSED_DIR = "data/processed"
VISUALIZATION_DIR = "results/visualizations"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)


# ============================================================
# 2. NSL-KDD columns
# ============================================================

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]


# ============================================================
# 3. Attack mapping: detailed labels -> 5 IDS classes
# ============================================================

DOS_ATTACKS = {
    "back", "land", "neptune", "pod", "smurf", "teardrop",
    "apache2", "udpstorm", "processtable", "mailbomb"
}

PROBE_ATTACKS = {
    "ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"
}

R2L_ATTACKS = {
    "ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
    "warezclient", "warezmaster", "sendmail", "named",
    "snmpgetattack", "snmpguess", "xlock", "xsnoop",
    "httptunnel", "worm"
}

U2R_ATTACKS = {
    "buffer_overflow", "loadmodule", "perl", "rootkit",
    "ps", "sqlattack", "xterm"
}


def map_attack_to_category(label):
    """
    Convert NSL-KDD detailed attack labels into 5 main IDS classes:
    normal, DoS, Probe, R2L, U2R.
    """
    label = str(label).strip()

    if label == "normal":
        return "normal"
    if label in DOS_ATTACKS:
        return "DoS"
    if label in PROBE_ATTACKS:
        return "Probe"
    if label in R2L_ATTACKS:
        return "R2L"
    if label in U2R_ATTACKS:
        return "U2R"

    raise ValueError(f"Unknown attack label found: {label}")


# ============================================================
# 4. Sequence creation for LSTM
# ============================================================

def create_sequences(X, y, seq_length=10):
    """
    Create temporal sequences for LSTM.

    X shape before: (samples, features)
    X shape after:  (samples - seq_length, seq_length, features)
    """
    X_seq = []
    y_seq = []

    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length - 1])

    return np.array(X_seq), np.array(y_seq)


# ============================================================
# 5. Main preprocessing pipeline
# ============================================================

def main():
    print("=" * 60)
    print("NSL-KDD Feature Engineering Pipeline")
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train_df = pd.read_csv(TRAIN_PATH, names=COLUMNS)
    test_df = pd.read_csv(TEST_PATH, names=COLUMNS)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    # --------------------------------------------------------
    # Convert detailed labels to 5 classes
    # --------------------------------------------------------

    train_df["category"] = train_df["label"].apply(map_attack_to_category)
    test_df["category"] = test_df["label"].apply(map_attack_to_category)

    print("\nClass distribution - Train:")
    print(train_df["category"].value_counts())

    print("\nClass distribution - Test:")
    print(test_df["category"].value_counts())

    # --------------------------------------------------------
    # Encode categorical features
    # --------------------------------------------------------

    categorical_cols = ["protocol_type", "service", "flag"]

    for col in categorical_cols:
        encoder = LabelEncoder()

        combined_values = pd.concat([train_df[col], test_df[col]], axis=0)
        encoder.fit(combined_values)

        train_df[col] = encoder.transform(train_df[col])
        test_df[col] = encoder.transform(test_df[col])

    print("\nCategorical features encoded successfully.")

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    for df in [train_df, test_df]:
        df["byte_rate"] = (df["src_bytes"] + df["dst_bytes"]) / (df["duration"] + 1)
        df["total_bytes"] = df["src_bytes"] + df["dst_bytes"]
        df["byte_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)

    print("Additional engineered features created.")

    # --------------------------------------------------------
    # Separate features and labels
    # --------------------------------------------------------

    X_train_df = train_df.drop(["label", "difficulty", "category"], axis=1)
    X_test_df = test_df.drop(["label", "difficulty", "category"], axis=1)

    y_train_text = train_df["category"].values
    y_test_text = test_df["category"].values

    feature_names = np.array(X_train_df.columns)

    # --------------------------------------------------------
    # Encode 5 classes
    # --------------------------------------------------------

    label_encoder = LabelEncoder()
    label_encoder.fit(["normal", "DoS", "Probe", "R2L", "U2R"])

    y_train = label_encoder.transform(y_train_text)
    y_test = label_encoder.transform(y_test_text)

    class_names = label_encoder.classes_

    print("\nClass encoding:")
    for cls, idx in zip(class_names, label_encoder.transform(class_names)):
        print(f"  {cls} -> {idx}")

    # --------------------------------------------------------
    # Normalize features
    # --------------------------------------------------------

    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)

    print("\nNormalization done.")
    print(f"X_train_scaled shape: {X_train_scaled.shape}")
    print(f"X_test_scaled shape:  {X_test_scaled.shape}")
    print(f"Min value: {np.min(X_train_scaled):.4f}")
    print(f"Max value: {np.max(X_train_scaled):.4f}")

    # --------------------------------------------------------
    # Create LSTM sequences
    # --------------------------------------------------------

    SEQ_LENGTH = 5

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, SEQ_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, SEQ_LENGTH)

    print("\nSequence creation done.")
    print(f"X_train_seq shape: {X_train_seq.shape}")
    print(f"y_train_seq shape: {y_train_seq.shape}")
    print(f"X_test_seq shape:  {X_test_seq.shape}")
    print(f"y_test_seq shape:  {y_test_seq.shape}")

    # --------------------------------------------------------
    # Save processed files
    # --------------------------------------------------------

    np.save(os.path.join(PROCESSED_DIR, "X_train_scaled.npy"), X_train_scaled)
    np.save(os.path.join(PROCESSED_DIR, "X_test_scaled.npy"), X_test_scaled)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

    np.save(os.path.join(PROCESSED_DIR, "X_train_seq.npy"), X_train_seq)
    np.save(os.path.join(PROCESSED_DIR, "X_test_seq.npy"), X_test_seq)
    np.save(os.path.join(PROCESSED_DIR, "y_train_seq.npy"), y_train_seq)
    np.save(os.path.join(PROCESSED_DIR, "y_test_seq.npy"), y_test_seq)

    np.save(os.path.join(PROCESSED_DIR, "feature_names.npy"), feature_names)
    np.save(os.path.join(PROCESSED_DIR, "class_names.npy"), class_names)

    print("\nProcessed files saved successfully in data/processed/")
    print("Saved files:")
    print("  - X_train_scaled.npy")
    print("  - X_test_scaled.npy")
    print("  - y_train.npy")
    print("  - y_test.npy")
    print("  - X_train_seq.npy")
    print("  - X_test_seq.npy")
    print("  - y_train_seq.npy")
    print("  - y_test_seq.npy")
    print("  - feature_names.npy")
    print("  - class_names.npy")

    print("\nFeature engineering pipeline completed successfully.")


if __name__ == "__main__":
    main()
