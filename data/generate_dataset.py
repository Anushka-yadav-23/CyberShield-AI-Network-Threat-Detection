
import numpy as np
import pandas as pd
from pathlib import Path

# Reproducible synthetic network traffic data
rng = np.random.default_rng(42)

N = 2000
normal_count = 1400
attack_count = N - normal_count

# Generate normal traffic
normal = pd.DataFrame({
    "duration": rng.exponential(2, normal_count),
    "src_bytes": rng.integers(100, 5000, normal_count),
    "dst_bytes": rng.integers(100, 8000, normal_count),
    "failed_logins": rng.choice([0, 1], normal_count, p=[0.97, 0.03]),
    "connection_count": rng.integers(1, 20, normal_count),
    "unique_ports": rng.integers(1, 5, normal_count),
    "protocol": rng.choice(["TCP", "UDP", "ICMP"], normal_count,
                           p=[0.70, 0.25, 0.05]),
    "label": "Normal"
})

# Generate simulated suspicious traffic
attack_types = rng.choice(
    ["DoS", "Brute Force", "Port Scan"],
    attack_count,
    p=[0.40, 0.35, 0.25]
)

attacks = pd.DataFrame({
    "duration": rng.exponential(8, attack_count),
    "src_bytes": rng.integers(20, 1500, attack_count),
    "dst_bytes": rng.integers(0, 2500, attack_count),
    "failed_logins": np.where(
        attack_types == "Brute Force",
        rng.integers(5, 30, attack_count),
        rng.integers(0, 3, attack_count)
    ),
    "connection_count": np.where(
        attack_types == "DoS",
        rng.integers(80, 250, attack_count),
        rng.integers(10, 80, attack_count)
    ),
    "unique_ports": np.where(
        attack_types == "Port Scan",
        rng.integers(15, 60, attack_count),
        rng.integers(1, 10, attack_count)
    ),
    "protocol": rng.choice(["TCP", "UDP", "ICMP"], attack_count),
    "label": attack_types
})

# Combine and shuffle all records
df = pd.concat([normal, attacks], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save dataset relative to the project root
output_path = Path(__file__).resolve().parent / "network_traffic.csv"
df.to_csv(output_path, index=False)

print("Synthetic network dataset created successfully!")
print(f"Total records: {len(df)}")
print("\nTraffic class distribution:")
print(df["label"].value_counts())
print(f"\nDataset saved to: {output_path}")
print("\nFirst five rows:")
print(df.head().to_string(index=False))
