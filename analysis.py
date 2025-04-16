import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths
files = {
    "Default": "metric.csv",
    "Handmade": "metric_custom_1.csv",
    "Transition-based": "metric_transit.csv",
    "Potential-based": "metric_pot.csv"
}

# --- Plot Reward, Episode Length, and Q-value ---
columns = ["length", "total_reward", "avg_q"]
for col in columns:
    plt.figure(figsize=(12, 6))
    for label, path in files.items():
        df = pd.read_csv(path)
        plt.plot(df[col], label=label)
    plt.xlabel("Episode")
    plt.ylabel(col)
    plt.title(f"{col.capitalize()} Comparison across Methods")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

# --- Soft Landing Plot and Summary Collection ---
summary_data = []
plt.figure(figsize=(10, 5))
for label, path in files.items():
    df = pd.read_csv(path)

    soft_landing_flags = df["soft_land"].astype(int)
    landing_flags = df["land"].astype(int)

    rolling_soft = soft_landing_flags.rolling(window=50).mean()

    # Reward normalization and stat collection
    rewards = df["total_reward"].values
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    rewards_normalized = (rewards - mean_reward) / std_reward
    last_200_rewards = rewards_normalized[-200:]
    mean_last_200 = last_200_rewards.mean()
    std__last_200 = last_200_rewards.std()

    summary_data.append({
        "Method": label,
        "Mean": mean_last_200,
        "Std Dev": std__last_200,
        "Episodes": len(df),
        "Landing Success Rate (%)": 100 * landing_flags.mean(),
        "Soft Landing Rate (%)": 100 * soft_landing_flags.mean()
    })

    plt.plot(rolling_soft, label=label)

plt.title("Soft Landing Rate Over Episodes (Rolling Avg, Window=50)")
plt.xlabel("Episode")
plt.ylabel("Soft Landing Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- General Landing Plot ---
plt.figure(figsize=(10, 5))
for label, path in files.items():
    df = pd.read_csv(path)
    landing_flags = df["land"].astype(int)
    rolling_land = landing_flags.rolling(window=50).mean()
    plt.plot(rolling_land, label=label)

plt.title("Overall Landing Rate Over Episodes (Rolling Avg, Window=50)")
plt.xlabel("Episode")
plt.ylabel("Landing Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- Visual Summary Table ---
summary_df = pd.DataFrame(summary_data)

fig, ax = plt.subplots(figsize=(12, 2 + 0.3 * len(summary_df)))
ax.axis('off')
table = ax.table(
    cellText=summary_df.round(2).values,
    colLabels=summary_df.columns,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

plt.title("Reward Summary Statistics Table (Normalised, Last 200)", fontsize=15, pad=25)
plt.tight_layout()
plt.show()
