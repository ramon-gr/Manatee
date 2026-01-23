import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_and_plot(path, title, original):
    # Load 200 rows, but also limit to 200 columns for the heatmap
    df = pd.read_csv(path, sep=' ', header=None, dtype='float32', nrows=500, engine='c')
    
    # Slice to a square for a clean visualization
    df_plot = df.iloc[:, :200] 

    if not original:
        threshold = 3.2
        df_plot = df_plot.where(df_plot >= threshold, 0)
    
    # Stats
    vals = df_plot.values
    v_min, v_max = vals.min(), vals.max()
    v_mean = vals.mean()
    zero_pct = (vals == 0).sum() / vals.size * 100
    
    print(f"\n=== {title} ===")
    print(f"Plotting Shape: {df_plot.shape}")
    print(f"Value range: {v_min:.4f} to {v_max:.4f}")
    print(f"Mean: {v_mean:.4f}")
    print(f"% zeros: {zero_pct:.2f}%")

    plt.figure(figsize=(10, 8))
    # Using robust=True handles outliers in the color scale automatically
    sns.heatmap(df_plot, cmap='viridis', cbar=True, robust=True)
    plt.title(title)
    plt.tight_layout()
    
    return df

# File paths
data_x = r"GSE72857\processed\data_x.csv.gz"
data_x_predict = r"GSE72857\processed\encode_predict_x_rec_data.csv.gz"
data_x_gen = r"GSE72857\benchmark\encode_generate_x_rec_data.csv.gz"

# check file sizes
print("File sizes:")
print(f"Original: {os.path.getsize(data_x) / 1024 / 1024:.2f} MB")
print(f"Predict: {os.path.getsize(data_x_predict) / 1024 / 1024:.2f} MB")
print(f"Generated: {os.path.getsize(data_x_gen) / 1024 / 1024:.2f} MB")

# Load and plot one at a time
df_original = load_and_plot(data_x, "Original Data", True)
df_predict = load_and_plot(data_x_predict, "Predicted Data", False)
df_generated = load_and_plot(data_x_gen, "Generated Data", False)
plt.show()

original_means = df_original.mean(axis=0)
predict_means = df_predict.mean(axis=0)

plt.figure(figsize=(10, 4))
plt.hist(df_original.values.flatten(), bins=50, alpha=0.5, label='Original', range=(0, 20))
plt.hist(df_predict.values.flatten(), bins=50, alpha=0.5, label='Predicted', range=(0, 20))
plt.legend()
plt.title("Distribution of Expression Levels")
plt.show()

plt.scatter(original_means, predict_means, alpha=0.3)
plt.plot([0, original_means.max()], [0, original_means.max()], 'r--') # Identity line
plt.xlabel("Original Gene Means")
plt.ylabel("Predicted Gene Means")
plt.show()

correlation = original_means.corr(predict_means)
print(f"Gene-wise Mean Correlation: {correlation:.4f}")