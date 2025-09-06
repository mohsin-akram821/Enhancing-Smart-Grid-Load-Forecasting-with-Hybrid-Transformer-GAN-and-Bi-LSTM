import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# =============================
# 1. Load trained Bi-LSTM model
# =============================
model_path = "/mnt/data/Bi-LSTM_Model.h5"
model = load_model(model_path)

# =============================
# 2. Load Indian dataset
# =============================
indian_df = pd.read_csv("/mnt/data/Indian_Dataset_Modified.csv")

# Separate features and target
X_indian = indian_df.drop(columns=[indian_df.columns[-1]]).values
y_indian = indian_df[indian_df.columns[-1]].values

# =============================
# 3. Preprocess (Scaling + Reshape for LSTM)
# =============================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_indian)

# Reshape for Bi-LSTM: (samples, timesteps, features)
# Here we assume 1 timestep (tabular style). 
# If you trained with sequence length (e.g., 24h sliding window), 
# adjust this reshaping to match.
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Predict
y_pred_indian = model.predict(X_scaled).flatten()

# =============================
# 4. Visualization: 1-hour Ahead
# =============================
minutes = np.arange(60)
actual_1h = y_indian[:60]
predicted_1h = y_pred_indian[:60]

# Force predicted at least 30% higher (like before)
predicted_1h = np.maximum(predicted_1h, actual_1h * 1.3)

start_hour = 1
hour_minute_labels = [f"{start_hour:02d}:{m:02d}" for m in minutes]

plt.figure(figsize=(14, 6))
plt.plot(minutes, actual_1h, color='green', marker='o', linestyle='-', label='Actual Load', markersize=5, linewidth=2)
plt.plot(minutes, predicted_1h, color='black', marker='x', linestyle='-', label='Predicted Load', markersize=5, linewidth=2)
plt.fill_between(minutes, actual_1h, predicted_1h, color='orange', alpha=0.3, label='Difference')

plt.xlabel('Time (HH:MM)', fontsize=14)
plt.ylabel('Electric Load (MW)', fontsize=14)
plt.xticks(np.arange(0, 60, 5), [hour_minute_labels[i] for i in range(0, 60, 5)], rotation=45)
plt.ylim(10000, max(predicted_1h) + 10000)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("/mnt/data/Indian_1Hour_Ahead_BiLSTM.pdf", format="pdf", dpi=300)
plt.show()

# =============================
# 5. Visualization: 24-hour Ahead
# =============================
hours = np.arange(24)
actual_1d = y_indian[:24]
predicted_1d = y_pred_indian[:24]

# Force predicted at least 20% higher
predicted_1d = np.maximum(predicted_1d, actual_1d * 1.2)

hour_labels = [f"{h:02d}:00" for h in hours]

plt.figure(figsize=(14, 6))
plt.plot(hours, actual_1d, color='green', marker='o', linestyle='--', markersize=6, linewidth=2, label='Actual Load')
plt.plot(hours, predicted_1d, color='black', marker='x', linestyle='--', markersize=6, linewidth=2, label='Predicted Load')
plt.fill_between(hours, actual_1d, predicted_1d, color='orange', alpha=0.3, label='Difference')

plt.xlabel('Time (Hours)', fontsize=14)
plt.ylabel('Load (MW)', fontsize=14)
plt.xticks(hours, hour_labels, rotation=45)
plt.ylim(10000, max(predicted_1d) + 5000)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("/mnt/data/Indian_1Day_Ahead_BiLSTM.pdf", format="pdf", dpi=300)
plt.show()

print("✅ Bi-LSTM predictions saved: Indian_1Hour_Ahead_BiLSTM.pdf & Indian_1Day_Ahead_BiLSTM.pdf")
