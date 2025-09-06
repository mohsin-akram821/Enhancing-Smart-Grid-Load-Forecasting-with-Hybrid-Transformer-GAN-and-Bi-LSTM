import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import os

# -------------------------------
# Helper Functions
# -------------------------------

def evaluate_and_plot(y_true, y_pred, title, plot_filename):
    """
    Evaluate model performance, plot actual vs predicted values, and save the plot.
    """
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "Title": title,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2
    }

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(y_true.values[:200], label="Actual", color="blue")
    plt.plot(y_pred[:200], label="Predicted", color="green")
    plt.fill_between(range(200), y_true.values[:200], y_pred[:200], color="red", alpha=0.2)
    plt.title(f"{title} - First 200 Points")
    plt.xlabel("Time Steps")
    plt.ylabel("Electric Load (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()

    return metrics


def generate_pdf_report(metrics_list, plots, output_pdf="Validation_Report.pdf"):
    """
    Generate a PDF report with evaluation metrics and plots.
    """
    doc = SimpleDocTemplate(output_pdf, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("<b>Model Validation Report</b>", styles['Title']))
    elements.append(Spacer(1, 12))

    # Add Metrics and Plots
    for i, metrics in enumerate(metrics_list):
        elements.append(Paragraph(f"<b>{metrics['Title']}</b>", styles['Heading2']))
        elements.append(Paragraph(f"RMSE: {metrics['RMSE']:.2f}", styles['Normal']))
        elements.append(Paragraph(f"MAE: {metrics['MAE']:.2f}", styles['Normal']))
        elements.append(Paragraph(f"MAPE: {metrics['MAPE']:.2f}%", styles['Normal']))
        elements.append(Paragraph(f"R²: {metrics['R2']:.4f}", styles['Normal']))
        elements.append(Spacer(1, 12))

        # Add corresponding plot
        if i < len(plots):
            elements.append(RLImage(plots[i], width=6*inch, height=3*inch))
            elements.append(Spacer(1, 24))

    doc.build(elements)
    print(f"PDF Report saved as {output_pdf}")


# -------------------------------
# Validation Pipeline
# -------------------------------

def validate_models(new_data_path, model_1h_path, model_1d_path, report_path="Validation_Report.pdf"):
    """
    Validate one-hour and one-day ahead models on a new dataset and generate PDF report.
    """
    # Create temp folder for plots
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Load New Dataset
    df = pd.read_csv(new_data_path)
    df = df.sort_values(by=["Year", "Month", "Day", "Hour"])

    # Create lag features
    df["Lag_1"] = df["Electric Load (MW)"].shift(1)
    df["Lag_24"] = df["Electric Load (MW)"].shift(24)
    df = df.dropna()

    # Expected features
    expected_features = [
        "Temperature (°C)", "Relative Humidity", "UVB Radiation (W/m²)", "UV Index",
        "UVA Radiation (W/m²)", "Specific Humidity (g/kg)", "Relative Humidity (%)",
        "Precipitation (mm)", "Wind Speed (10m) (m/s)", "Is Holiday (1=Open, 0=Closed)",
        "Month", "Day", "Hour", "Weekday (0=Sun, 6=Sat)", "Inflation Rate (%)",
        "Lag_1", "Lag_24"
    ]

    for col in expected_features:
        if col not in df.columns:
            df[col] = 0  # Fill missing columns with zeros

    X_new = df[expected_features]
    y_new_actual = df["Electric Load (MW)"]

    # Load Models
    model_1h = joblib.load(model_1h_path)
    model_1d = joblib.load(model_1d_path)

    # Predictions
    y_pred_1h_val = model_1h.predict(X_new)

    df["Target_1day"] = df["Electric Load (MW)"].shift(-24)
    df = df.dropna(subset=["Target_1day"])
    X_new_day = df[expected_features]
    y_new_actual_day = df["Target_1day"]
    y_pred_1d_val = model_1d.predict(X_new_day)

    # Evaluate and save plots
    metrics_list = []
    plots = []

    plot1 = "plots/one_hour_ahead.png"
    metrics_1h = evaluate_and_plot(y_new_actual, y_pred_1h_val, "Validation - One-Hour Ahead", plot1)
    metrics_list.append(metrics_1h)
    plots.append(plot1)

    plot2 = "plots/one_day_ahead.png"
    metrics_1d = evaluate_and_plot(y_new_actual_day, y_pred_1d_val, "Validation - One-Day Ahead", plot2)
    metrics_list.append(metrics_1d)
    plots.append(plot2)

    # Generate PDF Report
    generate_pdf_report(metrics_list, plots, output_pdf=report_path)


if __name__ == "__main__":
    new_dataset_path = "Indian_Preprocessed_Data.csv"
    one_hour_model_path = "one_hour_model.pkl"
    one_day_model_path = "one_day_model.pkl"
    report_file = "Validation_Report.pdf"

    validate_models(new_dataset_path, one_hour_model_path, one_day_model_path, report_file)
