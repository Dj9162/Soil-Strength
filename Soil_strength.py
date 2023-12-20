import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Load the data
df = pd.read_excel("BTPdata.xlsx", sheet_name="Table")

# Separate features (X) and target variables (y)
X = df[['S/B', 'slope angle', 'Force angle', 'friction angle']]
y = df[['Nc', 'Nq', 'Ny']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to optimize model for a target variable
def optimize_model(X_train, y_train, X_test, target_variable):
    # Define the parameter grid for XGBoost
    param_grid_xgb = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 150]
    }

    # Create the grid search for XGBoost
    grid_search_xgb = GridSearchCV(xgb.XGBRegressor(random_state=42), param_grid_xgb, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    # Fit the grid search to the data
    grid_search_xgb.fit(X_train, y_train[target_variable])

    # Get the best parameters
    best_params_xgb = grid_search_xgb.best_params_

    # Train an XGBoost model with the best parameters
    best_model_xgb = xgb.XGBRegressor(random_state=42, **best_params_xgb)
    best_model_xgb.fit(X_train, y_train[target_variable])

    # Make predictions on the test set
    y_pred_target_variable_xgb = best_model_xgb.predict(X_test)

    # Calculate the optimized RMSE for the target variable using XGBoost
    rmse_target_variable_optimized_xgb = mean_squared_error(y_test[target_variable], y_pred_target_variable_xgb, squared=False)

    # Return the trained model and predictions
    return best_model_xgb, y_pred_target_variable_xgb

# Optimize and predict Nc
model_nc, y_pred_nc = optimize_model(X_train, y_train, X_test, 'Nc')

# Optimize and predict Nq
model_nq, y_pred_nq = optimize_model(X_train, y_train, X_test, 'Nq')

# Optimize and predict Ny
model_ny, y_pred_ny = optimize_model(X_train, y_train, X_test, 'Ny')

# Calculate RMSE for each target variable
rmse_nc = mean_squared_error(y_test['Nc'], y_pred_nc, squared=False)
rmse_nq = mean_squared_error(y_test['Nq'], y_pred_nq, squared=False)
rmse_ny = mean_squared_error(y_test['Ny'], y_pred_ny, squared=False)

# Assuming you have the original and predicted values for Nc, Nq, and Ny
original_Nc_values = y_test['Nc'].values
predicted_Nc_values = y_pred_nc

original_Nq_values = y_test['Nq'].values
predicted_Nq_values = y_pred_nq

original_Ny_values = y_test['Ny'].values
predicted_Ny_values = y_pred_ny

# Create a range of values for the straight line
x_range = np.linspace(min(original_Nc_values.min(), original_Nq_values.min(), original_Ny_values.min()),
                      max(original_Nc_values.max(), original_Nq_values.max(), original_Ny_values.max()), 100)

# GUI
root = tk.Tk()
root.title("Prediction and RMSE Plots")

# Center the main window on the screen
window_width = 1300  # Set your desired width
window_height = 350  # Set your desired height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_position = (screen_width - window_width) // 2
y_position = (screen_height - window_height) // 2

root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

# Entry widgets
labels = ['Setback Distance,S(m):', 'Footing Width,B(m): ', 'Slope Angle,𝛽°', 'Force Angle,ɑ°', 'Friction Angle,φ°']
entries = [ttk.Entry(root) for _ in labels]

# Load and resize the image
image_path = "Picture1.png"  # Replace with the path to your image file
image = Image.open(image_path)
image = image.resize((750, 300), Image.LANCZOS)
tk_image = ImageTk.PhotoImage(image)

# Create a label to display the image
image_label = tk.Label(root, image=tk_image)
image_label.grid(row=0, column=len(labels), rowspan=len(labels) + 3, padx=40, pady=10, sticky='e')

# Result label
result_label = ttk.Label(root, text="Calculated Values - Nc: 0.00, Nq: 0.00, Ny: 0.00")

# Buttons
predict_button = ttk.Button(root, text="Calculate", command=lambda: get_predictions(
    *[float(entry.get()) for entry in entries]
)) 

# Function to show RMSE plots
def show_rmse_plots():
    # Plotting for Nc
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(x_range, x_range, color='red', label='Original Data')
    plt.scatter(original_Nc_values, predicted_Nc_values, color='blue', alpha=0.5, label='Predicted Values')
    plt.xlabel('Original Nc Values')
    plt.ylabel('Predicted Nc Values')
    plt.title('Original vs Predicted Values for Nc')
    plt.legend()

    # Plotting for Nq
    plt.subplot(1, 3, 2)
    plt.plot(x_range, x_range, color='red', label='Original Data')
    plt.scatter(original_Nq_values, predicted_Nq_values, color='green', alpha=0.5, label='Predicted Values')
    plt.xlabel('Original Nq Values')
    plt.ylabel('Predicted Nq Values')
    plt.title('Original vs Predicted Values for Nq')
    plt.legend()

    # Plotting for Ny
    plt.subplot(1, 3, 3)
    plt.plot(x_range, x_range, color='red', label='Original Data')
    plt.scatter(original_Ny_values, predicted_Ny_values, color='purple', alpha=0.5, label='Predicted Values')
    plt.xlabel('Original Ny Values')
    plt.ylabel('Predicted Ny Values')
    plt.title('Original vs Predicted Values for Ny')
    plt.legend()

    plt.tight_layout()
    plt.show()

rmse_button = ttk.Button(root, text="Show RMSE Plots", command=show_rmse_plots)

# Layout
for i, label in enumerate(labels):
    ttk.Label(root, text=label).grid(row=i, column=0, padx=10, pady=5, sticky='e')
    entries[i].grid(row=i, column=1, padx=10, pady=5, sticky='w')

result_label.grid(row=len(labels), column=0, columnspan=2, pady=10)
predict_button.grid(row=len(labels) + 1, column=0, columnspan=2, pady=10)
rmse_button.grid(row=len(labels) + 2, column=0, columnspan=2, pady=10)

def get_predictions(S, B, slope_angle, force_angle, friction_angle):
    # Check conditions and display warnings if necessary
    warning_message = ""

    if not (0 <= S/B <= 3):
        warning_message += "Warning: S/B should be between 0 and 3.\n"

    if not (0 <= slope_angle <= 40):
        warning_message += "Warning: Slope angle should be between 0 and 40.\n"

    if not (0 <= force_angle <= 20):
        warning_message += "Warning: Force angle should be between 0 and 20.\n"

    if not (0 <= friction_angle <= 40):
        warning_message += "Warning: Friction angle should be between 0 and 40.\n"

    # Display warnings on a dedicated label
    warning_label.config(text=warning_message)

    # If there are warnings, set predictions to zero and update result label
    if warning_message:
        result_label.config(text="Calculated Values - Nc: 0.00, Nq: 0.00, Ny: 0.00")
        return

    # Prepare input for prediction
    input_data = [[S/B, slope_angle, force_angle, friction_angle]]

    # Get predictions
    pred_nc, pred_nq, pred_ny = model_nc.predict(input_data)[0], model_nq.predict(input_data)[0], model_ny.predict(input_data)[0]

    # Set negative predictions to zero
    pred_nc = max(pred_nc, 0)
    pred_nq = max(pred_nq, 0)
    pred_ny = max(pred_ny, 0)

    # Update result label with predictions
    result_label.config(text=f"Values - Nc: {pred_nc:.2f}, Nq: {pred_nq:.2f}, Ny: {pred_ny:.2f}")

# ... (Rest of the GUI code)


# ...
# ... (Previous GUI code)

# Create labels for input value ranges
range_labels = [
    ttk.Label(root, text="S/B (0 to 3):", foreground="blue"),
    ttk.Label(root, text="S/B (0 to 3):", foreground="blue"),
    ttk.Label(root, text="Slope Angle (0 to 40):", foreground="blue"),
    ttk.Label(root, text="Force Angle (0 to 20):", foreground="blue"),
    ttk.Label(root, text="Friction Angle (0 to 40):", foreground="blue"),
]

# Layout for input value range labels
for i, range_label in enumerate(range_labels):
    range_label.grid(row=i, column=2, padx=10, pady=5, sticky='e')

# ... (Rest of the GUI code)


# Create a label for displaying warnings
warning_label = ttk.Label(root, text="", foreground="red")
warning_label.grid(row=len(labels) + 3, column=0, columnspan=2, pady=1)

# ...

# Run the Tkinter event loop
root.mainloop()