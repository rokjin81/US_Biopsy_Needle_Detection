import torch
import pandas as pd
import numpy as np
import cv2
import os
import time

# Load the trained LSTM model
class LSTMNetwork(torch.nn.Module):
    def __init__(self, input_size=5, hidden_size=128, output_size=3, num_layers=1):
        super(LSTMNetwork, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Output for the last time step
        return out

# Define function to process input CSV and prepare sequences
def prepare_sequences(input_csv, time_steps=10):
    data = pd.read_csv(input_csv)
    data['bottom_left_x'] /= 660
    data['bottom_left_y'] /= 616
    data['box_angle'] /= 90
    data['r_encode'] /= 90
    data['l_encode'] /= 100

    sequences = []
    frame_numbers = []
    input_features = data[['bottom_left_x', 'bottom_left_y', 'box_angle', 'r_encode', 'l_encode']].values
    frames = data['frame_number'].values

    for i in range(len(input_features) - time_steps + 1):
        if np.all(frames[i:i+time_steps] == np.arange(frames[i], frames[i] + time_steps)):
            sequences.append(input_features[i:i+time_steps])
            frame_numbers.append(frames[i+time_steps-1])

    return torch.tensor(sequences, dtype=torch.float32), frame_numbers

# Denormalize results for saving and visualization
def denormalize_results(results):
    results[:, 0] *= 660
    results[:, 1] *= 616
    results[:, 2] *= 90
    return results

# Draw and save results on images
def visualize_results(frame_numbers, results, input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for i, frame in enumerate(frame_numbers):
        image_path = os.path.join(input_folder, f"usimg{frame}.png")
        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path)
        x_tip, y_tip, angle = results[i]
        x_tip, y_tip = int(round(x_tip)), int(round(y_tip))

        # Skip if out of bounds
        if x_tip < 0 or x_tip > 660 or y_tip < 0 or y_tip > 616:
            continue

        # Draw the cyan dot at (x_tip, y_tip) and add the coordinates as text
        cv2.circle(image, (x_tip, y_tip), 5, (255, 255, 0), -1)
        cv2.putText(image, f"({x_tip}, {y_tip})", (x_tip - 200, y_tip),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        # Draw a line representing the angle
        length = max(image.shape[:2])
        end_x = int(x_tip + length * np.cos(np.radians(angle)))
        end_y = int(y_tip - length * np.sin(np.radians(angle)))
        cv2.line(image, (x_tip, y_tip), (end_x, end_y), (255, 255, 0), 2)
        cv2.putText(image, f"{angle:.2f} deg", (x_tip - 200, y_tip - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        output_path = os.path.join(output_folder, f"usimg{frame}.png")
        cv2.imwrite(output_path, image)

# Main inference script
if __name__ == "__main__":
    # Parameters
    input_csv = "lstm_inputs_veri1.csv"
    model_path = "lstm_model_best.pth"
    input_folder = "Image Data_3"
    output_folder = "Output_Images"
    output_csv = "inference_results.csv"
    time_steps = 10

    # Load model
    model = LSTMNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Prepare sequences
    sequences, frame_numbers = prepare_sequences(input_csv, time_steps)

    # Perform inference with timing
    timings = []
    with torch.no_grad():
        for i in range(len(sequences)):
            start_time = time.time()
            prediction = model(sequences[i:i+1]).numpy()
            end_time = time.time()
            timings.append(end_time - start_time)

            if i == 0:
                predictions = prediction
            else:
                predictions = np.vstack((predictions, prediction))

    # Denormalize results
    predictions = denormalize_results(predictions)

    # Set invalid results for out-of-bounds predictions
    for i, (x_tip, y_tip, _) in enumerate(predictions):
        if x_tip < 0 or x_tip > 660 or y_tip < 0 or y_tip > 616:
            predictions[i] = [-1, -1, -1]

    # Save results to CSV
    results_df = pd.DataFrame({
        "frame_number": frame_numbers,
        "x_tip": predictions[:, 0],
        "y_tip": predictions[:, 1],
        "angle": predictions[:, 2],
        "time_taken": timings
    })
    results_df.to_csv(output_csv, index=False)

    # Visualize and save results on images
    visualize_results(frame_numbers, predictions, input_folder, output_folder)

    print(f"Inference completed. Results saved to {output_csv}. Visualized images saved to {output_folder}.")
