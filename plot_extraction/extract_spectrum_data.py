#!/usr/bin/env python3
"""
Extract absorption spectrum data from transmission.png
This script processes the image to extract (wavelength, absorbance) data points
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json

def extract_spectrum_data(image_path, max_points=100):
    """
    Extract spectrum data from a plot image
    
    Args:
        image_path: Path to the plot image
        max_points: Maximum number of data points to extract
        
    Returns:
        dict: JSON-like structure with extracted data points
    """
    
    # Load and convert image to numpy array
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        # Convert RGB to grayscale using standard weights
        img_gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        img_gray = img_array
    
    # Get image dimensions
    height, width = img_gray.shape
    
    # Define plot area (approximate boundaries based on typical matplotlib plots)
    # These values may need adjustment based on the specific image
    left_margin = int(width * 0.12)   # Left axis area
    right_margin = int(width * 0.95)  # Right boundary
    top_margin = int(height * 0.12)   # Top boundary
    bottom_margin = int(height * 0.88) # Bottom axis area
    
    plot_width = right_margin - left_margin
    plot_height = bottom_margin - top_margin
    
    # Extract the plot region
    plot_region = img_gray[top_margin:bottom_margin, left_margin:right_margin]
    
    # Find the darkest pixels in each column (representing the data line)
    data_points = []
    
    # Sample columns evenly across the plot width
    x_indices = np.linspace(0, plot_width-1, min(max_points, plot_width), dtype=int)
    
    for i, x_idx in enumerate(x_indices):
        column = plot_region[:, x_idx]
        
        # Find the minimum value (darkest pixel) in this column
        # The data line should be the darkest part
        min_idx = np.argmin(column)
        
        # Convert pixel coordinates to data coordinates
        # X-axis: wavelength from 400 to 700 nm
        wavelength = 400 + (x_idx / (plot_width - 1)) * (700 - 400)
        
        # Y-axis: absorbance from 0 to 1.2 (inverted because image y=0 is at top)
        absorbance = 1.2 * (1 - min_idx / (plot_height - 1))
        
        data_points.append({
            "x_value": round(wavelength, 2),
            "y_value": round(max(0, absorbance), 4)  # Ensure non-negative values
        })
    
    # Create the output structure similar to axiomatic-plots format
    result = {
        "series_points": [
            {
                "series_unique_id": 0,
                "points": data_points
            }
        ]
    }
    
    return result

def smooth_data(data_points, window_size=5):
    """
    Apply simple moving average smoothing to the data
    """
    if len(data_points) < window_size:
        return data_points
    
    smoothed = []
    for i in range(len(data_points)):
        if i < window_size // 2 or i >= len(data_points) - window_size // 2:
            smoothed.append(data_points[i])
        else:
            window = data_points[i - window_size//2 : i + window_size//2 + 1]
            avg_y = np.mean([p["y_value"] for p in window])
            smoothed.append({
                "x_value": data_points[i]["x_value"],
                "y_value": round(avg_y, 4)
            })
    
    return smoothed

def main():
    image_path = "/Users/englund/Projects/2025-ai-experiments-MCPs/plot_extraction/transmission.png"
    
    print("Extracting spectrum data from transmission.png...")
    
    # Extract data points
    result = extract_spectrum_data(image_path, max_points=100)
    
    # Apply smoothing to reduce noise
    result["series_points"][0]["points"] = smooth_data(
        result["series_points"][0]["points"], 
        window_size=3
    )
    
    # Save to JSON file
    output_file = "/Users/englund/Projects/2025-ai-experiments-MCPs/plot_extraction/extracted_spectrum_data.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Data extracted and saved to {output_file}")
    print(f"Number of data points: {len(result['series_points'][0]['points'])}")
    
    # Display some sample points
    points = result["series_points"][0]["points"]
    print("\nSample data points:")
    for i in range(0, len(points), len(points)//10):
        p = points[i]
        print(f"  Wavelength: {p['x_value']} nm, Absorbance: {p['y_value']}")
    
    # Find peaks
    y_values = [p["y_value"] for p in points]
    max_abs = max(y_values)
    max_idx = y_values.index(max_abs)
    peak_wavelength = points[max_idx]["x_value"]
    
    print(f"\nSpectrum characteristics:")
    print(f"  Maximum absorbance: {max_abs} at {peak_wavelength} nm")
    print(f"  Wavelength range: {points[0]['x_value']} - {points[-1]['x_value']} nm")
    print(f"  Absorbance range: {min(y_values)} - {max(y_values)}")
    
    return result

if __name__ == "__main__":
    main()