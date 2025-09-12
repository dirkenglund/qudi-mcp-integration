#!/usr/bin/env python3
"""
Simple transmission spectrum extractor using PIL
Extracts approximate data points from transmission.png for RKHS analysis
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json

def extract_spectrum_from_image(image_path, wavelength_range=(400, 700), transmission_range=(0, 100)):
    """
    Extract spectrum data from image using simple pixel analysis.
    
    This is a basic approach - for production use, consider using axiomatic-plots MCP server.
    """
    
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    print(f"Image dimensions: {img_array.shape}")
    
    # Convert to grayscale if color image
    if len(img_array.shape) == 3:
        # Convert to grayscale using standard weights
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        gray = img_array
    
    height, width = gray.shape
    print(f"Processing grayscale image: {width} x {height}")
    
    # Find the darkest path (spectrum line) in each column
    x_pixels = []
    y_pixels = []
    
    # Look for the darkest points in each column (assuming dark spectrum line on light background)
    for x in range(width):
        column = gray[:, x]
        # Find the darkest point in this column
        min_idx = np.argmin(column)
        # Only include if it's significantly darker than average
        if column[min_idx] < np.mean(column) - np.std(column):
            x_pixels.append(x)
            y_pixels.append(min_idx)
    
    if len(x_pixels) < 10:
        # Try alternative approach - look for edges
        print("Trying edge-based approach...")
        edges = np.abs(np.diff(gray, axis=0))
        
        x_pixels = []
        y_pixels = []
        
        for x in range(width):
            column_edges = edges[:, x] if x < edges.shape[1] else []
            if len(column_edges) > 0:
                # Find strongest edge
                max_edge_idx = np.argmax(column_edges)
                if column_edges[max_edge_idx] > np.mean(column_edges) + np.std(column_edges):
                    x_pixels.append(x)
                    y_pixels.append(max_edge_idx)
    
    if len(x_pixels) < 10:
        raise ValueError(f"Could not extract sufficient data points from image. Only found {len(x_pixels)} points.")
    
    # Convert pixel coordinates to physical units
    x_pixels = np.array(x_pixels)
    y_pixels = np.array(y_pixels)
    
    # Map x pixels to wavelength
    wavelengths = wavelength_range[0] + (x_pixels / width) * (wavelength_range[1] - wavelength_range[0])
    
    # Map y pixels to transmission (flip y-axis since image coordinates are top-to-bottom)
    transmission = transmission_range[1] - (y_pixels / height) * (transmission_range[1] - transmission_range[0])
    
    # Sort by wavelength
    sort_idx = np.argsort(wavelengths)
    wavelengths = wavelengths[sort_idx]
    transmission = transmission[sort_idx]
    
    # Remove duplicates by averaging nearby points
    unique_wavelengths = []
    unique_transmission = []
    
    tolerance = (wavelength_range[1] - wavelength_range[0]) / width * 2  # 2-pixel tolerance
    
    i = 0
    while i < len(wavelengths):
        # Find all points within tolerance
        close_indices = np.abs(wavelengths - wavelengths[i]) < tolerance
        
        # Average them
        avg_wavelength = np.mean(wavelengths[close_indices])
        avg_transmission = np.mean(transmission[close_indices])
        
        unique_wavelengths.append(avg_wavelength)
        unique_transmission.append(avg_transmission)
        
        # Skip processed points
        i = np.max(np.where(close_indices)[0]) + 1
    
    return np.array(unique_wavelengths), np.array(unique_transmission)

def plot_extracted_data(wavelengths, transmission, save_path=None):
    """Plot the extracted spectrum data"""
    plt.figure(figsize=(12, 6))
    plt.plot(wavelengths, transmission, 'o-', markersize=3, linewidth=1)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmission (%)')
    plt.title('Extracted Transmission Spectrum Data')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Extracted data plot saved to: {save_path}")
    
    return plt.gcf()

def main():
    """Main execution"""
    image_path = '/Users/englund/Projects/2025-ai-experiments-MCPs/plot_extraction/transmission.png'
    
    try:
        print("Extracting spectrum data from transmission.png...")
        wavelengths, transmission = extract_spectrum_from_image(
            image_path,
            wavelength_range=(400, 700),  # Adjust based on actual spectrum range
            transmission_range=(0, 100)   # Adjust based on actual transmission range
        )
        
        print(f"✓ Extracted {len(wavelengths)} data points")
        print(f"Wavelength range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
        print(f"Transmission range: {transmission.min():.1f} - {transmission.max():.1f}%")
        
        # Look for double peak structure around 500-550nm
        peak_region = (wavelengths >= 500) & (wavelengths <= 550)
        if np.any(peak_region):
            peak_wavelengths = wavelengths[peak_region]
            peak_transmissions = transmission[peak_region]
            print(f"Double peak region (500-550nm): {len(peak_wavelengths)} points")
            print(f"Peak transmission range: {peak_transmissions.min():.1f} - {peak_transmissions.max():.1f}%")
        
        # Plot extracted data
        plot_extracted_data(wavelengths, transmission, 
                          '/Users/englund/Projects/2025-ai-experiments-MCPs/plot_extraction/extracted_spectrum.png')
        
        # Save extracted data
        data = {
            'wavelengths': wavelengths.tolist(),
            'transmission': transmission.tolist(),
            'extraction_method': 'simple_pil_based',
            'image_source': image_path,
            'data_points': len(wavelengths),
            'wavelength_range': [float(wavelengths.min()), float(wavelengths.max())],
            'transmission_range': [float(transmission.min()), float(transmission.max())]
        }
        
        with open('/Users/englund/Projects/2025-ai-experiments-MCPs/plot_extraction/extracted_data.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Data saved to extracted_data.json")
        
        return wavelengths, transmission
        
    except Exception as e:
        print(f"❌ Error extracting data: {e}")
        return None, None

if __name__ == "__main__":
    main()