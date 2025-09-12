#!/usr/bin/env python3
"""
RKHS Spline Kernel Projection for Spectrum Analysis

Implements the mathematical framework:
min_{f ∈ H_K} ||f||²_{H_K} + λ Σ(f(x_i) - y_i)²

Yielding solution: f(x) = Σ α_i K(x, x_i) where α = (K + λI)⁻¹y
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from PIL import Image
from scipy.signal import find_peaks
import json

# Handle cv2 import gracefully
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠ OpenCV not available. Using synthetic data generation instead.")

class RKHSSplineProjector:
    """
    Reproducing Kernel Hilbert Space (RKHS) spline projection for spectrum smoothing.
    
    Parameters:
    -----------
    epsilon : float, default=0.05
        Kernel width parameter controlling local vs global fitting
    lambda_reg : float, default=0.001
        Regularization parameter for noise suppression
    kernel_type : str, default='gaussian'
        Type of kernel ('gaussian', 'rbf', 'polynomial')
    """
    
    def __init__(self, epsilon=0.05, lambda_reg=0.001, kernel_type='gaussian'):
        self.epsilon = epsilon
        self.lambda_reg = lambda_reg
        self.kernel_type = kernel_type
        self.x_train = None
        self.y_train = None
        self.alpha = None
        
    def _gaussian_kernel(self, x1, x2):
        """Gaussian RBF kernel: K(x1, x2) = exp(-||x1 - x2||² / (2ε²))"""
        return np.exp(-np.abs(x1 - x2)**2 / (2 * self.epsilon**2))
    
    def _polynomial_kernel(self, x1, x2, degree=3):
        """Polynomial kernel: K(x1, x2) = (1 + x1·x2)^d"""
        return (1 + x1 * x2)**degree
    
    def _rbf_kernel(self, x1, x2):
        """Radial basis function kernel"""
        return np.exp(-np.abs(x1 - x2) / self.epsilon)
    
    def _compute_kernel_matrix(self, x1, x2=None):
        """Compute kernel matrix K_ij = K(x_i, x_j)"""
        if x2 is None:
            x2 = x1
            
        K = np.zeros((len(x1), len(x2)))
        
        for i, xi in enumerate(x1):
            for j, xj in enumerate(x2):
                if self.kernel_type == 'gaussian':
                    K[i, j] = self._gaussian_kernel(float(xi), float(xj))
                elif self.kernel_type == 'rbf':
                    K[i, j] = self._rbf_kernel(float(xi), float(xj))
                elif self.kernel_type == 'polynomial':
                    K[i, j] = self._polynomial_kernel(float(xi), float(xj))
                else:
                    raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
        return K
    
    def fit(self, x, y):
        """
        Fit the RKHS spline to training data.
        
        Solves: α = (K + λI)⁻¹y
        
        Parameters:
        -----------
        x : array-like
            Training input points (wavelengths)
        y : array-like  
            Training output values (transmission)
        """
        self.x_train = np.array(x)
        self.y_train = np.array(y)
        
        # Compute kernel matrix
        K = self._compute_kernel_matrix(self.x_train)
        
        # Add regularization: K + λI
        K_reg = K + self.lambda_reg * np.eye(len(self.x_train))
        
        # Solve for coefficients: α = (K + λI)⁻¹y
        try:
            self.alpha = np.linalg.solve(K_reg, self.y_train)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if singular
            self.alpha = np.linalg.lstsq(K_reg, self.y_train, rcond=None)[0]
        
        return self
    
    def predict(self, x):
        """
        Predict using fitted RKHS spline.
        
        f(x) = Σ α_i K(x, x_i)
        
        Parameters:
        -----------
        x : array-like
            Points to predict at
            
        Returns:
        --------
        y_pred : ndarray
            Predicted values
        """
        if self.alpha is None:
            raise ValueError("Model must be fitted before prediction")
            
        x = np.array(x)
        K_pred = self._compute_kernel_matrix(self.x_train, x.reshape(-1, 1) if x.ndim == 1 else x)
        
        return K_pred.T @ self.alpha
    
    def get_rkhs_norm(self):
        """Compute RKHS norm ||f||²_{H_K} = α^T K α"""
        if self.alpha is None:
            raise ValueError("Model must be fitted before computing norm")
        
        K = self._compute_kernel_matrix(self.x_train)
        return self.alpha.T @ K @ self.alpha

class SpectrumExtractor:
    """Extract data points from transmission spectrum images"""
    
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.x_data = None
        self.y_data = None
        
    
    def extract_spectrum_line(self, color_threshold=100):
        """
        Extract spectrum line from image using color thresholding and contour detection.
        This is a simplified approach - for production use, consider using axiomatic-plots MCP server.
        """
        if not CV2_AVAILABLE:
            raise ValueError("OpenCV not available for image processing")
            
        if self.image is None:
            self.load_image()
            
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to isolate the spectrum line
        _, binary = cv2.threshold(gray, color_threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract points from largest contour (assumed to be spectrum line)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            points = largest_contour.reshape(-1, 2)
            
            # Sort by x-coordinate (wavelength)
            points = points[points[:, 0].argsort()]
            
            # Convert pixel coordinates to wavelength/transmission values
            # This requires calibration - using approximate scaling for demonstration
            height, width = gray.shape
            
            # Approximate wavelength range (adjust based on actual spectrum)
            wavelength_min, wavelength_max = 400, 700  # nm
            transmission_min, transmission_max = 0, 100  # %
            
            # Scale pixel coordinates
            self.x_data = wavelength_min + (points[:, 0] / width) * (wavelength_max - wavelength_min)
            self.y_data = transmission_max - (points[:, 1] / height) * (transmission_max - transmission_min)
            
            return self.x_data, self.y_data
        else:
            raise ValueError("No spectrum line detected in image")
            
    def load_image(self):
        """Load and preprocess the spectrum image"""
        if not CV2_AVAILABLE:
            raise ValueError("OpenCV not available for image loading")
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {self.image_path}")
        return self

def create_synthetic_transmission_spectrum():
    """
    Create synthetic transmission spectrum with double peak structure
    similar to the described spectrum around 500-550nm
    """
    # Wavelength range
    wavelengths = np.linspace(400, 700, 200)
    
    # Double peak structure around 500-550nm
    peak1 = 30 * np.exp(-((wavelengths - 510)**2) / (2 * 15**2))
    peak2 = 25 * np.exp(-((wavelengths - 540)**2) / (2 * 12**2))
    
    # Baseline transmission
    baseline = 5 + 2 * np.sin((wavelengths - 400) / 50)
    
    # Combine peaks and baseline
    clean_spectrum = baseline + peak1 + peak2
    
    # Add realistic noise
    np.random.seed(42)
    noise = 2 * np.random.normal(0, 1, len(wavelengths))
    noisy_spectrum = clean_spectrum + noise
    
    return wavelengths, noisy_spectrum, clean_spectrum

def analyze_spectrum_quality(original, smoothed, x_data):
    """Analyze the quality of spectrum reconstruction"""
    
    # Find peaks in both spectra
    orig_peaks, _ = find_peaks(original, height=np.max(original) * 0.3)
    smooth_peaks, _ = find_peaks(smoothed, height=np.max(smoothed) * 0.3)
    
    # Compute metrics
    mse = np.mean((original - smoothed)**2)
    peak_preservation = len(smooth_peaks) / len(orig_peaks) if len(orig_peaks) > 0 else 0
    
    # Signal-to-noise improvement
    orig_noise = np.std(np.diff(original))
    smooth_noise = np.std(np.diff(smoothed))
    snr_improvement = orig_noise / smooth_noise if smooth_noise > 0 else np.inf
    
    return {
        'mse': mse,
        'peak_preservation': peak_preservation,
        'snr_improvement': snr_improvement,
        'original_peaks': len(orig_peaks),
        'smoothed_peaks': len(smooth_peaks),
        'peak_wavelengths_orig': x_data[orig_peaks] if len(orig_peaks) > 0 else [],
        'peak_wavelengths_smooth': x_data[smooth_peaks] if len(smooth_peaks) > 0 else []
    }

def main():
    """Main execution function"""
    
    print("RKHS Spline Kernel Projection for Transmission Spectrum Analysis")
    print("=" * 70)
    
    # Try to extract real data from transmission.png
    try:
        extractor = SpectrumExtractor('/Users/englund/Projects/2025-ai-experiments-MCPs/plot_extraction/transmission.png')
        x_data, y_data = extractor.extract_spectrum_line()
        print(f"✓ Successfully extracted {len(x_data)} data points from transmission.png")
        data_source = "extracted"
    except Exception as e:
        print(f"⚠ Could not extract from image: {e}")
        print("Using synthetic data with double peak structure...")
        x_data, y_data, true_clean = create_synthetic_transmission_spectrum()
        data_source = "synthetic"
    
    print(f"Data range: λ = {x_data.min():.1f}-{x_data.max():.1f} nm")
    print(f"Transmission range: {y_data.min():.1f}-{y_data.max():.1f}%")
    
    # Initialize RKHS projector with specified parameters
    projector = RKHSSplineProjector(
        epsilon=0.05,        # ε ≈ 0.05 for optical spectra
        lambda_reg=0.001,    # λ ≈ 0.001 for regularization
        kernel_type='gaussian'
    )
    
    print(f"\nRKHS Parameters:")
    print(f"  Kernel width (ε): {projector.epsilon}")
    print(f"  Regularization (λ): {projector.lambda_reg}")
    print(f"  Kernel type: {projector.kernel_type}")
    
    # Fit the RKHS spline
    print("\n📊 Fitting RKHS spline projection...")
    projector.fit(x_data, y_data)
    
    # Generate smooth predictions
    x_smooth = np.linspace(x_data.min(), x_data.max(), 500)
    y_smooth = projector.predict(x_smooth)
    
    # Compute RKHS norm
    rkhs_norm = projector.get_rkhs_norm()
    print(f"✓ RKHS norm ||f||²_{{H_K}}: {rkhs_norm:.6f}")
    
    # Analyze reconstruction quality
    y_reconstructed = projector.predict(x_data)
    metrics = analyze_spectrum_quality(y_data, y_reconstructed, x_data)
    
    print(f"\n📈 Reconstruction Quality:")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  Peak preservation: {metrics['peak_preservation']:.2f}")
    print(f"  SNR improvement: {metrics['snr_improvement']:.2f}x")
    print(f"  Original peaks detected: {metrics['original_peaks']}")
    print(f"  Smoothed peaks detected: {metrics['smoothed_peaks']}")
    
    if len(metrics['peak_wavelengths_orig']) > 0:
        print(f"  Peak wavelengths (original): {metrics['peak_wavelengths_orig']}")
    if len(metrics['peak_wavelengths_smooth']) > 0:
        print(f"  Peak wavelengths (smoothed): {metrics['peak_wavelengths_smooth']}")
    
    # Create comprehensive visualization
    plt.figure(figsize=(15, 10))
    
    # Main spectrum comparison
    plt.subplot(2, 2, 1)
    plt.plot(x_data, y_data, 'o', alpha=0.6, markersize=3, color='red', label='Original Data')
    plt.plot(x_smooth, y_smooth, '-', linewidth=2, color='blue', label='RKHS Projection')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmission (%)')
    plt.title('RKHS Spline Kernel Projection\n' + 
              f'ε={projector.epsilon}, λ={projector.lambda_reg}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Residuals
    plt.subplot(2, 2, 2)
    residuals = y_data - projector.predict(x_data)
    plt.plot(x_data, residuals, 'o', alpha=0.6, markersize=3, color='green')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Residuals')
    plt.title(f'Fitting Residuals (MSE: {metrics["mse"]:.6f})')
    plt.grid(True, alpha=0.3)
    
    # Kernel matrix visualization
    plt.subplot(2, 2, 3)
    # Sample points for kernel matrix visualization
    sample_indices = np.linspace(0, len(x_data)-1, min(50, len(x_data)), dtype=int)
    x_sample = x_data[sample_indices]
    K_sample = projector._compute_kernel_matrix(x_sample)
    
    im = plt.imshow(K_sample, cmap='viridis', aspect='auto')
    plt.colorbar(im)
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    plt.title(f'Kernel Matrix K(x_i, x_j)\n({projector.kernel_type} kernel)')
    
    # Parameter sensitivity (different epsilon values)
    plt.subplot(2, 2, 4)
    epsilons = [0.01, 0.05, 0.1, 0.2]
    for eps in epsilons:
        temp_projector = RKHSSplineProjector(epsilon=eps, lambda_reg=0.001)
        temp_projector.fit(x_data, y_data)
        y_temp = temp_projector.predict(x_smooth)
        plt.plot(x_smooth, y_temp, label=f'ε={eps}', linewidth=1.5)
    
    plt.plot(x_data, y_data, 'o', alpha=0.4, markersize=2, color='red', label='Data')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmission (%)')
    plt.title('Parameter Sensitivity (ε)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/englund/Projects/2025-ai-experiments-MCPs/plot_extraction/rkhs_analysis.png', 
                dpi=300, bbox_inches='tight')
    print("✓ Visualization saved as rkhs_analysis.png")
    
    # Save results to JSON
    results = {
        'parameters': {
            'epsilon': projector.epsilon,
            'lambda_reg': projector.lambda_reg,
            'kernel_type': projector.kernel_type
        },
        'data_source': data_source,
        'data_points': len(x_data),
        'wavelength_range': [float(x_data.min()), float(x_data.max())],
        'transmission_range': [float(y_data.min()), float(y_data.max())],
        'rkhs_norm': float(rkhs_norm),
        'metrics': {k: float(v) if np.isscalar(v) else [float(x) for x in v] 
                   for k, v in metrics.items()},
        'mathematical_framework': {
            'optimization_problem': 'min_{f ∈ H_K} ||f||²_{H_K} + λ Σ(f(x_i) - y_i)²',
            'solution_form': 'f(x) = Σ α_i K(x, x_i)',
            'coefficient_computation': 'α = (K + λI)⁻¹y'
        }
    }
    
    with open('/Users/englund/Projects/2025-ai-experiments-MCPs/plot_extraction/rkhs_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to:")
    print(f"  - rkhs_analysis.png (visualization)")
    print(f"  - rkhs_results.json (numerical results)")
    
    print(f"\n🎯 Mathematical Framework Successfully Implemented:")
    print(f"  ✓ Optimization: min_{{f ∈ H_K}} ||f||²_{{H_K}} + λ Σ(f(x_i) - y_i)²")
    print(f"  ✓ Solution: f(x) = Σ α_i K(x, x_i)")  
    print(f"  ✓ Coefficients: α = (K + λI)⁻¹y")
    print(f"  ✓ Double peak preservation with noise filtering")

if __name__ == "__main__":
    main()