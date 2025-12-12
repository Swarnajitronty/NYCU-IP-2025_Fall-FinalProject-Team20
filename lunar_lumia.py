import cv2
import numpy as np
import pywt
from scipy import ndimage
from skimage import exposure
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
from pathlib import Path


class LunarLumina:
    """
    Implements classical signal processing techniques including:
    - Image stacking
    - Wavelet-based denoising
    - Adaptive contrast enhancement (CLAHE)
    - Multi-pass edge sharpening (Unsharp Mask)
    """
    
    def __init__(self):
        self.processing_steps = []
        
    def stack_images(self, images, alignment=True):
        """
        Stack multiple lunar images to improve SNR.
        SNR improves by √N where N is the number of images.
        
        Args:
            images: List of image arrays or file paths
            alignment: Whether to align images before stacking
            
        Returns:
            Stacked image with improved SNR
        """
        print(f"Stacking {len(images)} images...")
        
        # Load images if paths are provided
        img_arrays = []
        for img in images:
            if isinstance(img, str) or isinstance(img, Path):
                img_arrays.append(cv2.imread(str(img), cv2.IMREAD_GRAYSCALE))
            else:
                img_arrays.append(img)
        
        if alignment:
            # Align images using feature detection
            img_arrays = self._align_images(img_arrays)
        
        # Stack by averaging
        stacked = np.mean(img_arrays, axis=0).astype(np.float32)
        
        snr_improvement = np.sqrt(len(images))
        print(f"Theoretical SNR improvement: {snr_improvement:.2f}x")
        
        self.processing_steps.append(f"Stacked {len(images)} frames")
        return stacked
    
    def _align_images(self, images):
        """Align images using ORB feature detection and homography."""
        print("Aligning images...")
        reference = images[0]
        aligned = [reference]
        
        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=5000)
        
        for i, img in enumerate(images[1:], 1):
            # Detect keypoints and descriptors
            kp1, des1 = orb.detectAndCompute(reference.astype(np.uint8), None)
            kp2, des2 = orb.detectAndCompute(img.astype(np.uint8), None)
            
            # Match features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Get matched points
            if len(matches) > 10:
                src_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)
                
                # Find homography
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                # Warp image
                h, w = reference.shape
                aligned_img = cv2.warpPerspective(img, M, (w, h))
                aligned.append(aligned_img)
            else:
                print(f"Warning: Insufficient matches for image {i}, using unaligned")
                aligned.append(img)
        
        return aligned
    
    def wavelet_denoise(self, image, wavelet='db4', level=3, threshold_type='soft'):
        """
        Apply wavelet-based denoising using discrete wavelet transform.
        
        Args:
            image: Input image array
            wavelet: Wavelet type (default: 'db4' - Daubechies 4)
            level: Decomposition level
            threshold_type: 'soft' or 'hard' thresholding
            
        Returns:
            Denoised image
        """
        print(f"Applying wavelet denoising (wavelet={wavelet}, level={level})...")
        
        # Clean input: remove NaN and inf values
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure image has valid range
        if np.max(image) == 0:
            print("Warning: Image is all zeros, skipping denoising")
            return image
        
        # Normalize to [0, 1]
        img_min = np.min(image)
        img_max = np.max(image)
        img_norm = (image - img_min) / (img_max - img_min) if img_max > img_min else image
        
        # Perform 2D wavelet decomposition
        coeffs = pywt.wavedec2(img_norm, wavelet, level=level)
        
        # Estimate noise standard deviation from finest scale
        sigma = self._estimate_noise_sigma(coeffs[-1][0])
        
        # Avoid extremely small sigma that can cause numerical issues
        sigma = max(sigma, 1e-10)
        
        # Apply thresholding to detail coefficients
        coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients
        
        for i in range(1, len(coeffs)):
            # Threshold each detail coefficient (cH, cV, cD)
            cH, cV, cD = coeffs[i]
            
            # Universal threshold: λ = σ * √(2 * log(N))
            N = cH.size
            threshold = sigma * np.sqrt(2 * np.log(N)) * (1 / np.sqrt(2**i))
            
            if threshold_type == 'soft':
                # Use safe soft thresholding that avoids division by zero
                cH_t = self._soft_threshold(cH, threshold)
                cV_t = self._soft_threshold(cV, threshold)
                cD_t = self._soft_threshold(cD, threshold)
            else:
                cH_t = pywt.threshold(cH, threshold, mode='hard')
                cV_t = pywt.threshold(cV, threshold, mode='hard')
                cD_t = pywt.threshold(cD, threshold, mode='hard')
            
            coeffs_thresh.append((cH_t, cV_t, cD_t))
        
        # Reconstruct image
        denoised = pywt.waverec2(coeffs_thresh, wavelet)
        
        # Clean reconstruction
        denoised = np.nan_to_num(denoised, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Restore original scale
        denoised = denoised * (img_max - img_min) + img_min
        
        # Crop to original size (waverec2 may change size slightly)
        denoised = denoised[:image.shape[0], :image.shape[1]]
        
        self.processing_steps.append(f"Wavelet denoising ({wavelet}, level {level})")
        return denoised.astype(np.float32)
    
    def _soft_threshold(self, data, threshold):
        """
        Safe soft thresholding that avoids division issues.
        """
        magnitude = np.abs(data)
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.sign(data) * np.maximum(magnitude - threshold, 0)
        return np.nan_to_num(result, nan=0.0)
    
    def _estimate_noise_sigma(self, detail_coeffs):
        """Estimate noise standard deviation using MAD (Median Absolute Deviation)."""
        return np.median(np.abs(detail_coeffs - np.median(detail_coeffs))) / 0.6745
    
    def adaptive_contrast_enhancement(self, image, clip_limit=2.0, tile_size=(8, 8)):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Input image
            clip_limit: Contrast limiting threshold
            tile_size: Size of grid for histogram equalization
            
        Returns:
            Contrast-enhanced image
        """
        print(f"Applying CLAHE (clip_limit={clip_limit}, tile_size={tile_size})...")
        
        # Clean input: remove NaN and inf values
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure valid range
        img_min = np.min(image)
        img_max = np.max(image)
        
        if img_max == img_min:
            print("Warning: Image has no contrast, skipping CLAHE")
            return image
        
        # Normalize to 0-255 range safely
        img_normalized = ((image - img_min) / (img_max - img_min) * 255.0)
        img_8bit = np.clip(img_normalized, 0, 255).astype(np.uint8)
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        
        # Apply CLAHE
        enhanced = clahe.apply(img_8bit)
        
        # Convert back to float32
        enhanced = enhanced.astype(np.float32)
        
        self.processing_steps.append(f"CLAHE enhancement (clip={clip_limit})")
        return enhanced
    
    def unsharp_mask(self, image, sigma=1.0, amount=1.5, iterations=1):
        """
        Apply unsharp masking for edge enhancement.
        I_sharp = I + α * (I - Gaussian(I, σ))
        
        Args:
            image: Input image
            sigma: Gaussian blur standard deviation
            amount: Sharpening strength (α)
            iterations: Number of sharpening passes
            
        Returns:
            Sharpened image
        """
        print(f"Applying unsharp mask (σ={sigma}, amount={amount}, iterations={iterations})...")
        
        sharpened = image.copy()
        
        for i in range(iterations):
            # Apply Gaussian blur
            blurred = ndimage.gaussian_filter(sharpened, sigma=sigma)
            
            # Calculate unsharp mask
            mask = sharpened - blurred
            
            # Add weighted mask to original
            sharpened = sharpened + amount * mask
            
            # Clip values
            sharpened = np.clip(sharpened, 0, np.max(image))
        
        self.processing_steps.append(f"Unsharp mask (σ={sigma}, α={amount}, {iterations}x)")
        return sharpened
    
    def high_pass_filter(self, image, radius=3.0, strength=1.0):
        """
        Apply high-pass filtering for sharpening.
        
        Args:
            image: Input image
            radius: Filter radius
            strength: Sharpening strength
            
        Returns:
            Sharpened image
        """
        print(f"Applying high-pass filter (radius={radius}, strength={strength})...")
        
        # Low-pass filter (Gaussian blur)
        low_pass = ndimage.gaussian_filter(image, sigma=radius)
        
        # High-pass = Original - Low-pass
        high_pass = image - low_pass
        
        # Add high-pass back with strength
        sharpened = image + strength * high_pass
        
        # Clip values
        sharpened = np.clip(sharpened, 0, np.max(image))
        
        self.processing_steps.append(f"High-pass filter (r={radius}, s={strength})")
        return sharpened
    
    def calculate_metrics(self, original, processed):
        """
        Calculate image quality metrics: PSNR and SSIM.
        
        Args:
            original: Original image
            processed: Processed image
            
        Returns:
            Dictionary with PSNR and SSIM values
        """
        # Normalize both images to same range
        orig_norm = cv2.normalize(original, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        proc_norm = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Calculate PSNR
        psnr = peak_signal_noise_ratio(orig_norm, proc_norm)
        
        # Calculate SSIM
        ssim = structural_similarity(orig_norm, proc_norm)
        
        return {'PSNR': psnr, 'SSIM': ssim}
    
    def calculate_snr(self, image):
        """
        Calculate Signal-to-Noise Ratio.
        SNR = mean(signal) / std(noise)
        """
        # Estimate signal as mean of bright regions (top 25%)
        threshold = np.percentile(image, 75)
        signal = np.mean(image[image > threshold])
        
        # Estimate noise from dark regions (bottom 25%)
        noise_threshold = np.percentile(image, 25)
        noise = np.std(image[image < noise_threshold])
        
        snr = signal / noise if noise > 0 else float('inf')
        return snr
    
    def full_enhancement_pipeline(self, image_or_images, 
                                   stack=True,
                                   denoise=True,
                                   enhance_contrast=True,
                                   sharpen=True,
                                   wavelet='db4',
                                   clahe_clip=2.0,
                                   sharpen_sigma=1.0,
                                   sharpen_amount=1.5):
        """
        Complete enhancement pipeline for lunar photography.
        
        Args:
            image_or_images: Single image or list of images
            stack: Enable image stacking
            denoise: Enable wavelet denoising
            enhance_contrast: Enable CLAHE
            sharpen: Enable unsharp masking
            
        Returns:
            Enhanced image
        """
        print("\n" + "="*60)
        print("LUNAR LUMINA - Enhancement Pipeline")
        print("="*60 + "\n")
        
        self.processing_steps = []
        
        # Step 1: Stacking (if multiple images)
        if isinstance(image_or_images, list) and len(image_or_images) > 1 and stack:
            processed = self.stack_images(image_or_images)
        else:
            if isinstance(image_or_images, list):
                processed = image_or_images[0]
            else:
                processed = image_or_images
            
            if isinstance(processed, str) or isinstance(processed, Path):
                img = cv2.imread(str(processed), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    # Try loading as color and converting
                    img = cv2.imread(str(processed))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    else:
                        raise ValueError(f"Failed to load image: {processed}")
                processed = img.astype(np.float32)
        
        # Clean input data
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        
        original = processed.copy()
        
        # Step 2: Wavelet denoising
        if denoise:
            processed = self.wavelet_denoise(processed, wavelet=wavelet)
        
        # Step 3: Contrast enhancement
        if enhance_contrast:
            processed = self.adaptive_contrast_enhancement(processed, clip_limit=clahe_clip)
        
        # Step 4: Sharpening
        if sharpen:
            processed = self.unsharp_mask(processed, sigma=sharpen_sigma, amount=sharpen_amount)
        
        # Final cleanup
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate metrics
        print("\n" + "-"*60)
        print("PROCESSING COMPLETE")
        print("-"*60)
        print("\nProcessing steps applied:")
        for i, step in enumerate(self.processing_steps, 1):
            print(f"  {i}. {step}")
        
        print("\nQuality Metrics:")
        try:
            snr_original = self.calculate_snr(original)
            snr_processed = self.calculate_snr(processed)
            print(f"  SNR (Original):  {snr_original:.2f}")
            print(f"  SNR (Enhanced):  {snr_processed:.2f}")
            if snr_original > 0:
                print(f"  SNR Improvement: {(snr_processed/snr_original - 1)*100:.1f}%")
        except Exception as e:
            print(f"  Could not calculate SNR: {e}")
        
        return processed, original
    
    def visualize_results(self, original, enhanced, save_path=None):
        """
        Create visualization comparing original and enhanced images.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Normalize for display
        orig_display = cv2.normalize(original, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        enh_display = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Original image
        axes[0, 0].imshow(orig_display, cmap='gray')
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Enhanced image
        axes[0, 1].imshow(enh_display, cmap='gray')
        axes[0, 1].set_title('Enhanced Image', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Difference map
        diff = np.abs(enh_display.astype(float) - orig_display.astype(float))
        axes[0, 2].imshow(diff, cmap='hot')
        axes[0, 2].set_title('Difference Map', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Histograms
        axes[1, 0].hist(orig_display.ravel(), bins=256, color='blue', alpha=0.7)
        axes[1, 0].set_title('Original Histogram')
        axes[1, 0].set_xlabel('Pixel Intensity')
        axes[1, 0].set_ylabel('Frequency')
        
        axes[1, 1].hist(enh_display.ravel(), bins=256, color='green', alpha=0.7)
        axes[1, 1].set_title('Enhanced Histogram')
        axes[1, 1].set_xlabel('Pixel Intensity')
        axes[1, 1].set_ylabel('Frequency')
        
        # SNR comparison
        snr_orig = self.calculate_snr(original)
        snr_enh = self.calculate_snr(enhanced)
        
        axes[1, 2].bar(['Original', 'Enhanced'], [snr_orig, snr_enh], 
                       color=['blue', 'green'], alpha=0.7)
        axes[1, 2].set_title('Signal-to-Noise Ratio')
        axes[1, 2].set_ylabel('SNR')
        axes[1, 2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nVisualization saved to: {save_path}")
        
        plt.show()


def select_image_file():
    """Open file dialog to select an image."""
    from tkinter import Tk, filedialog
    
    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    file_path = filedialog.askopenfilename(
        title="Select Lunar Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("TIFF files", "*.tif *.tiff"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path


def select_multiple_images():
    """Open file dialog to select multiple images for stacking."""
    from tkinter import Tk, filedialog
    
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    file_paths = filedialog.askopenfilenames(
        title="Select Multiple Lunar Images for Stacking",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("TIFF files", "*.tif *.tiff"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return list(file_paths)


def interactive_enhancement():
    """Interactive mode with GUI file selection and parameter customization."""
    print("\n" + "="*70)
    print(" "*15 + "LUNAR LUMINA - Interactive Enhancement")
    print("="*70 + "\n")
    
    # Choose single or multiple images
    print("Choose processing mode:")
    print("  1. Single image enhancement")
    print("  2. Multiple images (stacking + enhancement)")
    
    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    # Select images
    if choice == '1':
        print("\nPlease select a lunar image from your PC...")
        image_path = select_image_file()
        
        if not image_path:
            print("No file selected. Exiting.")
            return
        
        print(f"Selected: {image_path}")
        images = image_path
        enable_stack = False
    else:
        print("\nPlease select multiple lunar images for stacking...")
        image_paths = select_multiple_images()
        
        if not image_paths:
            print("No files selected. Exiting.")
            return
        
        print(f"Selected {len(image_paths)} images:")
        for i, path in enumerate(image_paths, 1):
            print(f"  {i}. {Path(path).name}")
        
        images = image_paths
        enable_stack = True
    
    # Configure processing parameters
    print("\n" + "-"*70)
    print("Processing Parameters (press Enter for default values)")
    print("-"*70)
    
    # Denoising
    denoise_input = input("Enable wavelet denoising? (y/n) [default: y]: ").strip().lower()
    denoise = denoise_input != 'n'
    
    wavelet = 'db4'
    if denoise:
        wavelet_input = input("Wavelet type (db4/sym4/coif1) [default: db4]: ").strip()
        if wavelet_input in ['db4', 'sym4', 'coif1']:
            wavelet = wavelet_input
    
    # Contrast enhancement
    contrast_input = input("Enable CLAHE contrast enhancement? (y/n) [default: y]: ").strip().lower()
    enhance_contrast = contrast_input != 'n'
    
    clahe_clip = 2.0
    if enhance_contrast:
        clip_input = input("CLAHE clip limit (1.0-4.0) [default: 2.0]: ").strip()
        try:
            clahe_clip = float(clip_input) if clip_input else 2.0
            clahe_clip = max(1.0, min(4.0, clahe_clip))
        except ValueError:
            clahe_clip = 2.0
    
    # Sharpening
    sharpen_input = input("Enable unsharp mask sharpening? (y/n) [default: y]: ").strip().lower()
    sharpen = sharpen_input != 'n'
    
    sharpen_sigma = 1.0
    sharpen_amount = 1.5
    if sharpen:
        sigma_input = input("Sharpening sigma (0.5-3.0) [default: 1.0]: ").strip()
        try:
            sharpen_sigma = float(sigma_input) if sigma_input else 1.0
            sharpen_sigma = max(0.5, min(3.0, sharpen_sigma))
        except ValueError:
            sharpen_sigma = 1.0
        
        amount_input = input("Sharpening amount (0.5-3.0) [default: 1.5]: ").strip()
        try:
            sharpen_amount = float(amount_input) if amount_input else 1.5
            sharpen_amount = max(0.5, min(3.0, sharpen_amount))
        except ValueError:
            sharpen_amount = 1.5
    
    # Initialize enhancer
    enhancer = LunarLumina()
    
    # Process images
    print("\n" + "="*70)
    print("Starting enhancement process...")
    print("="*70)
    
    enhanced, original = enhancer.full_enhancement_pipeline(
        images,
        stack=enable_stack,
        denoise=denoise,
        enhance_contrast=enhance_contrast,
        sharpen=sharpen,
        wavelet=wavelet,
        clahe_clip=clahe_clip,
        sharpen_sigma=sharpen_sigma,
        sharpen_amount=sharpen_amount
    )
    
    # Save output
    print("\n" + "-"*70)
    print("Saving results...")
    print("-"*70)
    
    # Determine output filename
    if isinstance(images, str):
        base_name = Path(images).stem
        output_dir = Path(images).parent
    else:
        base_name = Path(images[0]).stem + "_stacked"
        output_dir = Path(images[0]).parent
    
    output_path = output_dir / f"{base_name}_enhanced.png"
    viz_path = output_dir / f"{base_name}_comparison.png"
    
    # Save enhanced image
    cv2.imwrite(str(output_path), enhanced)
    print(f"Enhanced image saved: {output_path}")
    
    # Generate and save visualization
    enhancer.visualize_results(original, enhanced, save_path=str(viz_path))
    print(f"Comparison visualization saved: {viz_path}")
    
    print("\n" + "="*70)
    print("Enhancement complete!")
    print("="*70 + "\n")


# Example usage
if __name__ == "__main__":
    # Run interactive mode with GUI file selection
    interactive_enhancement()

