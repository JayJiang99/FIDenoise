#!/usr/bin/env python3
"""
FIDenoise Inference Script

This script performs denoising inference using a pre-trained RIFE model
and evaluates the results using PSNR and SSIM metrics.
"""

import os
import cv2
import torch
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from torch.nn import functional as F
import warnings
from model.RIFE import Model
import numpy as np
from scipy.signal import convolve2d as conv2
from skimage import restoration
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from dataset import OCTDenoisingDatasetTest
from torch.utils.data import DataLoader

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DenoisingInference:
    """Class for handling denoising inference operations."""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the DenoisingInference class.
        
        Args:
            model_path: Path to the pre-trained model
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """
        Setup the device for inference.
        
        Args:
            device: Device specification
            
        Returns:
            torch.device: The device to use for inference
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        device_obj = torch.device(device)
        
        if device_obj.type == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                device_obj = torch.device("cpu")
            else:
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        
        logger.info(f"Using device: {device_obj}")
        return device_obj
    
    def _load_model(self, model_path: str) -> Model:
        """
        Load the pre-trained model.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Model: Loaded model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = Model()
        model.load_model(path=model_path)
        model.eval()
        model.device()
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    
    @staticmethod
    def deconv_richardson_lucy(img: np.ndarray, num_iter: int = 10) -> np.ndarray:
        """
        Perform Richardson-Lucy deconvolution.
        
        Args:
            img: Input image
            num_iter: Number of iterations
            
        Returns:
            np.ndarray: Deconvolved image
        """
        # Convert to grayscale by taking one channel
        image_gray = img
        image_gray = (image_gray - image_gray.min()) / (image_gray.max() - image_gray.min())
        
        # Create a simple 7x7 blurring kernel
        psf = np.zeros((7, 7))
        psf[3, 2] = 0.1
        psf[3, 3] = 0.2
        psf[3, 4] = 0.4
        psf[3, 5] = 0.2
        psf[3, 6] = 0.1
        psf /= psf.sum()

        # Perform Richardson-Lucy deconvolution
        deconvolved = restoration.richardson_lucy(image_gray, psf, num_iter=num_iter)
        return deconvolved * 255.0
    
    @staticmethod
    def flow2rgb(flow_map_np: np.ndarray) -> np.ndarray:
        """
        Convert flow map to RGB visualization.
        
        Args:
            flow_map_np: Flow map array
            
        Returns:
            np.ndarray: RGB visualization
        """
        h, w, _ = flow_map_np.shape
        rgb_map = np.ones((h, w, 3)).astype(np.float32)
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
        rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
        rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
        rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
        return rgb_map.clip(0, 1)

    @staticmethod
    def interleave_tensors(t1: torch.Tensor, t2: torch.Tensor, t3: torch.Tensor) -> torch.Tensor:
        """
        Interleave three tensors along the width dimension.
        
        Args:
            t1, t2, t3: Input tensors of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Interleaved tensor
        """
        B, C, H, W = t1.shape
        result = torch.zeros(B, C, H, W * 3, device=t1.device)
        result[:, :, :, 0::3] = t1
        result[:, :, :, 1::3] = t2
        result[:, :, :, 2::3] = t3
        return result

    def preprocess_images(self, noisy_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preprocess noisy images for inference.
        
        Args:
            noisy_img: Input noisy image tensor
            
        Returns:
            Tuple of preprocessed image tensors
        """
        noisy_img0 = noisy_img[:, :, :, 0::3]
        noisy_img1 = noisy_img[:, :, :, 1::3]
        noisy_img2 = noisy_img[:, :, :, 2::3]
        
        new_img0 = torch.cat([noisy_img0[:, :, :, 1:], noisy_img0[:, :, :, -1:]], 3)
        new_img2 = torch.cat([noisy_img2[:, :, :, :1], noisy_img2[:, :, :, :-1]], 3)
        
        return noisy_img0, noisy_img1, noisy_img2, new_img0, new_img2
    
    def run_inference(self, noisy_img0: torch.Tensor, noisy_img1: torch.Tensor, 
                     noisy_img2: torch.Tensor, new_img0: torch.Tensor, 
                     new_img2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run denoising inference on the input images.
        
        Args:
            noisy_img0, noisy_img1, noisy_img2: Input noisy images
            new_img0, new_img2: Shifted images
            
        Returns:
            Tuple of denoised predictions
        """
        with torch.no_grad():
            pred1, teacher1, upsampled_merged1, flow1 = self.model.inference_denoise(
                noisy_img0, noisy_img2, noisy_img1
            )
            pred2, teacher2, upsampled_merged2, _ = self.model.inference_denoise(
                noisy_img1, new_img0, noisy_img2
            )
            pred0, teacher0, upsampled_merged0, _ = self.model.inference_denoise(
                new_img2, noisy_img1, noisy_img0
            )
        
        return pred0, pred1, pred2, upsampled_merged0, upsampled_merged1, upsampled_merged2
    
    def postprocess_predictions(self, pred0: torch.Tensor, pred1: torch.Tensor, 
                              pred2: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Postprocess model predictions for evaluation.
        
        Args:
            pred0, pred1, pred2: Model predictions
            
        Returns:
            Tuple of postprocessed predictions as numpy arrays
        """
        # Convert to numpy and reshape
        pred0_np = pred0[0].permute(1, 2, 0).detach().cpu().numpy()
        pred1_np = pred1[0].permute(1, 2, 0).detach().cpu().numpy()
        pred2_np = pred2[0].permute(1, 2, 0).detach().cpu().numpy()
        
        # Convert to grayscale by averaging channels
        pred0_gray = np.mean(pred0_np, axis=2)
        pred1_gray = np.mean(pred1_np, axis=2)
        pred2_gray = np.mean(pred2_np, axis=2)

        # Convert to uint8
        pred0_uint8 = (pred0_gray * 255).astype(np.uint8)
        pred1_uint8 = (pred1_gray * 255).astype(np.uint8)
        pred2_uint8 = (pred2_gray * 255).astype(np.uint8)
        
        return pred0_uint8, pred1_uint8, pred2_uint8
    
    def postprocess_upsampled_predictions(self, upsampled_merged0: torch.Tensor, upsampled_merged1: torch.Tensor, 
                                         upsampled_merged2: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Postprocess upsampled predictions for evaluation.
        """
        upsampled_merged0_np = upsampled_merged0[0].permute(1, 2, 0).detach().cpu().numpy()
        upsampled_merged1_np = upsampled_merged1[0].permute(1, 2, 0).detach().cpu().numpy()
        upsampled_merged2_np = upsampled_merged2[0].permute(1, 2, 0).detach().cpu().numpy()
        
        upsampled_merged0_gray = np.mean(upsampled_merged0_np, axis=2)
        upsampled_merged1_gray = np.mean(upsampled_merged1_np, axis=2)
        upsampled_merged2_gray = np.mean(upsampled_merged2_np, axis=2)
        
        upsampled_merged0_uint8 = (upsampled_merged0_gray * 255).astype(np.uint8)
        upsampled_merged1_uint8 = (upsampled_merged1_gray * 255).astype(np.uint8)
        upsampled_merged2_uint8 = (upsampled_merged2_gray * 255).astype(np.uint8)
        
        return upsampled_merged0_uint8, upsampled_merged1_uint8, upsampled_merged2_uint8
    
    def save_predictions(self, pred0: np.ndarray, pred1: np.ndarray, pred2: np.ndarray, 
                        clean_img: np.ndarray, output_dir: str = ".") -> None:
        """
        Save predictions and clean image.
        
        Args:
            pred0, pred1, pred2: Predictions to save
            clean_img: Clean reference image
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        cv2.imwrite(str(output_path / 'pred0.png'), pred0)
        cv2.imwrite(str(output_path / 'pred1.png'), pred1)
        cv2.imwrite(str(output_path / 'pred2.png'), pred2)
        cv2.imwrite(str(output_path / 'clean.png'), clean_img)
        
        logger.info(f"Predictions saved to {output_path}")
    
    def calculate_metrics(self, clean_img: np.ndarray, pred0: np.ndarray, 
                         pred1: np.ndarray, pred2: np.ndarray) -> Tuple[List[float], List[float]]:
        """
        Calculate PSNR and SSIM metrics.
        
        Args:
            clean_img: Clean reference image
            pred0, pred1, pred2: Predictions to evaluate
            
        Returns:
            Tuple of PSNR and SSIM lists
        """
        # Resize predictions to match clean image dimensions
        pred0_resized = cv2.resize(pred0, (clean_img.shape[1], clean_img.shape[0]))
        pred1_resized = cv2.resize(pred1, (clean_img.shape[1], clean_img.shape[0]))
        pred2_resized = cv2.resize(pred2, (clean_img.shape[1], clean_img.shape[0]))
        
        # Calculate metrics
        psnr0 = peak_signal_noise_ratio(clean_img, pred0_resized)
        psnr1 = peak_signal_noise_ratio(clean_img, pred1_resized)
        psnr2 = peak_signal_noise_ratio(clean_img, pred2_resized)
        
        ssim0 = structural_similarity(clean_img, pred0_resized)
        ssim1 = structural_similarity(clean_img, pred1_resized)
        ssim2 = structural_similarity(clean_img, pred2_resized)
        
        return [psnr0, psnr1, psnr2], [ssim0, ssim1, ssim2]
    
    def evaluate_dataset(self, clean_img_path: str, noisy_img_path: str, 
                        crop_size: int = 480, batch_size: int = 8, 
                        num_workers: int = 4, output_dir: str = ".") -> Tuple[List[float], List[float]]:
        """
        Evaluate the entire dataset.
        
        Args:
            clean_img_path: Path to clean images
            noisy_img_path: Path to noisy images
            crop_size: Size for cropping
            batch_size: Batch size for evaluation
            num_workers: Number of workers for data loading
            output_dir: Output directory for saving results
            
        Returns:
            Tuple of PSNR and SSIM results
        """
        # Create dataset and dataloader
        dataset = OCTDenoisingDatasetTest(clean_img_path, noisy_img_path, crop_size=crop_size)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers, pin_memory=True)
        
        all_psnr = []
        all_ssim = []
        
        logger.info(f"Starting evaluation on {len(dataset)} images")
        
        for batch_idx, data in enumerate(val_loader):
            clean_img, noisy_img = data
            clean_img = clean_img.to(self.device)
            noisy_img = noisy_img.to(self.device)
            
            # Preprocess images
            noisy_img0, noisy_img1, noisy_img2, new_img0, new_img2 = self.preprocess_images(noisy_img)
            
            # Run inference
            pred0, pred1, pred2, upsampled_merged0, upsampled_merged1, upsampled_merged2 = self.run_inference(noisy_img0, noisy_img1, noisy_img2, new_img0, new_img2)
            
            # Postprocess predictions
            pred0_uint8, pred1_uint8, pred2_uint8 = self.postprocess_predictions(pred0, pred1, pred2)
            upsampled_merged0_uint8, upsampled_merged1_uint8, upsampled_merged2_uint8 = self.postprocess_upsampled_predictions(upsampled_merged0, upsampled_merged1, upsampled_merged2)
            
            # Process clean image
            clean_img_np = clean_img[0].permute(1, 2, 0).detach().cpu().numpy()
            clean_img_gray = np.mean(clean_img_np, axis=2)
            clean_img_uint8 = (clean_img_gray * 255).astype(np.uint8)
            
            # Save first batch predictions
            if batch_idx == 0:
                self.save_predictions(pred0_uint8, pred1_uint8, pred2_uint8, 
                                    clean_img_uint8, output_dir)
            
            # Calculate metrics
            psnr_values, ssim_values = self.calculate_metrics(
                clean_img_uint8, upsampled_merged0_uint8, upsampled_merged1_uint8, upsampled_merged2_uint8
            )
            
            all_psnr.append(psnr_values)
            all_ssim.append(ssim_values)
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(val_loader)} batches")
        
        return all_psnr, all_ssim
    
    def print_results(self, all_psnr: List[List[float]], all_ssim: List[List[float]]) -> None:
        """
        Print evaluation results.
        
        Args:
            all_psnr: List of PSNR values for each prediction
            all_ssim: List of SSIM values for each prediction
        """
        # Convert to numpy arrays for easier calculation
        psnr_array = np.array(all_psnr)
        ssim_array = np.array(all_ssim)
        
        logger.info("=== Evaluation Results ===")
        logger.info(f"PSNR - Pred0: {np.mean(psnr_array[:, 0]):.4f} ± {np.std(psnr_array[:, 0]):.4f}")
        logger.info(f"PSNR - Pred1: {np.mean(psnr_array[:, 1]):.4f} ± {np.std(psnr_array[:, 1]):.4f}")
        logger.info(f"PSNR - Pred2: {np.mean(psnr_array[:, 2]):.4f} ± {np.std(psnr_array[:, 2]):.4f}")
        logger.info(f"SSIM - Pred0: {np.mean(ssim_array[:, 0]):.4f} ± {np.std(ssim_array[:, 0]):.4f}")
        logger.info(f"SSIM - Pred1: {np.mean(ssim_array[:, 1]):.4f} ± {np.std(ssim_array[:, 1]):.4f}")
        logger.info(f"SSIM - Pred2: {np.mean(ssim_array[:, 2]):.4f} ± {np.std(ssim_array[:, 2]):.4f}")


def main():
    """Main function to run the denoising inference."""
    parser = argparse.ArgumentParser(description="FIDenoise Inference Script")
    parser.add_argument("--model_path", type=str, 
                       default="FIDenoise-main/train_log/fidenoise_model.pth",
                       help="Path to the pre-trained model")
    parser.add_argument("--clean_img_path", type=str,
                       default="../2Dimage_clean_cropped",
                       help="Path to clean images")
    parser.add_argument("--noisy_img_path", type=str,
                       default="../2Dimage_noisy_cropped",
                       help="Path to noisy images")
    parser.add_argument("--crop_size", type=int, default=480, help="Crop size for images")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    try:
        # Initialize inference
        inference = DenoisingInference(args.model_path, args.device)
        
        # Run evaluation
        all_psnr, all_ssim = inference.evaluate_dataset(
            args.clean_img_path, args.noisy_img_path, args.crop_size,
            args.batch_size, args.num_workers, args.output_dir
        )
        
        # Print results
        inference.print_results(all_psnr, all_ssim)
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
    