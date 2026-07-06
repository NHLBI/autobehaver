"""CNN-based chamber validation.

The autocropper can optionally run a small CNN classifier to decide whether each
detected chamber is occupied by a fly. Empty chambers can be skipped to reduce
I/O and downstream compute.

"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn


class ChamberCNN(nn.Module):
    """A simple CNN for classifying chamber images as 'empty' or 'occupied'."""

    def __init__(self) -> None:
        super().__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Second conv block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Third conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Calculate size after convolutions
        feature_size = 64 // 8  # After 3 max pooling layers

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * feature_size * feature_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2),  # 2 classes: empty and occupied
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class ChamberValidator:
    def __init__(self, model_path: Path, norm_params_path: Path, *, base_output_dir: Path):
        """Load the trained CNN model and normalization parameters."""
        self.base_output_dir = Path(base_output_dir)
        self.validation_dir = self.base_output_dir / "validation_results"

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"ChamberValidator using device: {self.device}")

        # Load model
        self.model = ChamberCNN()

        # Load checkpoint and extract model state dict
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            logging.info("Loading from checkpoint file...")
            model_state = checkpoint["model_state_dict"]
            logging.info(
                f"Checkpoint info: Epoch {checkpoint.get('epoch', 'unknown')}, "
                f"Val Loss: {checkpoint.get('val_loss', 'unknown')}"
            )
        else:
            logging.info("Loading from state dict file...")
            model_state = checkpoint

        self.model.load_state_dict(model_state)
        self.model.to(self.device)
        self.model.eval()
        logging.info("Model loaded successfully!")

        # Load normalization parameters
        logging.info("Loading normalization parameters...")
        with Path(norm_params_path).open("r") as f:
            self.norm_params = json.load(f)
        self.mean = torch.tensor(self.norm_params["mean"]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor(self.norm_params["std"]).view(3, 1, 1).to(self.device)
        logging.info(
            f"Normalization parameters: mean={self.norm_params['mean']}, std={self.norm_params['std']}"
        )

    def _get_frame_positions(self, video_path: str | Path, num_frames: int = 5) -> list[int]:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Always include first and last frame, then randomly sample the rest
        frame_positions = [0, total_frames - 1]
        middle_frames = random.sample(range(1, total_frames - 1), num_frames - 2)
        frame_positions.extend(middle_frames)
        return sorted(frame_positions)

    def _extract_frame(self, video_path: str | Path, frame_num: int, x: int, y: int, crop_dim: int) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Failed to read frame {frame_num} from {video_path}")

        # Crop frame
        x1 = max(0, x - crop_dim // 2)
        y1 = max(0, y - crop_dim // 2)
        x2 = min(frame.shape[1], x1 + crop_dim)
        y2 = min(frame.shape[0], y1 + crop_dim)
        cropped = frame[y1:y2, x1:x2]

        # Ensure we got the expected size
        if cropped.shape[:2] != (crop_dim, crop_dim):
            # Pad if necessary
            pad_frame = np.zeros((crop_dim, crop_dim, 3), dtype=np.uint8)
            pad_frame[: cropped.shape[0], : cropped.shape[1]] = cropped
            cropped = pad_frame

        return cropped

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        # Resize to 64x64
        resized = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)

        # Convert BGR to RGB (critical to match training preprocessing)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Convert to tensor and normalize
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self.device)
        tensor = (tensor - self.mean) / self.std

        return tensor

    def validate_chamber(self, video_path: str | Path, chamber_id: str, x: int, y: int, crop_dim: int) -> dict:
        """Validate one chamber by sampling frames and doing majority vote."""
        movie_name = Path(video_path).stem
        frame_positions = self._get_frame_positions(video_path)
        predictions: list[int] = []
        raw_outputs: list[list[float]] = []

        for frame_num in frame_positions:
            try:
                source_crop = self._extract_frame(video_path, frame_num, x, y, crop_dim)
                processed_tensor = self._preprocess_frame(source_crop)

                # Run inference
                with torch.no_grad():
                    output = self.model(processed_tensor.unsqueeze(0))
                    raw_output = output[0].cpu().numpy()
                    _, predicted = output.max(1)
                    prediction = int(predicted.item())

                predictions.append(prediction)
                raw_outputs.append([float(v) for v in raw_output])

            except Exception as e:
                logging.warning(
                    f"Error processing frame {frame_num} for {chamber_id} in {video_path}: {e}"
                )

        # Calculate final decision using majority vote of predictions
        if not predictions:
            logging.warning("No valid frames processed!")
            return {"is_occupied": True, "confidence": 0.0, "frame_predictions": []}

        occupied_count = sum(p == 1 for p in predictions)
        is_occupied = occupied_count > len(predictions) / 2
        confidence = float(occupied_count / len(predictions))

        # Save validation results
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        results_file = self.validation_dir / f"{movie_name}_validation.json"

        validation_results: dict[str, Any] = {}
        validation_results[chamber_id] = {
            "is_occupied": bool(is_occupied),
            "confidence": float(confidence),
            "frame_predictions": [int(p) for p in predictions],
            "raw_outputs": raw_outputs,
            "frame_numbers": [int(x) for x in frame_positions],
            "coordinates": {"x": int(x), "y": int(y), "crop_dim": int(crop_dim)},
        }

        # Update existing validation file if it exists
        if results_file.exists():
            try:
                with results_file.open("r") as f:
                    existing_results = json.load(f)
                existing_results.update(validation_results)
                validation_results = existing_results
            except json.JSONDecodeError:
                pass

        with results_file.open("w") as f:
            json.dump(validation_results, f, indent=4)

        return validation_results[chamber_id]
