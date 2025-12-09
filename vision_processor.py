# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Vision processor module for YOLOv8 object detection with Depth Pro depth estimation."""

import logging
import os
import warnings
from typing import Optional, Tuple

import cv2
import depth_pro
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# Filter warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

LOGGER = logging.getLogger(__name__)


class VisionProcessor:
    """Vision processor combining YOLOv8 object detection with Depth Pro depth estimation."""

    def __init__(
        self,
        yolo_model_path: Optional[str] = None,
        depth_checkpoint_path: Optional[str] = None,
        metric_alert_threshold: float = 1.5,
        device: Optional[str] = None,
    ):
        """Initialize the VisionProcessor.

        Args:
        ----
            yolo_model_path: Path to YOLOv8 model. Defaults to 'yolov8n.pt'.
            depth_checkpoint_path: Path to Depth Pro checkpoint.
                Defaults to 'checkpoints/depth_pro_checkpoint.pt'.
            metric_alert_threshold: Distance threshold in meters for alerts. Default is 1.5m.
            device: Device to use ('cuda', 'cpu', etc.). Auto-detected if None.

        """
        self.metric_alert_threshold = metric_alert_threshold

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        LOGGER.info(f"Using device: {self.device}")

        # Set default paths
        if yolo_model_path is None:
            yolo_model_path = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")
        if depth_checkpoint_path is None:
            depth_checkpoint_path = os.environ.get(
                "DEPTH_CHECKPOINT_PATH", "checkpoints/depth_pro_checkpoint.pt"
            )

        self.yolo_model_path = yolo_model_path
        self.depth_checkpoint_path = depth_checkpoint_path

        # Load models
        try:
            LOGGER.info(f"Loading YOLOv8 model from {self.yolo_model_path}")
            self.yolo_model = YOLO(self.yolo_model_path)
            LOGGER.info("YOLOv8 model loaded successfully")
        except Exception as e:
            LOGGER.error(f"Failed to load YOLOv8 model: {e}")
            raise

        try:
            LOGGER.info(f"Loading Depth Pro model from {self.depth_checkpoint_path}")
            # Create custom config with the checkpoint path
            from depth_pro.depth_pro import DepthProConfig

            config = DepthProConfig(
                patch_encoder_preset="dinov2l16_384",
                image_encoder_preset="dinov2l16_384",
                checkpoint_uri=self.depth_checkpoint_path,
                decoder_features=256,
                use_fov_head=True,
                fov_encoder_preset="dinov2l16_384",
            )

            # Determine precision based on device
            precision = torch.half if self.device.type == "cuda" else torch.float32

            self.depth_model, self.depth_transform = depth_pro.create_model_and_transforms(
                config=config, device=self.device, precision=precision
            )
            self.depth_model.eval()
            LOGGER.info("Depth Pro model loaded successfully")
        except Exception as e:
            LOGGER.error(f"Failed to load Depth Pro model: {e}")
            raise

    def process_frame(
        self, frame: np.ndarray, f_px: Optional[float] = None
    ) -> Tuple[np.ndarray, Optional[str]]:
        """Process a frame with object detection and depth estimation.

        Args:
        ----
            frame: Input frame as numpy array (BGR format from OpenCV).
            f_px: Optional focal length in pixels. If None, estimated from image.

        Returns:
        -------
            annotated_frame: Frame with bounding boxes and depth annotations.
            tts_alert: Optional TTS alert message if person detected within threshold.

        """
        tts_alert = None

        try:
            # Run YOLOv8 detection
            results = self.yolo_model(frame, verbose=False)

            # Check if any results
            if len(results) == 0 or results[0].boxes is None:
                return frame, None

            result = results[0]
            boxes = result.boxes

            # Get depth estimation
            # Convert BGR to RGB for depth model
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Transform and run depth inference
            image_tensor = self.depth_transform(pil_image)

            # Note on focal length (f_px):
            # If not provided, the model will estimate it using the FOV head.
            # For production use, camera calibration would provide more accurate f_px.
            prediction = self.depth_model.infer(image_tensor, f_px=f_px)
            depth_map = prediction["depth"].cpu().numpy()

            # Ensure depth_map matches frame dimensions
            if depth_map.shape != (frame.shape[0], frame.shape[1]):
                depth_map = cv2.resize(
                    depth_map, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR
                )

            # Process detections
            person_detected_within_threshold = False
            min_distance = float("inf")

            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # Get class name
                class_name = result.names[cls]

                # Check if it's a person (class 0 in COCO dataset)
                is_person = class_name.lower() == "person"

                # Calculate median depth within bounding box
                bbox_depth = depth_map[y1:y2, x1:x2]
                if bbox_depth.size > 0:
                    median_depth = float(np.median(bbox_depth))
                else:
                    median_depth = 0.0

                # Check if person is within alert threshold
                if is_person and median_depth > 0 and median_depth <= self.metric_alert_threshold:
                    person_detected_within_threshold = True
                    min_distance = min(min_distance, median_depth)

                # Annotate frame
                color = (0, 0, 255) if is_person else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Add label with class name, confidence, and depth
                label = f"{class_name} {conf:.2f} {median_depth:.2f}m"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(
                    frame,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    color,
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            # Generate TTS alert if person within threshold
            if person_detected_within_threshold:
                tts_alert = f"Warning: Person detected at {min_distance:.2f} meters"
                LOGGER.info(tts_alert)

        except Exception as e:
            LOGGER.error(f"Error processing frame: {e}")
            # Return original frame on error
            return frame, None

        return frame, tts_alert
