## Depth Pro: Sharp Monocular Metric Depth in Less Than a Second

This software project accompanies the research paper:
**[Depth Pro: Sharp Monocular Metric Depth in Less Than a Second](https://arxiv.org/abs/2410.02073)**, 
*Aleksei Bochkovskii, Amaël Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, and Vladlen Koltun*.

![](data/depth-pro-teaser.jpg)

We present a foundation model for zero-shot metric monocular depth estimation. Our model, Depth Pro, synthesizes high-resolution depth maps with unparalleled sharpness and high-frequency details. The predictions are metric, with absolute scale, without relying on the availability of metadata such as camera intrinsics. And the model is fast, producing a 2.25-megapixel depth map in 0.3 seconds on a standard GPU. These characteristics are enabled by a number of technical contributions, including an efficient multi-scale vision transformer for dense prediction, a training protocol that combines real and synthetic datasets to achieve high metric accuracy alongside fine boundary tracing, dedicated evaluation metrics for boundary accuracy in estimated depth maps, and state-of-the-art focal length estimation from a single image.


The model in this repository is a reference implementation, which has been re-trained. Its performance is close to the model reported in the paper but does not match it exactly.

## Getting Started

### Prerequisites

- **Python 3.10+** (3.12 recommended)
- **CUDA-capable GPU** (optional, but recommended for performance)
- **Docker** (optional, for containerized deployment)

### Installation

We recommend setting up a virtual environment. Using e.g. miniconda, the `depth_pro` package can be installed via:

```bash
conda create -n depth-pro -y python=3.10
conda activate depth-pro

# Install the depth_pro package
pip install -e .

# Install additional dependencies for the API server
pip install fastapi uvicorn[standard] websockets opencv-python-headless ultralytics
```

### Model and Checkpoint Setup

1. **Download Depth Pro checkpoint:**
   ```bash
   source get_pretrained_models.sh   # Downloads to `checkpoints` directory.
   ```
   
   This will download `depth_pro.pt` to the `checkpoints/` directory. The API server expects the checkpoint at `checkpoints/depth_pro_checkpoint.pt`, so you may need to rename or symlink it:
   ```bash
   ln -s checkpoints/depth_pro.pt checkpoints/depth_pro_checkpoint.pt
   ```
   
   Alternatively, set the `DEPTH_CHECKPOINT_PATH` environment variable to point to your checkpoint.

2. **YOLOv8 model:**
   The YOLOv8n model (`yolov8n.pt`) will be automatically downloaded on first use. You can also set a custom path using the `YOLO_MODEL_PATH` environment variable.

### Running Locally

#### Command Line Tool (Original)

We provide a helper script to directly run the model on a single image:
```bash
# Run prediction on a single image:
depth-pro-run -i ./data/example.jpg
# Run `depth-pro-run -h` for available options.
```

#### FastAPI Server (New)

Run the FastAPI WebSocket server for real-time video processing:

```bash
# Ensure models are downloaded (see Model and Checkpoint Setup above)
# Run the server
python api_server.py
```

Or with uvicorn directly:
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000` with the following endpoints:
- `GET /` - Server status
- `GET /health` - Health check
- `WS /video/in` - WebSocket for sending video frames (binary JPEG)
- `WS /video/out` - WebSocket for receiving processed frames (binary JPEG)
- `WS /tts` - WebSocket for receiving text-to-speech alerts

#### Environment Variables

- `YOLO_MODEL_PATH` - Path to YOLOv8 model (default: `yolov8n.pt`)
- `DEPTH_CHECKPOINT_PATH` - Path to Depth Pro checkpoint (default: `checkpoints/depth_pro_checkpoint.pt`)
- `METRIC_ALERT_THRESHOLD` - Distance threshold in meters for alerts (default: 1.5)
- `ALLOWED_ORIGINS` - Comma-separated list of allowed CORS origins (default: `*` for all origins). For production, set to specific domains like `http://localhost:3000,https://yourdomain.com`

### Running with Docker

1. **Build the Docker image:**
   ```bash
   docker build -t depth-pro-api .
   ```

2. **Run the container:**
   
   First, ensure you have the Depth Pro checkpoint downloaded. Then run:
   ```bash
   docker run -p 8000:8000 \
     -v $(pwd)/checkpoints:/app/checkpoints \
     depth-pro-api
   ```

   For GPU support (requires nvidia-docker):
   ```bash
   docker run --gpus all -p 8000:8000 \
     -v $(pwd)/checkpoints:/app/checkpoints \
     depth-pro-api
   ```

3. **Access the server:**
   The API will be available at `http://localhost:8000`

### WebSocket Usage Examples

#### Sending Video Frames (Python)

```python
import asyncio
import websockets
import cv2

async def send_frames():
    uri = "ws://localhost:8000/video/in"
    async with websockets.connect(uri) as websocket:
        cap = cv2.VideoCapture(0)  # Open webcam
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Send to server
            await websocket.send(frame_bytes)
            await asyncio.sleep(0.033)  # ~30 FPS

asyncio.run(send_frames())
```

#### Receiving Processed Frames (Python)

```python
import asyncio
import websockets
import cv2
import numpy as np

async def receive_frames():
    uri = "ws://localhost:8000/video/out"
    async with websockets.connect(uri) as websocket:
        while True:
            # Receive processed frame
            frame_bytes = await websocket.recv()
            
            # Decode and display
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imshow('Processed Frame', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

asyncio.run(receive_frames())
```

#### Receiving TTS Alerts (Python)

```python
import asyncio
import websockets

async def receive_alerts():
    uri = "ws://localhost:8000/tts"
    async with websockets.connect(uri) as websocket:
        while True:
            alert = await websocket.recv()
            print(f"Alert: {alert}")

asyncio.run(receive_alerts())
```

### Python API Usage (Original)

```python
from PIL import Image
import depth_pro

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Load and preprocess an image.
image, _, f_px = depth_pro.load_rgb(image_path)
image = transform(image)

# Run inference.
prediction = model.infer(image, f_px=f_px)
depth = prediction["depth"]  # Depth in [m].
focallength_px = prediction["focallength_px"]  # Focal length in pixels.
```

### Notes on Focal Length (f_px)

The Depth Pro model can estimate focal length using its FOV (Field of View) head when not provided. However, for production use with calibrated cameras, providing the actual focal length in pixels (`f_px`) will yield more accurate metric depth estimates. The focal length can be calculated from camera intrinsics or EXIF data.


### Evaluation (boundary metrics) 

Our boundary metrics can be found under `eval/boundary_metrics.py` and used as follows:

```python
# for a depth-based dataset
boundary_f1 = SI_boundary_F1(predicted_depth, target_depth)

# for a mask-based dataset (image matting / segmentation) 
boundary_recall = SI_boundary_Recall(predicted_depth, target_mask)
```


## Citation

If you find our work useful, please cite the following paper:

```bibtex
@inproceedings{Bochkovskii2024:arxiv,
  author     = {Aleksei Bochkovskii and Ama\"{e}l Delaunoy and Hugo Germain and Marcel Santos and
               Yichao Zhou and Stephan R. Richter and Vladlen Koltun},
  title      = {Depth Pro: Sharp Monocular Metric Depth in Less Than a Second},
  booktitle  = {International Conference on Learning Representations},
  year       = {2025},
  url        = {https://arxiv.org/abs/2410.02073},
}
```

## License
This sample code is released under the [LICENSE](LICENSE) terms.

The model weights are released under the [LICENSE](LICENSE) terms.

## Acknowledgements

Our codebase is built using multiple opensource contributions, please see [Acknowledgements](ACKNOWLEDGEMENTS.md) for more details.

Please check the paper for a complete list of references and datasets used in this work.
