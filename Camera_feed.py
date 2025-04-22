import torch
import torchvision
import cv2
import numpy as np
from pathlib import Path
from boxmot import BotSort
import os
import threading
import queue
import time

# Global flag to signal threads to stop
stop_event = threading.Event()

# Frame queue to pass data between threads
frame_queue = queue.Queue(maxsize=10)  # Buffer up to 10 frames

def capture_frames(camera_url, frame_queue, stop_event):
    """Thread to capture frames from the IP camera"""
    vid = cv2.VideoCapture(camera_url)
    if not vid.isOpened():
        print(f"Could not open camera stream at {camera_url}")
        stop_event.set()
        return

    while not stop_event.is_set():
        ret, frame = vid.read()
        if not ret:
            print("Failed to grab frame from IP camera. Check connection or stream URL.")
            stop_event.set()
            break
        try:
            # Put frame in queue (non-blocking, drops if full)
            frame_queue.put_nowait(frame)
        except queue.Full:
            # If queue is full, skip this frame to avoid blocking
            pass

    vid.release()
    print("Capture thread stopped")

def process_frames(frame_queue, stop_event, output_path, fps, width, height, device):
    """Thread to process frames with detection and tracking"""
    # Load a pre-trained Faster R-CNN model
    detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    detector.eval().to(device)

    # Initialize the tracker
    tracker = BotSort(
        reid_weights=Path('osnet_x0_25_msmt17.pt'),  # Adjust path if downloaded
        device=device,
        half=torch.cuda.is_available()
    )

    # Video writer for output
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_count = 0
    while not stop_event.is_set() or not frame_queue.empty():
        try:
            # Get frame from queue (timeout to check stop_event)
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue  # No frame available, loop to check stop_event

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processing frame {frame_count}")

        # Convert frame to tensor and move to device
        frame_tensor = torchvision.transforms.functional.to_tensor(frame).to(device)

        # Perform detection
        with torch.no_grad():
            detections = detector([frame_tensor])[0]

        # Filter detections
        confidence_threshold = 0.5
        dets = []
        for i, score in enumerate(detections['scores']):
            if score >= confidence_threshold:
                bbox = detections['boxes'][i].cpu().numpy()
                label = detections['labels'][i].item()
                conf = score.item()
                dets.append([*bbox, conf, label])

        dets = np.array(dets) if len(dets) > 0 else np.empty((0, 6))

        # Update tracker
        res = tracker.update(dets, frame)

        # Plot results
        tracker.plot_results(frame, show_trajectories=True)

        # Display the frame
        cv2.imshow('IP Camera - Object Detection and Tracking', frame)

        # Write to output video
        out.write(frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()

        frame_queue.task_done()  # Mark frame as processed

    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete! Output saved to {output_path}")

# Main execution
if __name__ == "__main__":
    # Device setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"CUDA Available: {torch.cuda.is_available()}, Device Count: {torch.cuda.device_count()}, Current Device: {torch.cuda.current_device() if torch.cuda.is_available() else 'N/A'}")

    # IP camera stream URL
    #camera_url = "rtsp://192.168.1.68:554"  # No credentials provided, adjust if needed
    camera_url =  "rtsp://192.168.74.166:8080/h264_ulaw.sdp"
    # Camera properties (defaults if not detected)
    vid = cv2.VideoCapture(camera_url)
    if not vid.isOpened():
        raise ValueError(f"Could not open camera stream at {camera_url}. Check URL, network, or credentials.")
    fps = vid.get(cv2.CAP_PROP_FPS) or 30
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    vid.release()  # Close initial connection, capture thread will reopen
    print(f"Camera FPS: {fps}, Resolution: {width}x{height}")

    # Output path
    output_path = 'ip_camera_output.mp4'

    # Start threads
    capture_thread = threading.Thread(target=capture_frames, args=(camera_url, frame_queue, stop_event))
    process_thread = threading.Thread(target=process_frames, args=(frame_queue, stop_event, output_path, fps, width, height, device))

    capture_thread.start()
    process_thread.start()

    # Wait for threads to finish
    capture_thread.join()
    process_thread.join()

    print("All threads stopped")