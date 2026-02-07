#!/usr/bin/env python3
"""
Camera demo - verify webcam capture works.

Opens the webcam, captures a few frames, and reports device info.
"""

import sys
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from open_eyes.src.services.camera_capture import CameraCapture


def main():
    print("=== Molt-See Camera Demo ===\n")

    # List available devices
    print("Scanning for cameras...")
    devices = CameraCapture.list_devices()
    if not devices:
        print("No cameras found!")
        sys.exit(1)

    for dev in devices:
        print(f"  Camera {dev['index']}: {dev['resolution']}")

    # Capture a test frame
    print(f"\nOpening camera {devices[0]['index']}...")
    camera = CameraCapture(
        device_index=devices[0]["index"],
        target_fps=1.0,
        warmup_frames=3,
    )

    if not camera.start():
        print("Failed to start camera!")
        sys.exit(1)

    import time
    time.sleep(1.5)  # Wait for first frame

    frame = camera.get_frame()
    if frame is not None:
        h, w = frame.shape[:2]
        print(f"Captured frame: {w}x{h}")
        print(f"Frames captured: {camera.frame_count}")
        print("\nCamera is working!")
    else:
        print("No frame captured!")

    camera.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
