#!/usr/bin/env python3
"""
Webcam viewer for Docker/Linux using OpenCV.

This version uses:
- V4L2 backend
- MJPG pixel format
- 640x480 @ 30 FPS request

Usage:
  python3 webcam_view.py
"""

import sys
import cv2


def main():
    print("=" * 60)
    print("  Webcam Viewer")
    print("=" * 60)

    # ORBECC RGB
    camera_index = 4
    req_width = 1920
    req_height = 1080
    # NORMAL WEBCAM
    #camera_index = 0
    #req_width = 640
    #req_height = 480
    req_fps = 30

    print("[TEST] Opening webcam...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_height)
    cap.set(cv2.CAP_PROP_FPS, req_fps)

    if not cap.isOpened():
        print(f"[FAIL] Could not open webcam at index {camera_index}")
        sys.exit(1)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

    print(f"[PASS] Camera opened")
    print(f"[INFO] Resolution: {actual_width}x{actual_height}")
    print(f"[INFO] FPS: {actual_fps:.2f}")
    print(f"[INFO] FOURCC: {fourcc}")
    print("[INFO] Press 'q' to quit")

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("[WARN] Failed to read frame")
            continue

        frame_count += 1

        cv2.putText(
            frame,
            f"Frame: {frame_count}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("[PASS] Webcam closed cleanly")
    sys.exit(0)


if __name__ == "__main__":
    main()
