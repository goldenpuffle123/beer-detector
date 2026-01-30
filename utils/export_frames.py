import decord
import numpy as np
from pathlib import Path
import cv2

def export_frames(video_path: str, out_path: str, num_frames: int, num_start: int = 0):
    dir = Path(out_path)
    vr = decord.VideoReader(video_path)
    nums = np.rint(np.linspace(0, len(vr)-1, num_frames)).astype(int)
    batch = vr.get_batch(nums).asnumpy()
    for i, frame in enumerate(batch):
        out = dir / f"im_{i + num_start:04d}.png"
        cv2.imwrite(str(out), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"Exported {len(batch)} frames to {dir}")
    del vr

def main():
    video_paths = Path("videos").rglob("*13_53_Pro.mp4")
    out_path = "frames"
    Path(out_path).mkdir(exist_ok=True)
    num_start = 60
    for v in video_paths:
        export_frames(str(v), out_path, 50, num_start)
        num_start += 50

if __name__ == "__main__":
    main()