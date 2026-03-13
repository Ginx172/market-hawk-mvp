"""
PAUSABLE Video Frame Extraction — Checkpoint/Resume Support
Extracts key frames from 3,623 trading videos with scene change detection.
Handles Ctrl+C gracefully, saves progress, resumes from last checkpoint.
"""
import cv2
import json
import pickle
import platform
from pathlib import Path
from datetime import datetime
import logging
import signal
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('extraction.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


class PausableVideoFrameExtractor:
    """Extract key frames with checkpoint/resume support."""

    def __init__(self):
        # Auto-detect OS path
        if platform.system() == 'Windows':
            self.j_drive = Path(r'J:\E-Books\.....Trading Database')
            self.output_dir = Path(r'K:\_DEV_MVP_2026\Market_Hawk_3\extracted_content\video_frames')
            self.checkpoint = Path(r'K:\_DEV_MVP_2026\Market_Hawk_3\extraction_checkpoint.pkl')
            self.progress_file = Path(r'K:\_DEV_MVP_2026\Market_Hawk_3\extraction_progress.json')
        else:  # WSL / Linux
            self.j_drive = Path('/mnt/j/E-Books/.....Trading Database')
            self.output_dir = Path('/mnt/k/_DEV_MVP_2026/Market_Hawk_3/extracted_content/video_frames')
            self.checkpoint = Path('/mnt/k/_DEV_MVP_2026/Market_Hawk_3/extraction_checkpoint.pkl')
            self.progress_file = Path('/mnt/k/_DEV_MVP_2026/Market_Hawk_3/extraction_progress.json')

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scene_change_threshold = 20.0
        self.processed_videos: set = set()
        self.total_frames_extracted: int = 0
        self.load_checkpoint()
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        logger.info("\n⏸️  PAUSING... Saving checkpoint...")
        self.save_checkpoint()
        logger.info("✅ Checkpoint saved! Resume with: python scripts/pausable_video_extraction.py")
        sys.exit(0)

    def save_checkpoint(self):
        with open(self.checkpoint, 'wb') as f:
            pickle.dump({
                'processed': self.processed_videos,
                'total_frames': self.total_frames_extracted,
                'timestamp': datetime.now().isoformat()
            }, f)
        with open(self.progress_file, 'w') as f:
            json.dump({
                'processed_count': len(self.processed_videos),
                'total_frames': self.total_frames_extracted,
                'last_saved': datetime.now().isoformat()
            }, f, indent=2)
        logger.info(f"✅ Checkpoint: {len(self.processed_videos)} videos, {self.total_frames_extracted} frames")

    def load_checkpoint(self):
        if self.checkpoint.exists():
            with open(self.checkpoint, 'rb') as f:
                data = pickle.load(f)
                self.processed_videos = data['processed']
                self.total_frames_extracted = data['total_frames']
                logger.info(f"📂 Resuming from checkpoint: {len(self.processed_videos)} videos done")
        else:
            logger.info("🆕 Fresh start — no checkpoint found")

    def extract_frames(self):
        mp4_files = sorted(list(self.j_drive.rglob('*.mp4')))
        total = len(mp4_files)
        remaining = [f for f in mp4_files if f.name not in self.processed_videos]

        logger.info(f"Found {total} MP4s | Done: {len(self.processed_videos)} | Remaining: {len(remaining)}\n")

        for idx, video_file in enumerate(remaining, 1):
            try:
                pos = len(self.processed_videos) + idx
                logger.info(f"[{pos}/{total}] {video_file.name}")

                cap = cv2.VideoCapture(str(video_file))
                if not cap.isOpened():
                    logger.warning(f"⚠️ Cannot open: {video_file.name}")
                    self.processed_videos.add(video_file.name)
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS)
                fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = fc / fps if fps > 0 else 0

                prev_frame, key_frames, frame_idx = None, [], 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if prev_frame is not None:
                        diff = cv2.absdiff(frame, prev_frame)
                        if (diff.mean() / 255.0) * 100 > self.scene_change_threshold:
                            key_frames.append(frame_idx)
                    prev_frame = frame.copy()
                    frame_idx += 1
                cap.release()

                # Save key frames
                out_dir = self.output_dir / video_file.stem
                out_dir.mkdir(parents=True, exist_ok=True)
                cap = cv2.VideoCapture(str(video_file))
                for i, fn in enumerate(key_frames):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
                    ret, frame = cap.read()
                    if ret:
                        cv2.imwrite(str(out_dir / f"frame_{i:04d}.jpg"), frame)
                cap.release()

                self.processed_videos.add(video_file.name)
                self.total_frames_extracted += len(key_frames)
                logger.info(f"✅ {len(key_frames)} frames ({duration:.0f}s)")

                if len(self.processed_videos) % 5 == 0:
                    self.save_checkpoint()

            except Exception as e:
                logger.error(f"❌ {video_file.name}: {e}")
                self.processed_videos.add(video_file.name)
                continue

        self.save_checkpoint()
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ DONE! {len(self.processed_videos)} videos, {self.total_frames_extracted} frames")


if __name__ == '__main__':
    PausableVideoFrameExtractor().extract_frames()
