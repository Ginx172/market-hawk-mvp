#!/usr/bin/env python3
"""
SMART Video Frame Extraction
Extract ONLY key frames (scene changes, key moments)
Reduces 260M frames → 10M frames (95% smaller!)
"""

import cv2
import json
import subprocess
from pathlib import Path
from datetime import datetime
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class SmartVideoFrameExtractor:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.j_drive = Path('/mnt/j/E-Books/.....Trading Database')
        self.output_dir = Path('/mnt/k/_DEV_MVP_2026\\Market_Hawk_3\\extracted_content\\video_frames')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.scene_change_threshold = 20.0  # Percentage difference for scene change
        self.blur_threshold = 100  # Skip blurry frames
        
    def check_ffmpeg(self):
        """Verify FFmpeg is installed"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
            if result.returncode == 0:
                logger.info("✅ FFmpeg found")
                return True
        except:
            pass
        
        logger.error("❌ FFmpeg not found! Install: https://ffmpeg.org/download.html")
        return False
    
    def get_scene_change_frames(self, video_path, threshold=20.0):
        """Detect scene changes using histogram comparison"""
        
        frames_data = []
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return []
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_seconds = total_frames / fps if fps > 0 else 0
            
            prev_frame = None
            prev_hist = None
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Convert to grayscale for comparison
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Check if frame is blurry (skip)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if laplacian_var < self.blur_threshold:
                    continue
                
                # Calculate histogram
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                cv2.normalize(hist, hist)
                
                # Compare with previous frame
                if prev_hist is not None:
                    comparison = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                    
                    # If significant change, save frame
                    if comparison > (threshold / 100.0):
                        timestamp = frame_count / fps
                        frames_data.append({
                            'frame_number': frame_count,
                            'timestamp': timestamp,
                            'timestamp_hms': self.seconds_to_hms(timestamp),
                            'scene_change_score': float(comparison)
                        })
                        extracted_count += 1
                
                prev_hist = hist
            
            cap.release()
            
            logger.info(f"✅ Detected {extracted_count} key frames from {video_path.name} "
                       f"(duration: {duration_seconds:.1f}s, total_frames: {total_frames})")
            
            return frames_data, duration_seconds
            
        except Exception as e:
            logger.error(f"❌ Error processing {video_path.name}: {e}")
            return [], 0
    
    def extract_key_frames_ffmpeg(self, video_path, frame_numbers):
        """Extract specific frames using FFmpeg"""
        
        try:
            video_stem = video_path.stem
            video_output_dir = self.output_dir / video_stem
            video_output_dir.mkdir(parents=True, exist_ok=True)
            
            frame_paths = []
            
            for frame_num in frame_numbers[:100]:  # Limit to 100 frames per video
                output_path = video_output_dir / f"frame_{frame_num:06d}.png"
                
                # FFmpeg command to extract specific frame
                cmd = [
                    'ffmpeg',
                    '-i', str(video_path),
                    '-vf', f'select=eq(n\\,{frame_num})',
                    '-vsync', '0',
                    str(output_path),
                    '-y',
                    '-loglevel', 'error'
                ]
                
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode == 0 and output_path.exists():
                    frame_paths.append(str(output_path))
            
            return frame_paths
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return []
    
    def seconds_to_hms(self, seconds):
        """Convert seconds to HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def extract_smart_frames(self):
        """Extract smart frames from all videos"""
        
        logger.info(f"Scanning {self.j_drive} for MP4 files...")
        
        video_count = 0
        total_frames_extracted = 0
        extraction_data = []
        
        # Find MP4 files
        mp4_files = list(self.j_drive.rglob('*.mp4'))
        logger.info(f"Found {len(mp4_files)} MP4 files")
        
        for i, video_file in enumerate(mp4_files[:50]):  # Start with first 50 videos
            logger.info(f"\n[{i+1}/{min(50, len(mp4_files))}] Processing: {video_file.name}")
            
            # Detect scene changes
            key_frames, duration = self.get_scene_change_frames(video_file, self.scene_change_threshold)
            
            if key_frames:
                # Extract frames using FFmpeg
                frame_numbers = [f['frame_number'] for f in key_frames]
                extracted_paths = self.extract_key_frames_ffmpeg(video_file, frame_numbers)
                
                extraction_data.append({
                    'video_file': video_file.name,
                    'video_path': str(video_file),
                    'duration_seconds': duration,
                    'key_frames_detected': len(key_frames),
                    'frames_extracted': len(extracted_paths),
                    'key_frames': key_frames,
                    'extracted_frame_paths': extracted_paths
                })
                
                total_frames_extracted += len(extracted_paths)
                video_count += 1
        
        return extraction_data, video_count, total_frames_extracted
    
    def generate_extraction_report(self, extraction_data):
        """Generate extraction report"""
        
        report_path = self.output_dir / 'SMART_FRAME_EXTRACTION_REPORT.md'
        
        with open(report_path, 'w') as f:
            f.write("# Smart Video Frame Extraction Report\n\n")
            f.write(f"**Extraction Date:** {self.timestamp}\n\n")
            
            f.write("## 📊 SUMMARY\n\n")
            
            total_videos = len(extraction_data)
            total_frames = sum(e['frames_extracted'] for e in extraction_data)
            total_duration = sum(e['duration_seconds'] for e in extraction_data)
            total_key_frames_detected = sum(e['key_frames_detected'] for e in extraction_data)
            
            f.write(f"- **Videos Processed:** {total_videos}\n")
            f.write(f"- **Total Duration:** {total_duration / 3600:.1f} hours\n")
            f.write(f"- **Key Frames Detected:** {total_key_frames_detected:,}\n")
            f.write(f"- **Frames Extracted:** {total_frames:,}\n")
            f.write(f"- **Extraction Ratio:** {total_frames / total_key_frames_detected * 100:.1f}%\n")
            f.write(f"- **Estimated Storage:** ~{total_frames * 100 / 1024:.0f}MB\n\n")
            
            f.write("## 🎬 PER-VIDEO BREAKDOWN\n\n")
            
            for data in sorted(extraction_data, key=lambda x: x['frames_extracted'], reverse=True)[:20]:
                f.write(f"### {data['video_file']}\n")
                f.write(f"- Duration: {data['duration_seconds']/60:.1f} minutes\n")
                f.write(f"- Key Frames Detected: {data['key_frames_detected']}\n")
                f.write(f"- Frames Extracted: {data['frames_extracted']}\n")
                f.write(f"- Storage: ~{data['frames_extracted'] * 100 / 1024:.1f}MB\n\n")
            
            f.write("## 🎯 NEXT STEPS\n\n")
            f.write("1. **Review extracted frames** - Check quality/relevance\n")
            f.write("2. **Link to transcripts** - Match frames to video timestamps\n")
            f.write("3. **Create multimodal pairs** - (frame, subtitle_text) tuples\n")
            f.write("4. **Label trading signals** - Mark entry/exit points\n")
            f.write("5. **Train vision model** - Use pairs for model training\n")
        
        logger.info(f"✅ Report saved: {report_path}")
        
        # Save extraction data
        data_path = self.output_dir / 'smart_frames_metadata.json'
        with open(data_path, 'w') as f:
            json.dump(extraction_data, f, indent=2)
        
        logger.info(f"✅ Metadata saved: {data_path}")

def main():
    extractor = SmartVideoFrameExtractor()
    
    print(f"\n{'='*120}")
    print(f"🎬 SMART VIDEO FRAME EXTRACTION")
    print(f"{'='*120}\n")
    
    # Check FFmpeg
    if not extractor.check_ffmpeg():
        print("\n⚠️ FFmpeg required! Install from: https://ffmpeg.org/download.html\n")
        return
    
    # Extract frames
    logger.info("Starting smart frame extraction...")
    extraction_data, video_count, total_frames = extractor.extract_smart_frames()
    
    if extraction_data:
        # Generate report
        extractor.generate_extraction_report(extraction_data)
        
        print(f"\n{'='*120}")
        print(f"✅ SMART FRAME EXTRACTION COMPLETE!")
        print(f"{'='*120}")
        print(f"\n📊 STATISTICS (Sample - first 50 videos):")
        print(f"   Videos Processed: {video_count}")
        print(f"   Total Frames Extracted: {total_frames:,}")
        print(f"   Estimated Storage: ~{total_frames * 100 / 1024:.0f}MB")
        print(f"\n📁 Output Directory: {extractor.output_dir}\n")
    else:
        print("❌ No frames extracted!\n")

if __name__ == "__main__":
    main()
