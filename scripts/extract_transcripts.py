#!/usr/bin/env python3
"""
Extract & Parse Video Transcripts (SRT/VTT files)
Convert to structured JSON with timestamps
"""

import json
import re
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class TranscriptExtractor:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.j_drive = Path('J:\\E-Books\\.....Trading Database')
        self.output_dir = Path('K:\\_DEV_MVP_2026\\Market_Hawk_3\\extracted_content\\transcripts')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def parse_srt(self, content):
        """Parse SRT subtitle format"""
        subtitles = []
        blocks = content.strip().split('\n\n')
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                try:
                    # Format: index, timecode, text
                    timecode = lines[1]
                    text = ' '.join(lines[2:])
                    
                    start, end = timecode.split(' --> ')
                    subtitles.append({
                        'start': start.strip(),
                        'end': end.strip(),
                        'text': text.strip()
                    })
                except:
                    pass
        
        return subtitles
    
    def parse_vtt(self, content):
        """Parse VTT subtitle format"""
        subtitles = []
        lines = content.strip().split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for timecode line
            if ' --> ' in line:
                timecode = line
                text_lines = []
                i += 1
                
                while i < len(lines) and lines[i].strip() and ' --> ' not in lines[i]:
                    text_lines.append(lines[i].strip())
                    i += 1
                
                try:
                    start, end = timecode.split(' --> ')
                    subtitles.append({
                        'start': start.strip(),
                        'end': end.strip(),
                        'text': ' '.join(text_lines)
                    })
                except:
                    pass
            else:
                i += 1
        
        return subtitles
    
    def timecode_to_seconds(self, timecode):
        """Convert HH:MM:SS,mmm to seconds"""
        try:
            parts = timecode.replace(',', '.').split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        except:
            return 0
    
    def extract_transcripts(self):
        """Extract all SRT/VTT files from J: drive"""
        
        logger.info(f"Scanning {self.j_drive} for subtitle files...")
        
        transcript_count = 0
        total_words = 0
        transcript_data = []
        
        # Find all SRT and VTT files
        for subtitle_file in self.j_drive.rglob('*'):
            if subtitle_file.suffix.lower() not in ['.srt', '.vtt']:
                continue
            
            try:
                logger.info(f"Processing: {subtitle_file.name}")
                
                with open(subtitle_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Determine format and parse
                if subtitle_file.suffix.lower() == '.srt':
                    subtitles = self.parse_srt(content)
                else:
                    subtitles = self.parse_vtt(content)
                
                if not subtitles:
                    continue
                
                # Create transcript entry
                transcript = {
                    'filename': subtitle_file.name,
                    'source_path': str(subtitle_file),
                    'format': subtitle_file.suffix.lower(),
                    'extracted_at': self.timestamp,
                    'subtitle_count': len(subtitles),
                    'duration_seconds': self.timecode_to_seconds(subtitles[-1]['end']) if subtitles else 0,
                    'full_text': ' '.join([s['text'] for s in subtitles]),
                    'subtitles': subtitles
                }
                
                # Stats
                words = len(transcript['full_text'].split())
                total_words += words
                transcript['word_count'] = words
                
                transcript_data.append(transcript)
                transcript_count += 1
                
                logger.info(f"✅ Extracted: {subtitle_file.name} ({len(subtitles)} subs, {words} words)")
                
            except Exception as e:
                logger.error(f"❌ Error processing {subtitle_file.name}: {e}")
                continue
        
        return transcript_data, transcript_count, total_words
    
    def generate_transcript_index(self, transcripts):
        """Generate searchable index of all transcripts"""
        
        index = {
            'timestamp': self.timestamp,
            'total_transcripts': len(transcripts),
            'total_words': sum(t['word_count'] for t in transcripts),
            'total_subtitles': sum(t['subtitle_count'] for t in transcripts),
            'avg_duration_minutes': sum(t['duration_seconds'] for t in transcripts) / 60 / len(transcripts) if transcripts else 0,
            'transcripts': []
        }
        
        # Create index entries (without full text for efficiency)
        for transcript in transcripts:
            index['transcripts'].append({
                'filename': transcript['filename'],
                'format': transcript['format'],
                'subtitle_count': transcript['subtitle_count'],
                'word_count': transcript['word_count'],
                'duration_minutes': transcript['duration_seconds'] / 60,
                'keywords': self.extract_keywords(transcript['full_text'])
            })
        
        return index
    
    def extract_keywords(self, text, top_n=10):
        """Extract top keywords from text"""
        words = text.lower().split()
        
        # Filter stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'}
        
        filtered = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Count frequency
        word_freq = {}
        for w in filtered:
            word_freq[w] = word_freq.get(w, 0) + 1
        
        # Top N
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [w[0] for w in top_words]
    
    def save_results(self, transcripts, index):
        """Save extracted transcripts and index"""
        
        # Save full transcripts (JSONL format - one per line)
        transcripts_file = self.output_dir / 'transcripts_full.jsonl'
        with open(transcripts_file, 'w') as f:
            for transcript in transcripts:
                f.write(json.dumps(transcript) + '\n')
        
        logger.info(f"✅ Saved full transcripts: {transcripts_file}")
        
        # Save index (for quick search)
        index_file = self.output_dir / 'transcripts_index.json'
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)
        
        logger.info(f"✅ Saved index: {index_file}")
        
        # Save summary report
        report_file = self.output_dir / 'TRANSCRIPTS_EXTRACTION_REPORT.md'
        with open(report_file, 'w') as f:
            f.write("# Transcript Extraction Report\n\n")
            f.write(f"**Extraction Date:** {self.timestamp}\n\n")
            
            f.write("## 📊 SUMMARY\n\n")
            f.write(f"- **Total Transcripts Extracted:** {index['total_transcripts']}\n")
            f.write(f"- **Total Words:** {index['total_words']:,}\n")
            f.write(f"- **Total Subtitles:** {index['total_subtitles']:,}\n")
            f.write(f"- **Average Duration:** {index['avg_duration_minutes']:.1f} minutes\n")
            f.write(f"- **Total Duration:** {sum(t['duration_seconds'] for t in transcripts) / 3600:.1f} hours\n\n")
            
            f.write("## 📝 TOP KEYWORDS\n\n")
            all_keywords = {}
            for t in index['transcripts']:
                for kw in t['keywords']:
                    all_keywords[kw] = all_keywords.get(kw, 0) + 1
            
            top_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)[:20]
            for kw, count in top_keywords:
                f.write(f"- `{kw}`: {count} transcripts\n")
            f.write("\n")
            
            f.write("## 📄 TRANSCRIPT LIST\n\n")
            for t in sorted(index['transcripts'], key=lambda x: x['word_count'], reverse=True)[:50]:
                f.write(f"- **{t['filename']}** | {t['word_count']} words | {t['subtitle_count']} subs\n")
        
        logger.info(f"✅ Saved report: {report_file}")

def main():
    extractor = TranscriptExtractor()
    
    print(f"\n{'='*120}")
    print(f"📝 EXTRACTING VIDEO TRANSCRIPTS")
    print(f"{'='*120}\n")
    
    # Extract transcripts
    transcripts, count, total_words = extractor.extract_transcripts()
    
    if transcripts:
        # Generate index
        index = extractor.generate_transcript_index(transcripts)
        
        # Save results
        extractor.save_results(transcripts, index)
        
        print(f"\n{'='*120}")
        print(f"✅ TRANSCRIPT EXTRACTION COMPLETE!")
        print(f"{'='*120}")
        print(f"\n📊 STATISTICS:")
        print(f"   Total Transcripts: {count}")
        print(f"   Total Words: {total_words:,}")
        print(f"   Avg Words/Transcript: {total_words // count if count > 0 else 0}")
        print(f"\n📁 Output Directory: {extractor.output_dir}\n")
    else:
        print("❌ No transcripts found!\n")

if __name__ == "__main__":
    main()