#!/usr/bin/env python3
"""
This script actually RUNS the video processing using your new .py files
"""

from video_processor import VideoChunkProcessor

def main():
    print("🚀 Starting Video Processing...")
    print("=" * 50)
    
    try:
        # Create the processor using your NEW .py files
        processor = VideoChunkProcessor(
            chunks_folder='chunks',
            modifications_json='modifications.json',
            output_video_path='final_video.mp4',
            filters_file='filters.py',      # ✅ NEW - uses the clean .py file
            transitions_file='transitions.py'  # ✅ NEW - uses the clean .py file
        )
        
        # Actually process the chunks
        processor.process_chunks()
        
        print("=" * 50)
        print("🎉 SUCCESS! Video processing completed!")
        print("📹 Your final video is saved as: final_video.mp4")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()