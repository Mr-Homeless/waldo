import librosa
import numpy as np
import os
import subprocess
import soundfile as sf
import argparse
import sys
import shutil
import uuid
import time

def get_ffmpeg_path():
    """Finds the absolute path to the ffmpeg executable."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        ffmpeg_path = os.path.join(conda_prefix, "bin", "ffmpeg")
        if os.path.exists(ffmpeg_path):
            return ffmpeg_path
    raise FileNotFoundError("Could not find ffmpeg executable.")

def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration", 
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None

def get_file_size_mb(file_path):
    """Get file size in MB."""
    return os.path.getsize(file_path) / (1024 * 1024)

def extract_audio(video_path, audio_output_path, resample_rate, max_duration=None):
    """Extracts and resamples the first audio stream from a video."""
    ffmpeg_path = get_ffmpeg_path()
    
    # First get video duration for progress calculation
    duration = get_video_duration(video_path)
    
    command = [
        ffmpeg_path,
        '-i', video_path,
        '-map', '0:a:0',
        '-acodec', 'pcm_s16le',  # Use faster PCM codec instead of default
        '-ac', '1',
        '-ar', str(resample_rate),
        '-progress', 'pipe:1',  # Output progress to stdout
        '-stats_period', '1',    # Update every 1 second for faster feedback
        '-threads', '0'          # Use all available CPU threads
    ]
    
    # Add duration limit if specified
    if max_duration:
        command.extend(['-t', str(max_duration)])
        duration = min(duration, max_duration) if duration else max_duration
    
    command.extend(['-y', audio_output_path])
    
    print(f"PROGRESS:0:Starting audio extraction from {os.path.basename(video_path)}...", flush=True)
    sys.stdout.flush()  # Force flush
    
    start_time = time.time()
    
    try:
        # Use Popen for real-time output
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                 text=True, bufsize=1, universal_newlines=True)
        
        last_progress = 0
        while True:
            line = process.stdout.readline()
            if not line:
                break
                
            # Parse ffmpeg progress output
            if 'out_time_ms=' in line:
                try:
                    # Extract time in microseconds
                    time_ms = int(line.split('out_time_ms=')[1].strip()) / 1000000
                    if duration and duration > 0:
                        # Calculate progress percentage (0-10% of total progress)
                        progress = min(int((time_ms / duration) * 10), 10)
                        if progress > last_progress:
                            print(f"PROGRESS:{progress}:Extracting audio... {int((time_ms/duration)*100)}% complete", flush=True)
                            sys.stdout.flush()
                            last_progress = progress
                except (ValueError, IndexError):
                    pass
        
        # Wait for process to complete
        process.wait()
        if process.returncode != 0:
            stderr_output = process.stderr.read()
            raise subprocess.CalledProcessError(process.returncode, command, stderr=stderr_output)
            
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e.stderr}", file=sys.stderr)
        raise
    
    elapsed = time.time() - start_time
    print(f"PROGRESS:10:Audio extraction completed in {elapsed:.1f} seconds.", flush=True)
    sys.stdout.flush()
    return None

def detect_kill_sounds(audio_path, sample_sound_path, resample_rate, cooldown_period):
    """Detects kill sounds in an audio file."""
    print(f"PROGRESS:15:Loading sample kill sound...", flush=True)
    sys.stdout.flush()
    kill_sound, _ = librosa.load(sample_sound_path, sr=resample_rate)
    
    print(f"PROGRESS:20:Loading audio file for analysis...", flush=True)
    sys.stdout.flush()
    
    # Get file size to estimate loading time
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    print(f"PROGRESS:22:Audio file size: {file_size_mb:.1f}MB - this may take a moment...", flush=True)
    sys.stdout.flush()
    
    start_time = time.time()
    audio, _ = librosa.load(audio_path, sr=resample_rate)
    load_time = time.time() - start_time
    print(f"PROGRESS:40:Audio loaded in {load_time:.1f} seconds. Duration: {len(audio)/resample_rate:.1f} seconds", flush=True)
    sys.stdout.flush()

    print("PROGRESS:45:Performing correlation analysis...", flush=True)
    sys.stdout.flush()
    start_time = time.time()
    correlation = np.correlate(audio, kill_sound, mode='valid')
    correlation /= np.max(np.abs(correlation))
    correlation_time = time.time() - start_time
    print(f"PROGRESS:60:Correlation analysis completed in {correlation_time:.1f} seconds.", flush=True)
    sys.stdout.flush()
    
    threshold = 0.8
    detections = np.where(correlation >= threshold)[0]
    timestamps = [(det / resample_rate) for det in detections]

    filtered_timestamps = []
    last_time = -cooldown_period
    for t in timestamps:
        if t - last_time >= cooldown_period:
            filtered_timestamps.append(t)
            last_time = t
    
    if not filtered_timestamps:
        print(f"No unique kill sounds detected in {os.path.basename(audio_path)}.", flush=True)
    else:
        print(f"Detected {len(filtered_timestamps)} unique kill sounds in {os.path.basename(audio_path)}.", flush=True)
    sys.stdout.flush()
    return filtered_timestamps

def extract_clips(video_path, timestamps, output_dir, video_basename):
    """Extracts clips from a video based on timestamps."""
    ffmpeg_path = get_ffmpeg_path()
    os.makedirs(output_dir, exist_ok=True)
    
    total_clips = len(timestamps)
    print(f"PROGRESS:65:Extracting {total_clips} clips...")
    for i, timestamp in enumerate(timestamps):
        # Calculate progress from 65% to 95%
        progress = 65 + int((i / total_clips) * 30)
        start_time = max(0, timestamp - 1.7)
        clip_output_path = os.path.join(output_dir, f'{video_basename}_clip_{i + 1}.mp4')
        command = [
            ffmpeg_path,
            '-ss', str(start_time),
            '-i', video_path,
            '-t', '2',
            '-c', 'copy',
            '-y',
            clip_output_path
        ]
        try:
            subprocess.run(command, shell=False, check=True, capture_output=True, text=True)
            print(f"PROGRESS:{progress}:Extracted clip {i+1}/{total_clips}: {os.path.basename(clip_output_path)}")
        except subprocess.CalledProcessError as e:
            print(f"  Error extracting clip {i+1}: {e}", file=sys.stderr)

def validate_input_files(input_dir, max_file_size_mb=None, max_duration_minutes=None):
    """Validate input files before processing."""
    issues = []
    
    for filename in os.listdir(input_dir):
        if filename.endswith((".mp4", ".mov", ".avi")):
            file_path = os.path.join(input_dir, filename)
            
            # Check file size only if limit is specified
            if max_file_size_mb is not None:
                size_mb = get_file_size_mb(file_path)
                if size_mb > max_file_size_mb:
                    issues.append(f"{filename}: File too large ({size_mb:.1f}MB > {max_file_size_mb}MB)")
            
            # Check duration only if limit is specified
            if max_duration_minutes is not None:
                duration = get_video_duration(file_path)
                if duration and duration > (max_duration_minutes * 60):
                    duration_mins = duration / 60
                    issues.append(f"{filename}: Video too long ({duration_mins:.1f}min > {max_duration_minutes}min)")
    
    return issues

def main():
    parser = argparse.ArgumentParser(description="Process video files to extract clips based on sound detection.")
    parser.add_argument('--input-dir', required=True, help="Directory containing input video files.")
    parser.add_argument('--output-dir', required=True, help="Directory to save the output clips.")
    parser.add_argument('--sample-sound', required=True, help="Path to the sample kill sound.")
    parser.add_argument('--temp-dir', required=True, help="Directory for temporary files.")
    parser.add_argument('--max-file-size-mb', type=int, default=None, help="Maximum file size in MB (default: no limit)")
    parser.add_argument('--max-duration-minutes', type=int, default=None, help="Maximum video duration in minutes (default: no limit)")
    parser.add_argument('--skip-validation', action='store_true', help="Skip file validation (use with caution)")
    args = parser.parse_args()

    resample_rate = 8000
    cooldown_period = 1

    print("=== WALD01 Auto-Edit Video Processor ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sample sound: {args.sample_sound}")
    print(f"Max file size: {'No limit' if args.max_file_size_mb is None else f'{args.max_file_size_mb}MB'}")
    print(f"Max duration: {'No limit' if args.max_duration_minutes is None else f'{args.max_duration_minutes} minutes'}")
    print()

    # Validate inputs
    if not args.skip_validation:
        print("Validating input files...")
        validation_issues = validate_input_files(
            args.input_dir, 
            args.max_file_size_mb, 
            args.max_duration_minutes
        )
        
        if validation_issues:
            print("❌ Validation failed:")
            for issue in validation_issues:
                print(f"  - {issue}")
            print("\nOptions:")
            print("  1. Use smaller/shorter video files")
            print("  2. Increase limits with --max-duration-minutes or remove file size limit")
            print("  3. Skip validation with --skip-validation (not recommended)")
            return 1
        else:
            print("✅ All files passed validation.")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    video_files = [f for f in os.listdir(args.input_dir) if f.endswith((".mp4", ".mov", ".avi"))]
    
    if not video_files:
        print("No video files found in input directory.")
        return 1
    
    print(f"\nProcessing {len(video_files)} video file(s):")
    
    total_clips_extracted = 0
    
    for i, filename in enumerate(video_files):
        print(f"\n[{i+1}/{len(video_files)}] Processing: {filename}", flush=True)
        video_path = os.path.join(args.input_dir, filename)
        video_basename = os.path.splitext(filename)[0]
        temp_audio_path = os.path.join(args.temp_dir, f"{uuid.uuid4()}.wav")
        
        # Show file info
        size_mb = get_file_size_mb(video_path)
        duration = get_video_duration(video_path)
        print(f"  File size: {size_mb:.1f}MB", flush=True)
        if duration:
            print(f"  Duration: {duration/60:.1f} minutes", flush=True)
        print(f"PROGRESS:0:Starting processing of {filename}...", flush=True)
        sys.stdout.flush()  # Force immediate output

        try:
            # Extract audio
            extract_audio(video_path, temp_audio_path, resample_rate)
            
            # Detect kill sounds
            timestamps = detect_kill_sounds(temp_audio_path, args.sample_sound, resample_rate, cooldown_period)
            
            # Extract clips
            if timestamps:
                extract_clips(video_path, timestamps, args.output_dir, video_basename)
                total_clips_extracted += len(timestamps)
            else:
                print(f"PROGRESS:95:No clips extracted from {filename}")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Error processing {filename}: {e.stderr}", file=sys.stderr)
        except Exception as e:
            print(f"❌ Unexpected error with {filename}: {e}", file=sys.stderr)
        finally:
            # Clean up temp file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
    
    print(f"PROGRESS:100:Processing Complete")
    print(f"\n=== Processing Complete ===")
    print(f"Total clips extracted: {total_clips_extracted}")
    print(f"Output directory: {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
