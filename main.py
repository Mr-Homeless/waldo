from flask import Flask, render_template, request, redirect, url_for, flash, Response
import os
import subprocess
import uuid
import shutil
import json
import tempfile
import cv2
from datetime import datetime
from werkzeug.utils import secure_filename

def get_cs2_detect_python_path():
    """
    Dynamically detect the Python executable path for the cs2-detect-env conda environment.
    Works across different systems and usernames.
    """
    import os
    import subprocess

    # First try to get the conda environment path using conda itself
    try:
        # Try to find conda and get the environment path
        result = subprocess.run(['conda', 'info', '--envs'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'cs2-detect-env' in line and '*' not in line:  # Skip current env marker
                    parts = line.split()
                    if len(parts) >= 2:
                        env_path = parts[-1]  # Last part is the path
                        python_path = os.path.join(env_path, 'bin', 'python')
                        if os.path.exists(python_path):
                            return python_path
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass

    # Fallback: try common conda installation locations
    import getpass
    username = getpass.getuser()

    # Common conda locations to check
    conda_bases = [
        f'/home/{username}/miniconda3',
        f'/home/{username}/anaconda3',
        '/opt/conda',
        '/opt/miniconda3',
        '/opt/anaconda3'
    ]

    for conda_base in conda_bases:
        python_path = os.path.join(conda_base, 'envs', 'cs2-detect-env', 'bin', 'python')
        if os.path.exists(python_path):
            return python_path

    # If all else fails, try using the current Python and hope it's in the right environment
    import sys
    if 'cs2-detect-env' in sys.executable:
        return sys.executable

    # Last resort: return the hardcoded path and let it fail with a clear error
    raise FileNotFoundError(
        "Could not find Python executable for cs2-detect-env conda environment. "
        "Please ensure the environment is properly installed and activated."
    )

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['PROCESSED_FOLDER'] = os.path.join(BASE_DIR, 'processed_clips')
app.config['TRAINING_DATA_FOLDER'] = os.path.join(BASE_DIR, 'processed_vids')
app.config['MODELS_OUTPUT_FOLDER'] = os.path.join(BASE_DIR, 'deepcheat', 'VideoMAEv2', 'output')

# In-memory dictionary to store process commands and PIDs
PROCESS_STORE = {}

# Directory for persistent evaluation results
EVALUATIONS_DIR = os.path.join(BASE_DIR, 'evaluation_results')

def save_evaluation_results(process_id, results_data, temp_eval_dir):
    """Save evaluation results and frames to persistent storage"""
    os.makedirs(EVALUATIONS_DIR, exist_ok=True)

    eval_dir = os.path.join(EVALUATIONS_DIR, process_id)
    os.makedirs(eval_dir, exist_ok=True)

    # Save results data as JSON
    results_file = os.path.join(eval_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    # Copy frames directory if it exists
    if os.path.exists(temp_eval_dir):
        frames_dir = os.path.join(eval_dir, 'frames')
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        shutil.copytree(temp_eval_dir, frames_dir)

    # Update frame paths to new location
    for clip in results_data['clip_results']:
        if clip['frame_paths']:
            clip['frame_paths'] = [
                p.replace(temp_eval_dir, os.path.join(eval_dir, 'frames'))
                for p in clip['frame_paths']
            ]

    # Save updated results with new paths
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    return eval_dir

def load_evaluation_results(process_id):
    """Load saved evaluation results"""
    eval_dir = os.path.join(EVALUATIONS_DIR, process_id)
    results_file = os.path.join(eval_dir, 'results.json')

    if not os.path.exists(results_file):
        return None

    with open(results_file, 'r') as f:
        return json.load(f)

def get_saved_evaluations():
    """Get list of saved evaluation results"""
    if not os.path.exists(EVALUATIONS_DIR):
        return []

    evaluations = []
    for dirname in os.listdir(EVALUATIONS_DIR):
        eval_dir = os.path.join(EVALUATIONS_DIR, dirname)
        results_file = os.path.join(eval_dir, 'results.json')

        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                data = json.load(f)
                evaluations.append({
                    'process_id': dirname,
                    'timestamp': data['evaluation_metadata']['evaluation_timestamp'],
                    'clips_count': data['summary_stats']['total_clips'],
                    'model': os.path.basename(data['evaluation_metadata']['model_path']),
                    'dataset': os.path.basename(data['evaluation_metadata']['clips_path'])
                })

    return sorted(evaluations, key=lambda x: x['timestamp'], reverse=True)

def get_adaptive_crop_size(video_width, video_height, base_resolution=(3840, 2160), base_crop_size=240):
    """Calculate adaptive crop size based on video resolution to maintain same field of view"""
    base_width, base_height = base_resolution

    # Calculate the crop size as a percentage of the original training resolution
    crop_percentage = base_crop_size / base_width

    # Apply the same percentage to the current video width
    adaptive_crop_size = int(video_width * crop_percentage)

    # Ensure crop size is even and reasonable
    adaptive_crop_size = max(200, min(adaptive_crop_size, min(video_width, video_height)))

    return adaptive_crop_size

def generate_evaluation_results_data(predictions_file, clips_path, temp_eval_dir, process_info):
    """Generate comprehensive evaluation results data including statistics and clip analysis"""
    import numpy as np
    import math

    # Read prediction scores
    with open(predictions_file, 'r') as f:
        scores = [float(line.strip()) for line in f.readlines()]

    # Get clip names (in order they were processed)
    clips = sorted([f for f in os.listdir(clips_path) if f.endswith('.mp4')])

    # Ensure we have matching clips and scores
    if len(clips) != len(scores):
        clips = clips[:len(scores)]  # Trim to match scores

    # Calculate statistics
    scores_array = np.array(scores)

    def get_confidence_category(score):
        """Convert normalized score to confidence category"""
        if score >= 0.8:
            return {"category": "Very High Confidence", "label": "Likely Cheating", "color": "#dc3545", "level": 5}
        elif score >= 0.6:
            return {"category": "High Confidence", "label": "Possible Cheating", "color": "#fd7e14", "level": 4}
        elif score >= 0.4:
            return {"category": "Medium Confidence", "label": "Uncertain", "color": "#ffc107", "level": 3}
        elif score >= 0.2:
            return {"category": "Low Confidence", "label": "Likely Legitimate", "color": "#20c997", "level": 2}
        else:
            return {"category": "Very Low Confidence", "label": "Likely Legitimate", "color": "#28a745", "level": 1}

    def sigmoid_to_probability(normalized_score, min_logit=-1.9871155, max_logit=2.4927201):
        """Convert normalized score back to approximate probability using sigmoid"""
        # Reverse min-max normalization to get approximate logit
        estimated_logit = normalized_score * (max_logit - min_logit) + min_logit
        # Apply sigmoid to get probability
        probability = 1 / (1 + math.exp(-estimated_logit))
        return probability

    # Generate clip results
    clip_results = []
    for i, (clip, score) in enumerate(zip(clips, scores)):
        confidence = get_confidence_category(score)
        probability = sigmoid_to_probability(score)

        # Get frame paths if they exist
        frames_dir = os.path.join(temp_eval_dir, os.path.splitext(clip)[0])
        frame_paths = []
        if os.path.exists(frames_dir):
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
            frame_paths = [os.path.join(frames_dir, f) for f in frame_files]

        clip_data = {
            'id': i,
            'filename': clip,
            'clip_name': os.path.splitext(clip)[0],
            'normalized_score': score,
            'probability': probability,
            'confidence': confidence,
            'video_path': os.path.join(clips_path, clip),
            'frame_paths': frame_paths
        }
        clip_results.append(clip_data)

    # Calculate summary statistics
    summary_stats = {
        'total_clips': len(scores),
        'mean_score': float(np.mean(scores_array)),
        'median_score': float(np.median(scores_array)),
        'std_score': float(np.std(scores_array)),
        'min_score': float(np.min(scores_array)),
        'max_score': float(np.max(scores_array)),
        'high_confidence_count': len([s for s in scores if s >= 0.6]),
        'medium_confidence_count': len([s for s in scores if 0.4 <= s < 0.6]),
        'low_confidence_count': len([s for s in scores if s < 0.4])
    }

    # Calculate distribution for histogram
    bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
    hist, _ = np.histogram(scores_array, bins=bins)

    distribution = {
        'bins': [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)],
        'counts': hist.tolist()
    }

    return {
        'clip_results': clip_results,
        'summary_stats': summary_stats,
        'distribution': distribution,
        'evaluation_metadata': {
            'model_path': process_info['model_path'],
            'clips_path': clips_path,
            'total_clips_processed': len(scores),
            'evaluation_timestamp': datetime.now().isoformat()
        }
    }

# --- Helper Functions ---
def get_processed_clips_dirs():
    if not os.path.exists(app.config['PROCESSED_FOLDER']):
        return []
    return sorted([d for d in os.listdir(app.config['PROCESSED_FOLDER']) if os.path.isdir(os.path.join(app.config['PROCESSED_FOLDER'], d))])

def get_existing_models():
    models = []
    if not os.path.exists(app.config['MODELS_OUTPUT_FOLDER']):
        return []

    # Check for models in the main output directory
    for item in os.listdir(app.config['MODELS_OUTPUT_FOLDER']):
        item_path = os.path.join(app.config['MODELS_OUTPUT_FOLDER'], item)

        if item.endswith('.pth'):
            # Direct .pth file in main directory (legacy models)
            models.append({'path': item, 'display_name': item, 'full_path': item_path})
        elif os.path.isdir(item_path):
            # Check subdirectories for models
            model_info_path = os.path.join(item_path, 'model_info.json')
            checkpoint_files = [f for f in os.listdir(item_path) if f.endswith('.pth')]

            if checkpoint_files:
                # Get the latest checkpoint
                latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('-')[1].split('.')[0]) if '-' in x and x.split('-')[1].split('.')[0].isdigit() else 0)
                full_checkpoint_path = os.path.join(item_path, latest_checkpoint)

                # Try to get display name from metadata
                display_name = item
                if os.path.exists(model_info_path):
                    try:
                        with open(model_info_path, 'r') as f:
                            metadata = json.load(f)

                        # Build display name with training history
                        total_clips = metadata.get('total_clips_trained', metadata.get('clips_count', '?'))
                        training_type = metadata.get('training_type', 'unknown')
                        label_type = metadata.get('label_type', 'unknown')
                        timestamp = metadata.get('timestamp', item)

                        # Create summary of training history
                        if 'training_history' in metadata and len(metadata['training_history']) > 1:
                            history = metadata['training_history']
                            steps = len(history)
                            # Show label types from training history
                            label_types = list(set(step.get('label_type', 'unknown') for step in history))
                            label_summary = '+'.join(label_types) if len(label_types) > 1 else label_types[0]
                            display_name = f"{label_summary} ({total_clips} total clips, {steps} training steps) - {timestamp}"
                        else:
                            display_name = f"{label_type} ({total_clips} clips) - {timestamp}"

                    except:
                        # Fallback to directory name
                        display_name = item

                models.append({
                    'path': os.path.join(item, latest_checkpoint),
                    'display_name': display_name,
                    'full_path': full_checkpoint_path
                })

    return sorted(models, key=lambda x: x['display_name'])

# --- Routes ---
@app.route('/')
def index():
    processed_clips = get_processed_clips_dirs()
    existing_models = get_existing_models()
    return render_template('index.html', processed_clips=processed_clips, existing_models=existing_models)

@app.route('/process', methods=['POST'])
def process_video():
    if 'videoFile' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['videoFile']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        temp_input_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{uuid.uuid4().hex[:8]}")
        os.makedirs(temp_input_dir)
        uploaded_video_path = os.path.join(temp_input_dir, filename)
        file.save(uploaded_video_path)

        output_dir_name = f"clips_{os.path.splitext(filename)[0]}_{uuid.uuid4().hex[:4]}"
        output_dir_path = os.path.join(app.config['PROCESSED_FOLDER'], output_dir_name)
        
        sample_sound_path = os.path.join(BASE_DIR, 'deepcheat', 'utils', 'autoedit', 'sample', 'csheadshot.wav')
        autoedit_script_path = os.path.join(BASE_DIR, 'deepcheat', 'utils', 'autoedit', 'autoedit_improved.py')

        command = [
            'python', autoedit_script_path,
            '--input-dir', temp_input_dir,
            '--output-dir', output_dir_path,
            '--sample-sound', sample_sound_path,
            '--temp-dir', app.config['UPLOAD_FOLDER']
        ]
        
        process_id = str(uuid.uuid4())
        PROCESS_STORE[process_id] = {'command': command, 'cleanup_dir': temp_input_dir}
        
        return redirect(url_for('show_processing', process_id=process_id))

    return redirect(url_for('index'))

@app.route('/processing/<process_id>')
def show_processing(process_id):
    return render_template('processing.html', process_id=process_id)

@app.route('/training/<process_id>')
def show_training(process_id):
    return render_template('training.html', process_id=process_id)

@app.route('/stream-training/<process_id>')
def stream_training(process_id):
    process_info = PROCESS_STORE.get(process_id)
    if not process_info:
        return Response("Process not found.", mimetype='text/plain')

    command = process_info['command']
    clips_path = process_info['clips_path']
    temp_training_dir = process_info['temp_training_dir']
    label = process_info['label']

    def generate():
        try:
            # Step 1: Preprocess clips from MP4 to frames
            yield f"data: Starting preprocessing of clips...<br>\n\n"

            # Get list of clips
            clips = [f for f in os.listdir(clips_path) if f.endswith('.mp4')]
            total_clips = len(clips)

            if total_clips == 0:
                yield f"data: <b>Error:</b> No MP4 clips found in {clips_path}<br>\n\n"
                return

            # Check minimum clip requirement
            min_clips_required = 48  # batch_size * update_freq
            if total_clips < min_clips_required:
                yield f"data: <br><b>Error:</b> Training requires at least {min_clips_required} clips.<br>\n\n"
                yield f"data: You currently have {total_clips} clips.<br>\n\n"
                yield f"data: Please process more video footage or add more clips before training.<br>\n\n"
                return

            yield f"data: Found {total_clips} clips to process<br>\n\n"

            # Create CSV file for training
            csv_lines = []

            for idx, clip_file in enumerate(clips, 1):
                clip_name = os.path.splitext(clip_file)[0]
                clip_output_dir = os.path.join(temp_training_dir, clip_name)
                os.makedirs(clip_output_dir, exist_ok=True)

                yield f"data: Processing clip {idx}/{total_clips}: {clip_name}<br>\n\n"

                # Extract frames from video using the same method as the working version
                video_path = os.path.join(clips_path, clip_file)

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    yield f"data: Warning: Could not open {clip_name}<br>\n\n"
                    continue

                # Get video resolution for adaptive cropping
                video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                crop_size = get_adaptive_crop_size(video_width, video_height)

                yield f"data: Video resolution: {video_width}x{video_height}, using {crop_size}px crop<br>\n\n"

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_number = 0
                save_frame_number = 0

                # Process frames similar to nick_crop.py
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Extract frames 85-100 (where the killshot happens)
                    if 85 <= frame_number <= 100:
                        # Center crop with adaptive size
                        y, x, c = frame.shape
                        start_x = x // 2 - (crop_size // 2)
                        start_y = y // 2 - (crop_size // 2)
                        cropped_frame = frame[start_y:start_y + crop_size, start_x:start_x + crop_size]

                        # Save frame as image
                        img_filename = f"img_{save_frame_number:010d}.jpg"
                        img_path = os.path.join(clip_output_dir, img_filename)
                        cv2.imwrite(img_path, cropped_frame)

                        save_frame_number += 1

                        if save_frame_number >= 16:  # We only need 16 frames
                            break

                    frame_number += 1
                    if frame_number > 100:  # No need to read beyond frame 100
                        break

                cap.release()

                if save_frame_number < 16:
                    yield f"data: Warning: Only extracted {save_frame_number} frames from {clip_name}<br>\n\n"

                # Add to CSV
                csv_lines.append(f"{clip_output_dir} 15 {label}\n")

            # Write CSV files
            for csv_name in ['train.csv', 'val.csv', 'test.csv']:
                csv_path = os.path.join(temp_training_dir, csv_name)
                with open(csv_path, 'w') as f:
                    f.writelines(csv_lines)

            yield f"data: <br><b>Preprocessing complete!</b> Starting model training...<br><br>\n\n"

            # Step 2: Run training
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            env['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU if available

            # Run with the same Python that's running Flask (should have correct environment)
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, universal_newlines=True, env=env,
                                     cwd=os.path.join(BASE_DIR, 'deepcheat', 'VideoMAEv2'))

            while True:
                line = process.stdout.readline()
                if not line:
                    break
                # Clean and format the line
                line = line.strip()
                if line:
                    # Highlight important training metrics
                    if 'epoch' in line.lower() or 'loss' in line.lower() or 'accuracy' in line.lower():
                        yield f"data: <b>{line}</b><br>\n\n"
                    else:
                        yield f"data: {line}<br>\n\n"

            process.stdout.close()
            return_code = process.wait()

            if return_code != 0:
                yield f"data: <br><b>Error:</b> Training exited with code {return_code}<br>\n\n"
            else:
                yield f"data: <br><b>Training completed successfully!</b><br>\n\n"
                # Get the actual output directory from the command
                output_dir = app.config['MODELS_OUTPUT_FOLDER']
                try:
                    output_idx = command.index('--output_dir')
                    if output_idx + 1 < len(command):
                        output_dir = command[output_idx + 1]
                except ValueError:
                    pass

                # Check for the latest checkpoint in the output directory
                try:
                    checkpoint_files = [f for f in os.listdir(output_dir) if f.startswith('checkpoint-') and f.endswith('.pth')]
                    if checkpoint_files:
                        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('-')[1].split('.')[0]))
                        checkpoint_path = os.path.join(output_dir, latest_checkpoint)

                        # Get model name from directory
                        model_name = os.path.basename(output_dir)
                        yield f"data: <br><b>New model '{model_name}' saved successfully!</b><br>\n\n"
                        yield f"data: Checkpoint: {checkpoint_path}<br>\n\n"
                        yield f"data: This model is now available for fine-tuning and evaluation.<br>\n\n"
                    else:
                        yield f"data: Model checkpoints saved to: {output_dir}<br>\n\n"
                except OSError:
                    yield f"data: Model checkpoints saved to: {output_dir}<br>\n\n"

        except Exception as e:
            yield f"data: <br><b>An unexpected error occurred:</b> {e}<br>\n\n"
        finally:
            # Cleanup temporary directory
            if os.path.exists(temp_training_dir):
                shutil.rmtree(temp_training_dir)
            yield "data: PROCESS_COMPLETE\n\n"
            if process_id in PROCESS_STORE:
                del PROCESS_STORE[process_id]

    return Response(generate(), mimetype='text/event-stream')

@app.route('/stream-logs/<process_id>')
def stream_logs(process_id):
    process_info = PROCESS_STORE.get(process_id)
    if not process_info:
        return Response("Process not found.", mimetype='text/plain')

    command = process_info['command']
    cleanup_dir = process_info.get('cleanup_dir')

    def generate():
        try:
            # Start the process immediately and notify client
            yield f"data: Starting video processing...\n\n"
            
            # Set PYTHONUNBUFFERED to ensure immediate output
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                     text=True, bufsize=1, universal_newlines=True, env=env)
            
            # Read output line by line with no buffering
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                # Send each line immediately
                yield f"data: {line.strip()}\n\n"
                
            process.stdout.close()
            return_code = process.wait()
            if return_code != 0:
                yield f"data: \n--- \n**Error:** Process exited with code {return_code}.\n\n"
        except Exception as e:
            yield f"data: \n--- \n**An unexpected error occurred:** {e}\n\n"
        finally:
            if cleanup_dir and os.path.exists(cleanup_dir):
                shutil.rmtree(cleanup_dir)
            yield "data: PROCESS_COMPLETE\n\n"
            # Clean up the process from the store
            if process_id in PROCESS_STORE:
                del PROCESS_STORE[process_id]

    return Response(generate(), mimetype='text/event-stream')

@app.route('/train', methods=['POST'])
def train_model():
    clips_directory = request.form.get('clipsDirectory')
    training_type = request.form.get('trainingType')
    model_type = request.form.get('modelType')
    existing_model = request.form.get('existingModel')

    if not clips_directory:
        flash('Please select a clips directory')
        return redirect(url_for('index'))

    # Full path to clips directory
    clips_path = os.path.join(app.config['PROCESSED_FOLDER'], clips_directory)

    # Create a temporary training data directory
    temp_training_dir = os.path.join(BASE_DIR, 'temp_processing', f"training_{uuid.uuid4().hex[:8]}")
    os.makedirs(temp_training_dir, exist_ok=True)

    # Prepare the command for training
    training_script_path = os.path.join(BASE_DIR, 'deepcheat', 'VideoMAEv2', 'train_cheater_pred.py')

    # Determine the label (0 for not cheater, 1 for cheater)
    label = 1 if model_type == 'cheater' else 0

    # Use the Python from cs2-detect-env specifically (dynamically detected)
    python_executable = get_cs2_detect_python_path()

    # Base command
    command = [
        python_executable, training_script_path,
        '--model', 'vit_giant_patch14_224',
        '--data_set', 'cheater',
        '--nb_classes', '1',
        '--data_path', temp_training_dir,
        '--data_root', temp_training_dir,
        '--log_dir', app.config['MODELS_OUTPUT_FOLDER'],
        '--output_dir', app.config['MODELS_OUTPUT_FOLDER'],
        '--batch_size', '4',
        '--update_freq', '12',
        '--input_size', '224',
        '--short_side_size', '224',
        '--save_ckpt_freq', '20',
        '--num_frames', '16',
        '--sampling_rate', '1',
        '--num_sample', '1',
        '--num_workers', '4',
        '--opt', 'adamw',
        '--lr', '1e-3',
        '--drop_path', '0.1',
        '--clip_grad', '1.0',
        '--layer_decay', '0.9',
        '--opt_betas', '0.9', '0.999',
        '--weight_decay', '0.000',
        '--warmup_epochs', '10',
        '--epochs', '100',
        '--test_num_segment', '5',
        '--test_num_crop', '3'
    ]

    # Add finetune flag if fine-tuning
    if training_type == 'finetune':
        if existing_model:
            # The model path is now relative from get_existing_models()
            model_path = os.path.join(app.config['MODELS_OUTPUT_FOLDER'], existing_model)
        else:
            # Default to checkpoint-99.pth if it exists
            model_path = os.path.join(app.config['MODELS_OUTPUT_FOLDER'], 'checkpoint-99.pth')
        command.extend(['--finetune', model_path])

        # For fine-tuning, also create a unique output directory to avoid auto-resume conflicts
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_label = model_type.replace('-', '_')
        clips_count = len([f for f in os.listdir(clips_path) if f.endswith('.mp4')])

        finetune_dir_name = f"finetune_{model_label}_{clips_count}clips_{timestamp}"
        unique_output_dir = os.path.join(app.config['MODELS_OUTPUT_FOLDER'], finetune_dir_name)
        os.makedirs(unique_output_dir, exist_ok=True)

        # Get training history from base model
        training_history = []
        total_clips_trained = clips_count

        print(f"DEBUG START: existing_model = '{existing_model}'")
        print(f"DEBUG START: clips_count = {clips_count}")
        print(f"DEBUG START: training_type = '{training_type}'")

        # Try to load base model metadata to get its history
        if existing_model:
            # Extract the model directory from the existing_model path (remove checkpoint filename)
            model_dir_from_existing = os.path.dirname(existing_model) if '/' in existing_model else existing_model.replace('.pth', '')
            base_model_dir = os.path.join(app.config['MODELS_OUTPUT_FOLDER'], model_dir_from_existing)
            base_model_info_path = os.path.join(base_model_dir, 'model_info.json')

            print(f"DEBUG: existing_model = {existing_model}")
            print(f"DEBUG: model_dir_from_existing = {model_dir_from_existing}")
            print(f"DEBUG: base_model_dir = {base_model_dir}")
            print(f"DEBUG: base_model_info_path = {base_model_info_path}")
            print(f"DEBUG: Path exists? = {os.path.exists(base_model_info_path)}")

            if os.path.exists(base_model_info_path):
                try:
                    with open(base_model_info_path, 'r') as f:
                        base_metadata = json.load(f)

                    print(f"DEBUG: Successfully loaded base metadata from {base_model_info_path}")
                    print(f"DEBUG: Base metadata keys: {list(base_metadata.keys())}")

                    # Add base model's complete history to our history
                    if 'training_history' in base_metadata:
                        training_history = base_metadata['training_history'].copy()
                        print(f"DEBUG: Copied {len(training_history)} entries from base training history")
                    else:
                        # If no history in base model, create it from the base model's metadata
                        training_history = [{
                            'step': 1,
                            'model_name': base_metadata.get('model_name', existing_model),
                            'training_type': base_metadata.get('training_type', 'unknown'),
                            'label_type': base_metadata.get('label_type', 'unknown'),
                            'clips_count': base_metadata.get('clips_count', 0),
                            'clips_directory': base_metadata.get('clips_directory', 'unknown'),
                            'timestamp': base_metadata.get('timestamp', 'unknown')
                        }]
                        print(f"DEBUG: Created training history from base model metadata")

                    # Calculate total clips: base model's total + current training clips
                    base_total_clips = base_metadata.get('total_clips_trained', base_metadata.get('clips_count', 0))
                    total_clips_trained = base_total_clips + clips_count
                    print(f"DEBUG: base_total_clips = {base_total_clips}, current clips = {clips_count}, total = {total_clips_trained}")

                except Exception as e:
                    print(f"Warning: Could not read base model metadata: {e}")

        # Add current training step to history
        current_step = len(training_history) + 1
        training_history.append({
            'step': current_step,
            'model_name': finetune_dir_name,
            'training_type': training_type,
            'label_type': model_type,
            'clips_count': clips_count,
            'clips_directory': clips_directory,
            'timestamp': timestamp
        })

        print(f"DEBUG: Final training_history has {len(training_history)} steps")
        print(f"DEBUG: Final total_clips_trained = {total_clips_trained}")

        # Save model metadata with full history
        metadata = {
            'model_name': finetune_dir_name,
            'training_type': training_type,
            'label_type': model_type,
            'clips_count': clips_count,  # This training session's clips
            'total_clips_trained': total_clips_trained,  # Cumulative across all training
            'clips_directory': clips_directory,
            'timestamp': timestamp,
            'created': datetime.now().isoformat(),
            'base_model': existing_model or 'checkpoint-99.pth',
            'training_history': training_history
        }

        with open(os.path.join(unique_output_dir, 'model_info.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        # Update output directories in command
        for i, arg in enumerate(command):
            if arg == '--log_dir':
                command[i+1] = unique_output_dir
            elif arg == '--output_dir':
                command[i+1] = unique_output_dir

        # Disable auto-resume for fine-tuning to start fresh
        command.extend(['--no_auto_resume'])
    else:
        # For new training, use the base pre-trained model and disable auto-resume
        pretrained_model_path = os.path.join(BASE_DIR, 'deepcheat', 'vit_g_ps14_ak_ft_ckpt_7_clean.pth')
        if os.path.exists(pretrained_model_path):
            command.extend(['--finetune', pretrained_model_path])
        # Disable auto-resume for new model training
        command.extend(['--no_auto_resume'])
        # Create a descriptive model directory name
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_label = model_type.replace('-', '_')
        clips_count = len([f for f in os.listdir(clips_path) if f.endswith('.mp4')])

        model_dir_name = f"{model_label}_model_{clips_count}clips_{timestamp}"
        unique_output_dir = os.path.join(app.config['MODELS_OUTPUT_FOLDER'], model_dir_name)
        os.makedirs(unique_output_dir, exist_ok=True)

        # Save model metadata with initial training history
        training_history = [{
            'step': 1,
            'model_name': model_dir_name,
            'training_type': training_type,
            'label_type': model_type,
            'clips_count': clips_count,
            'clips_directory': clips_directory,
            'timestamp': timestamp
        }]

        metadata = {
            'model_name': model_dir_name,
            'training_type': training_type,
            'label_type': model_type,
            'clips_count': clips_count,
            'total_clips_trained': clips_count,  # For new models, same as clips_count
            'clips_directory': clips_directory,
            'timestamp': timestamp,
            'created': datetime.now().isoformat(),
            'training_history': training_history
        }

        with open(os.path.join(unique_output_dir, 'model_info.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        # Update output directories in command
        for i, arg in enumerate(command):
            if arg == '--log_dir':
                command[i+1] = unique_output_dir
            elif arg == '--output_dir':
                command[i+1] = unique_output_dir

    process_id = str(uuid.uuid4())
    PROCESS_STORE[process_id] = {
        'command': command,
        'clips_path': clips_path,
        'temp_training_dir': temp_training_dir,
        'label': label,
        'preprocessing_needed': True
    }

    return redirect(url_for('show_training', process_id=process_id))

@app.route('/evaluate', methods=['POST'])
def evaluate_clips():
    clips_directory = request.form.get('clipsDirectory')
    model_path = request.form.get('model')

    if not clips_directory:
        flash('Please select a clips directory')
        return redirect(url_for('index'))

    if not model_path:
        flash('Please select a model for evaluation')
        return redirect(url_for('index'))

    # Full path to clips directory
    clips_path = os.path.join(app.config['PROCESSED_FOLDER'], clips_directory)

    # Create a temporary evaluation data directory
    temp_eval_dir = os.path.join(BASE_DIR, 'temp_processing', f"evaluation_{uuid.uuid4().hex[:8]}")
    os.makedirs(temp_eval_dir, exist_ok=True)

    # Prepare the command for evaluation
    training_script_path = os.path.join(BASE_DIR, 'deepcheat', 'VideoMAEv2', 'train_cheater_pred.py')

    # Use the Python from cs2-detect-env specifically (dynamically detected)
    python_executable = get_cs2_detect_python_path()

    # Full path to the selected model
    model_full_path = os.path.join(app.config['MODELS_OUTPUT_FOLDER'], model_path)

    # Base evaluation command (similar to eval_cheater.sh)
    command = [
        python_executable, training_script_path,
        '--model', 'vit_giant_patch14_224',
        '--data_set', 'cheater',
        '--nb_classes', '1',
        '--finetune', model_full_path,
        '--batch_size', '8',
        '--input_size', '224',
        '--short_side_size', '224',
        '--num_frames', '16',
        '--sampling_rate', '1',
        '--num_sample', '1',
        '--num_workers', '4',
        '--opt', 'adamw',
        '--lr', '1e-3',
        '--drop_path', '0.3',
        '--clip_grad', '5.0',
        '--layer_decay', '0.9',
        '--opt_betas', '0.9', '0.999',
        '--weight_decay', '0.1',
        '--test_num_segment', '1',
        '--test_num_crop', '1',
        '--eval',  # This enables evaluation mode
        '--min_eval_score', '-1.9871155',
        '--max_eval_score', '2.4927201',
        '--output_dir', temp_eval_dir,
        '--data_path', temp_eval_dir,
        '--data_root', temp_eval_dir
    ]

    process_id = str(uuid.uuid4())
    PROCESS_STORE[process_id] = {
        'command': command,
        'clips_path': clips_path,
        'temp_eval_dir': temp_eval_dir,
        'model_path': model_path,
        'evaluation_mode': True
    }

    return redirect(url_for('show_evaluation', process_id=process_id))

@app.route('/evaluation/<process_id>')
def show_evaluation(process_id):
    process_info = PROCESS_STORE.get(process_id)
    if not process_info:
        return "Process not found.", 404

    model_name = os.path.basename(process_info['model_path'])
    data_path = process_info['clips_path']
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return render_template('evaluation.html',
                         process_id=process_id,
                         model_name=model_name,
                         data_path=data_path,
                         start_time=start_time)

@app.route('/stream-evaluation/<process_id>')
def stream_evaluation(process_id):
    process_info = PROCESS_STORE.get(process_id)
    if not process_info:
        return Response("Process not found.", mimetype='text/plain')

    command = process_info['command']
    clips_path = process_info['clips_path']
    temp_eval_dir = process_info['temp_eval_dir']
    model_path = process_info['model_path']

    def generate():
        try:
            # Step 1: Preprocess clips from MP4 to frames for evaluation
            yield f"data: {json.dumps({'type': 'log', 'content': 'Starting evaluation preprocessing...'})}\n\n"

            # Get list of clips
            clips = [f for f in os.listdir(clips_path) if f.endswith('.mp4')]
            total_clips = len(clips)

            if total_clips == 0:
                yield f"data: {json.dumps({'type': 'log', 'content': '<b>Error:</b> No MP4 clips found in ' + clips_path})}\n\n"
                yield f"data: {json.dumps({'type': 'status', 'status': 'error'})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'log', 'content': f'Found {total_clips} clips to evaluate'})}\n\n"
            yield f"data: {json.dumps({'type': 'log', 'content': f'Using model: {os.path.basename(model_path)}'})}\n\n"

            # Create CSV file for evaluation (all clips get label 0 for evaluation)
            csv_lines = []

            for idx, clip_file in enumerate(clips, 1):
                clip_name = os.path.splitext(clip_file)[0]
                clip_output_dir = os.path.join(temp_eval_dir, clip_name)
                os.makedirs(clip_output_dir, exist_ok=True)

                yield f"data: {json.dumps({'type': 'log', 'content': f'Processing clip {idx}/{total_clips}: {clip_name}'})}\n\n"

                # Extract frames from video using the same method as the working version
                video_path = os.path.join(clips_path, clip_file)

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    yield f"data: {json.dumps({'type': 'log', 'content': f'Warning: Could not open {clip_name}'})}\n\n"
                    continue

                # Get video resolution for adaptive cropping
                video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                crop_size = get_adaptive_crop_size(video_width, video_height)

                yield f"data: {json.dumps({'type': 'log', 'content': f'Video resolution: {video_width}x{video_height}, using {crop_size}px crop'})}\n\n"

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_number = 0
                save_frame_number = 0

                # Process frames similar to nick_crop.py
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Extract frames 85-100 (where the killshot happens)
                    if 85 <= frame_number <= 100:
                        # Center crop with adaptive size
                        y, x, c = frame.shape
                        start_x = x // 2 - (crop_size // 2)
                        start_y = y // 2 - (crop_size // 2)
                        cropped_frame = frame[start_y:start_y + crop_size, start_x:start_x + crop_size]

                        # Save frame as image
                        img_filename = f"img_{save_frame_number:010d}.jpg"
                        img_path = os.path.join(clip_output_dir, img_filename)
                        cv2.imwrite(img_path, cropped_frame)

                        save_frame_number += 1

                        if save_frame_number >= 16:  # We only need 16 frames
                            break

                    frame_number += 1
                    if frame_number > 100:  # No need to read beyond frame 100
                        break

                cap.release()

                if save_frame_number < 16:
                    yield f"data: {json.dumps({'type': 'log', 'content': f'Warning: Only extracted {save_frame_number} frames from {clip_name}'})}\n\n"

                # Add to CSV (label doesn't matter for evaluation, just use 0)
                csv_lines.append(f"{clip_output_dir} 15 0\n")

            # Write CSV files
            for csv_name in ['test.csv', 'val.csv']:
                csv_path = os.path.join(temp_eval_dir, csv_name)
                with open(csv_path, 'w') as f:
                    f.writelines(csv_lines)

            yield f"data: {json.dumps({'type': 'log', 'content': '<b>Preprocessing complete!</b> Starting evaluation...'})}\n\n"

            # Step 2: Run evaluation
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            env['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU if available

            # Run evaluation
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, universal_newlines=True, env=env,
                                     cwd=os.path.join(BASE_DIR, 'deepcheat', 'VideoMAEv2'))

            while True:
                line = process.stdout.readline()
                if not line:
                    break
                # Clean and format the line
                line = line.strip()
                if line:
                    # Highlight important evaluation metrics
                    if any(keyword in line.lower() for keyword in ['test', 'accuracy', 'evaluation', 'score']):
                        yield f"data: {json.dumps({'type': 'log', 'content': f'<b>{line}</b>'})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'log', 'content': line})}\n\n"

            process.stdout.close()
            return_code = process.wait()

            if return_code != 0:
                yield f"data: {json.dumps({'type': 'log', 'content': f'<b>Error:</b> Evaluation exited with code {return_code}'})}\n\n"
                yield f"data: {json.dumps({'type': 'status', 'status': 'error'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'log', 'content': '<b>Evaluation completed successfully!</b>'})}\n\n"

                # Check for prediction results and generate comprehensive analysis
                predictions_file = os.path.join(temp_eval_dir, 'cheater_preds.txt')
                if os.path.exists(predictions_file):
                    yield f"data: {json.dumps({'type': 'log', 'content': '<b>Generating comprehensive results analysis...</b>'})}\n\n"

                    # Generate detailed results analysis
                    results_data = generate_evaluation_results_data(predictions_file, clips_path, temp_eval_dir, process_info)

                    # Store results data for the results page
                    PROCESS_STORE[process_id]['results_data'] = results_data

                    total_clips = len(results_data['clip_results'])
                    yield f"data: {json.dumps({'type': 'results_ready', 'process_id': process_id, 'total_clips': total_clips})}\n\n"
                    yield f"data: {json.dumps({'type': 'log', 'content': f'<b>Analysis complete!</b> Processed {total_clips} clips.'})}\n\n"

                else:
                    yield f"data: {json.dumps({'type': 'log', 'content': f'Results saved to: {temp_eval_dir}'})}\n\n"

                yield f"data: {json.dumps({'type': 'status', 'status': 'completed'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'log', 'content': f'<b>An unexpected error occurred:</b> {str(e)}'})}\n\n"
            yield f"data: {json.dumps({'type': 'status', 'status': 'error'})}\n\n"
        finally:
            # Save evaluation results to persistent storage
            if process_id in PROCESS_STORE and 'results_data' in PROCESS_STORE[process_id]:
                save_evaluation_results(process_id, PROCESS_STORE[process_id]['results_data'], temp_eval_dir)
            # Don't delete temp_eval_dir anymore - we need the frames
            # Don't delete process from store yet - let it be cleaned up later

    return Response(generate(), mimetype='text/event-stream')

@app.route('/evaluation-results/<process_id>')
def show_evaluation_results(process_id):
    """Display comprehensive evaluation results"""
    # First check if results are in memory
    process_info = PROCESS_STORE.get(process_id)
    if process_info and 'results_data' in process_info:
        results_data = process_info['results_data']
    else:
        # Try to load from saved results
        results_data = load_evaluation_results(process_id)
        if not results_data:
            return "Results not found.", 404

    return render_template('evaluation_results.html',
                         process_id=process_id,
                         results=results_data)

@app.route('/clip-analysis/<process_id>/<int:clip_id>')
def show_clip_analysis(process_id, clip_id):
    """Display individual clip analysis"""
    # First check if results are in memory
    process_info = PROCESS_STORE.get(process_id)
    if process_info and 'results_data' in process_info:
        results_data = process_info['results_data']
    else:
        # Try to load from saved results
        results_data = load_evaluation_results(process_id)
        if not results_data:
            return "Results not found.", 404

    if clip_id >= len(results_data['clip_results']):
        return "Clip not found.", 404

    clip_data = results_data['clip_results'][clip_id]
    return render_template('clip_analysis.html',
                         process_id=process_id,
                         clip=clip_data,
                         metadata=results_data['evaluation_metadata'])

@app.route('/serve-video/<process_id>/<int:clip_id>')
def serve_video(process_id, clip_id):
    """Serve video files for playback"""
    # First check if results are in memory
    process_info = PROCESS_STORE.get(process_id)
    if process_info and 'results_data' in process_info:
        results_data = process_info['results_data']
    else:
        # Try to load from saved results
        results_data = load_evaluation_results(process_id)
        if not results_data:
            return "Video not found.", 404

    if clip_id >= len(results_data['clip_results']):
        return "Video not found.", 404

    clip_data = results_data['clip_results'][clip_id]
    video_path = clip_data['video_path']

    if not os.path.exists(video_path):
        return "Video file not found.", 404

    return Response(
        open(video_path, 'rb').read(),
        mimetype='video/mp4',
        headers={'Content-Disposition': f'inline; filename="{clip_data["filename"]}"'}
    )

@app.route('/serve-frame/<process_id>/<int:clip_id>/<int:frame_idx>')
def serve_frame(process_id, clip_id, frame_idx):
    """Serve individual frame images"""
    # First check if results are in memory
    process_info = PROCESS_STORE.get(process_id)
    if process_info and 'results_data' in process_info:
        results_data = process_info['results_data']
    else:
        # Try to load from saved results
        results_data = load_evaluation_results(process_id)
        if not results_data:
            return "Frame not found.", 404

    if clip_id >= len(results_data['clip_results']):
        return "Clip not found.", 404

    clip_data = results_data['clip_results'][clip_id]
    if not clip_data.get('frame_paths') or frame_idx >= len(clip_data['frame_paths']):
        return "Frame not found.", 404

    frame_path = clip_data['frame_paths'][frame_idx]
    if not os.path.exists(frame_path):
        return "Frame file not found.", 404

    return Response(
        open(frame_path, 'rb').read(),
        mimetype='image/jpeg',
        headers={'Content-Disposition': f'inline; filename="frame_{frame_idx}.jpg"'}
    )

@app.route('/saved-evaluations')
def show_saved_evaluations():
    """Show list of saved evaluation results"""
    evaluations = get_saved_evaluations()
    return render_template('saved_evaluations.html', evaluations=evaluations)

@app.route('/export-results/<process_id>')
def export_results(process_id):
    """Export evaluation results as JSON"""
    # First check if results are in memory
    process_info = PROCESS_STORE.get(process_id)
    if process_info and 'results_data' in process_info:
        results_data = process_info['results_data']
    else:
        # Try to load from saved results
        results_data = load_evaluation_results(process_id)
        if not results_data:
            return "Results not found.", 404

    # Prepare export data without frame paths (too large)
    export_data = {
        'evaluation_metadata': results_data['evaluation_metadata'],
        'summary_stats': results_data['summary_stats'],
        'distribution': results_data['distribution'],
        'clip_results': [{
            'id': c['id'],
            'filename': c['filename'],
            'clip_name': c['clip_name'],
            'normalized_score': c['normalized_score'],
            'probability': c['probability'],
            'confidence': c['confidence']
        } for c in results_data['clip_results']]
    }

    return Response(
        json.dumps(export_data, indent=2),
        mimetype='application/json',
        headers={'Content-Disposition': f'attachment; filename="evaluation_{process_id}.json"'}
    )

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    os.makedirs(app.config['TRAINING_DATA_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODELS_OUTPUT_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'final_footage'), exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
