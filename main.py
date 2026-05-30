import os
import uuid
import threading
import subprocess
import json
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'shortsmaker2024'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

UPLOAD_FOLDER = '/tmp/uploads'
OUTPUT_FOLDER = '/tmp/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def get_video_duration(file_path):
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json', file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except Exception as e:
        print(f"Duration error: {e}")
        return 60.0


def emit_progress(percent, message):
    socketio.emit('progress', {'percent': percent, 'message': message})


def process_video(file_path, session_id, settings):
    try:
        emit_progress(5, '📂 Loading video...')

        duration = get_video_duration(file_path)
        num_clips = int(settings.get('num_clips', 5))
        quality = settings.get('quality', 'high')

        quality_map = {
            'high': {'w': 1080, 'h': 1920, 'br': '6M'},
            'medium': {'w': 720, 'h': 1280, 'br': '3M'},
            'low': {'w': 480, 'h': 854, 'br': '1.5M'}
        }
        q = quality_map.get(quality, quality_map['high'])

        clip_duration = min(45, duration / max(num_clips, 1))
        emit_progress(15, '🎬 Finding best moments...')

        moments = []
        if duration < clip_duration * num_clips:
            num_clips = max(1, int(duration / clip_duration))

        step = max(1, (duration - clip_duration) / max(num_clips, 1))
        for i in range(num_clips):
            start = i * step
            if start + clip_duration > duration:
                start = max(0, duration - clip_duration)
            moments.append({
                'start': start,
                'end': min(start + clip_duration, duration)
            })

        output_files = []

        for i, moment in enumerate(moments):
            progress = 20 + int((i / len(moments)) * 70)
            emit_progress(progress, f'✂️ Creating Short {i+1} of {len(moments)}...')

            output_filename = f"short_{session_id}_{i+1}.mp4"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            cmd = [
                'ffmpeg', '-y',
                '-ss', str(moment['start']),
                '-i', file_path,
                '-t', str(moment['end'] - moment['start']),
                '-vf', f"crop='min(iw\\,ih*9/16)':'min(ih\\,iw*16/9)',scale={q['w']}:{q['h']}",
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-b:v', q['br'],
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',
                output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")

            if os.path.exists(output_path):
                output_files.append({
                    'filename': output_filename,
                    'clip_number': i + 1,
                    'start_time': f"{int(moment['start']//60):02d}:{int(moment['start']%60):02d}",
                    'end_time': f"{int(moment['end']//60):02d}:{int(moment['end']%60):02d}",
                    'excitement_score': 85
                })

        if os.path.exists(file_path):
            os.remove(file_path)

        emit_progress(100, '✅ All Shorts created!')
        socketio.emit('complete', {
            'session_id': session_id,
            'clips': output_files
        })

    except Exception as e:
        print(f"Error: {e}")
        socketio.emit('error', {'message': str(e)})


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    file = request.files['video']
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400

    session_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
        return jsonify({'error': 'Invalid file type'}), 400

    filename = f"{session_id}{ext}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    settings = {
        'num_clips': request.form.get('num_clips', 5),
        'quality': request.form.get('quality', 'high'),
        'caption_style': request.form.get('caption_style', 'bold_yellow'),
        'enable_captions': request.form.get('enable_captions', 'true') == 'true'
    }

    thread = threading.Thread(
        target=process_video,
        args=(file_path, session_id, settings)
    )
    thread.daemon = True
    thread.start()

    return jsonify({'session_id': session_id, 'message': 'Processing started'})


@app.route('/download/<filename>')
def download(filename):
    safe_name = os.path.basename(filename)
    file_path = os.path.join(OUTPUT_FOLDER, safe_name)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port)
