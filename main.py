import os
import json
import uuid
import threading
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import librosa
import whisper
import tempfile
import shutil
import eventlet
eventlet.monkey_patch()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'shortsmaker2024secret'
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['OUTPUT_FOLDER'] = '/tmp/outputs'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',
    ping_timeout=120,
    ping_interval=25
)

os.makedirs('/tmp/uploads', exist_ok=True)
os.makedirs('/tmp/outputs', exist_ok=True)

processing_status = {}


def emit_progress(session_id, percent, message):
    processing_status[session_id] = {
        'percent': percent,
        'message': message
    }
    socketio.emit('progress', {
        'percent': percent,
        'message': message,
        'session_id': session_id
    })
    eventlet.sleep(0)


def detect_exciting_moments(audio_path, video_duration, num_clips=5):
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    hop_length = 512
    frame_length = 2048

    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_norm = (onset_env - onset_env.min()) / (
        onset_env.max() - onset_env.min() + 1e-8
    )

    min_len = min(len(rms_norm), len(onset_norm))
    excitement = rms_norm[:min_len] * 0.6 + onset_norm[:min_len] * 0.4

    times = librosa.frames_to_time(
        np.arange(len(excitement)),
        sr=sr,
        hop_length=hop_length
    )

    clip_duration = 45
    min_gap = 30
    window_size = int(clip_duration * sr / hop_length)

    windowed = []
    for i in range(max(1, len(excitement) - window_size)):
        score = np.mean(excitement[i:i + window_size])
        windowed.append((i, score))

    windowed.sort(key=lambda x: x[1], reverse=True)

    moments = []
    used = []

    for frame_idx, score in windowed:
        if frame_idx >= len(times):
            continue
        start_time = float(times[frame_idx])

        if start_time + clip_duration > video_duration:
            continue

        too_close = any(abs(start_time - u) < min_gap for u in used)

        if not too_close:
            moments.append({
                'start': start_time,
                'end': min(start_time + clip_duration, video_duration),
                'score': float(score),
                'duration': min(clip_duration, video_duration - start_time)
            })
            used.append(start_time)

        if len(moments) >= num_clips:
            break

    if not moments:
        step = video_duration / max(num_clips, 1)
        for i in range(num_clips):
            start = i * step
            if start + clip_duration <= video_duration:
                moments.append({
                    'start': start,
                    'end': start + clip_duration,
                    'score': 0.5,
                    'duration': clip_duration
                })

    moments.sort(key=lambda x: x['start'])
    return moments


def transcribe_audio(audio_path, start_time, end_time, model):
    try:
        result = model.transcribe(
            audio_path,
            word_timestamps=True,
            language='en'
        )

        words = []
        for segment in result.get('segments', []):
            seg_start = segment['start']
            seg_end = segment['end']

            if seg_end < start_time or seg_start > end_time:
                continue

            text = segment['text'].strip()
            if text:
                words.append({
                    'text': text,
                    'start': max(0.0, seg_start - start_time),
                    'end': min(
                        end_time - start_time,
                        seg_end - start_time
                    )
                })
        return words
    except Exception as e:
        print(f"Transcription error: {e}")
        return []


def process_video_task(file_path, session_id, settings):
    try:
        from moviepy.editor import (
            VideoFileClip,
            CompositeVideoClip,
            TextClip,
            ColorClip
        )

        emit_progress(session_id, 5, '📂 Loading your video...')

        video = VideoFileClip(file_path)
        duration = video.duration

        num_clips = int(settings.get('num_clips', 5))
        caption_style = settings.get('caption_style', 'bold_yellow')
        enable_captions = settings.get('enable_captions', True)
        quality = settings.get('quality', 'high')

        emit_progress(session_id, 10, '🎵 Extracting audio track...')

        audio_path = file_path + '_audio.wav'
        video.audio.write_audiofile(
            audio_path,
            fps=22050,
            nbytes=2,
            buffersize=2000,
            verbose=False,
            logger=None
        )

        emit_progress(session_id, 20, '🔍 AI is finding the best moments...')
        moments = detect_exciting_moments(audio_path, duration, num_clips)

        whisper_model = None
        if enable_captions:
            emit_progress(
                session_id, 30,
                '🤖 Loading AI caption engine...'
            )
            whisper_model = whisper.load_model("base")

        quality_settings = {
            'high': {
                'width': 1080,
                'height': 1920,
                'bitrate': '8000k',
                'fps': 30
            },
            'medium': {
                'width': 720,
                'height': 1280,
                'bitrate': '4000k',
                'fps': 30
            },
            'low': {
                'width': 480,
                'height': 854,
                'bitrate': '2000k',
                'fps': 30
            }
        }

        q = quality_settings.get(quality, quality_settings['high'])
        output_files = []

        for i, moment in enumerate(moments):
            progress = 35 + int((i / len(moments)) * 60)
            emit_progress(
                session_id,
                progress,
                f'✂️ Creating Short {i + 1} of {len(moments)}...'
            )

            clip = video.subclip(moment['start'], moment['end'])

            orig_w = clip.w
            orig_h = clip.h
            target_ratio = 9 / 16

            if (orig_w / orig_h) > target_ratio:
                new_w = int(orig_h * target_ratio)
                x1 = (orig_w - new_w) // 2
                clip = clip.crop(x1=x1, x2=x1 + new_w)
            else:
                new_h = int(orig_w / target_ratio)
                y1 = (orig_h - new_h) // 2
                clip = clip.crop(y1=y1, y2=y1 + new_h)

            clip = clip.resize((q['width'], q['height']))

            if enable_captions and whisper_model:
                emit_progress(
                    session_id,
                    progress,
                    f'💬 Generating captions for Short {i + 1}...'
                )
                words = transcribe_audio(
                    audio_path,
                    moment['start'],
                    moment['end'],
                    whisper_model
                )

                caption_clips = build_captions(
                    words,
                    caption_style,
                    q['width'],
                    q['height']
                )

                if caption_clips:
                    clip = CompositeVideoClip([clip] + caption_clips)

            output_filename = f"short_{session_id}_{i + 1}.mp4"
            output_path = os.path.join('/tmp/outputs', output_filename)

            clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                bitrate=q['bitrate'],
                fps=q['fps'],
                preset='fast',
                verbose=False,
                logger=None,
                threads=2
            )

            clip.close()

            output_files.append({
                'filename': output_filename,
                'clip_number': i + 1,
                'start_time': format_time(moment['start']),
                'end_time': format_time(moment['end']),
                'excitement_score': round(moment['score'] * 100, 1)
            })

        video.close()

        if os.path.exists(audio_path):
            os.remove(audio_path)

        if os.path.exists(file_path):
            os.remove(file_path)

        emit_progress(session_id, 100, '✅ All Shorts created!')

        socketio.emit('complete', {
            'session_id': session_id,
            'clips': output_files
        })

    except Exception as e:
        print(f"Error: {e}")
        socketio.emit('error', {
            'session_id': session_id,
            'message': str(e)
        })


def build_captions(words, style, vid_w, vid_h):
    from moviepy.editor import TextClip, ColorClip, CompositeVideoClip

    if not words:
        return []

    style_configs = {
        'bold_yellow': {
            'color': 'yellow',
            'fontsize': 70,
            'font': 'Arial-Bold',
            'stroke_color': 'black',
            'stroke_width': 4,
            'y_pos': 0.75
        },
        'white_clean': {
            'color': 'white',
            'fontsize': 60,
            'font': 'Arial',
            'stroke_color': 'black',
            'stroke_width': 2,
            'y_pos': 0.75
        },
        'tiktok_style': {
            'color': 'white',
            'fontsize': 75,
            'font': 'Arial-Bold',
            'stroke_color': 'black',
            'stroke_width': 5,
            'y_pos': 0.70
        },
        'highlight_box': {
            'color': 'white',
            'fontsize': 65,
            'font': 'Arial-Bold',
            'stroke_color': 'black',
            'stroke_width': 3,
            'y_pos': 0.75
        },
        'fire_red': {
            'color': '#FF2200',
            'fontsize': 75,
            'font': 'Arial-Bold',
            'stroke_color': 'black',
            'stroke_width': 5,
            'y_pos': 0.75
        },
        'neon_green': {
            'color': '#00FF41',
            'fontsize': 70,
            'font': 'Arial-Bold',
            'stroke_color': 'black',
            'stroke_width': 4,
            'y_pos': 0.75
        }
    }

    s = style_configs.get(style, style_configs['bold_yellow'])
    caption_clips = []

    for word_data in words:
        text = word_data['text'].strip()
        start = word_data['start']
        dur = word_data['end'] - word_data['start']

        if not text or dur <= 0:
            continue

        try:
            txt = TextClip(
                text,
                fontsize=s['fontsize'],
                color=s['color'],
                font=s['font'],
                stroke_color=s['stroke_color'],
                stroke_width=s['stroke_width'],
                method='caption',
                size=(vid_w - 100, None),
                align='center'
            )
            txt = txt.set_start(start).set_duration(dur)
            y_position = int(vid_h * s['y_pos'])
            txt = txt.set_position(('center', y_position))
            caption_clips.append(txt)
        except Exception as e:
            print(f"Caption clip error: {e}")
            continue

    return caption_clips


def format_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    file = request.files['video']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    allowed = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
    if ext not in allowed:
        return jsonify({'error': f'Use: {", ".join(allowed)}'}), 400

    session_id = str(uuid.uuid4())[:8]
    filename = f"{session_id}{ext}"
    file_path = os.path.join('/tmp/uploads', filename)
    file.save(file_path)

    settings = {
        'num_clips': request.form.get('num_clips', 5),
        'caption_style': request.form.get('caption_style', 'bold_yellow'),
        'enable_captions': request.form.get('enable_captions', 'true') == 'true',
        'quality': request.form.get('quality', 'high')
    }

    thread = threading.Thread(
        target=process_video_task,
        args=(file_path, session_id, settings)
    )
    thread.daemon = True
    thread.start()

    return jsonify({
        'session_id': session_id,
        'message': 'Processing started'
    })


@app.route('/download/<filename>')
def download(filename):
    safe_name = os.path.basename(filename)
    file_path = os.path.join('/tmp/outputs', safe_name)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404


@app.route('/status/<session_id>')
def status(session_id):
    s = processing_status.get(
        session_id,
        {'percent': 0, 'message': 'Waiting...'}
    )
    return jsonify(s)


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
