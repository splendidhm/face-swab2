from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / 'workspace'
CREATE_NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


class Cancelled(RuntimeError):
    pass


def ffmpeg():
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(args, cancel=None):
    """Poll without blocking cancellation; file-backed stderr cannot fill a pipe."""
    import tempfile
    with tempfile.TemporaryFile() as log:
        proc = subprocess.Popen([ffmpeg(), '-hide_banner', '-loglevel', 'error', '-nostdin',
                                 *map(str, args)], stdout=subprocess.DEVNULL, stderr=log,
                                creationflags=CREATE_NO_WINDOW)
        try:
            while proc.poll() is None:
                if cancel is not None and cancel.is_set():
                    raise Cancelled('작업을 취소했습니다.')
                time.sleep(0.1)
            if proc.returncode:
                log.seek(0)
                raise RuntimeError('FFmpeg 처리 실패: ' + log.read().decode('utf-8', 'replace')[-3000:])
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError('영상을 열 수 없습니다.')
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(3)), int(cap.get(4))
        if not np.isfinite(fps) or fps <= 0 or count <= 0 or min(w, h) <= 0:
            raise ValueError('영상의 프레임 정보가 올바르지 않습니다.')
        return {'fps': fps, 'frames': count, 'width': w, 'height': h,
                'duration': count / fps}
    finally:
        cap.release()


def read_frame(path, index=0):
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError('선택한 프레임을 읽지 못했습니다.')
        return frame
    finally:
        cap.release()


def normalize_video(source, destination, cancel=None):
    """Apply orientation, letterbox to 720p and CFR before ROI selection."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError('입력과 출력 영상은 다른 파일이어야 합니다.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Optional audio mapping supports silent input. Final export adds stereo silence.
    run_ffmpeg(['-y', '-i', source, '-map', '0:v:0', '-map', '0:a:0?',
                '-vf', 'scale=1280:720:force_original_aspect_ratio=decrease:force_divisible_by=2,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1',
                '-r', '30', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-ac', '2', '-ar', '48000',
                '-b:a', '192k', '-movflags', '+faststart', destination], cancel)
    return video_info(destination)


def has_audio(path):
    # The bundled FFmpeg is sufficient; no separate ffprobe installation required.
    result = subprocess.run([ffmpeg(), '-hide_banner', '-i', str(path)],
                            capture_output=True, creationflags=CREATE_NO_WINDOW)
    return b'Audio:' in result.stderr


def export_mp4(silent_video, original, destination, duration, cancel=None):
    args = ['-y', '-i', silent_video]
    if has_audio(original):
        args += ['-i', original]
    else:
        args += ['-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=stereo']
    args += ['-map', '0:v:0', '-map', '1:a:0', '-c:v', 'libx264', '-crf', '20',
             '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
             '-ac', '2', '-ar', '48000', '-b:a', '192k', '-af', 'apad',
             '-t', f'{duration:.9f}', '-movflags', '+faststart', destination]
    run_ffmpeg(args, cancel)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
