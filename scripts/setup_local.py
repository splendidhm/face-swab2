"""Download the versioned official model and verify its published-in-project digest."""
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/'models'/'face_landmarker.task'
URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
SHA256 = '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'
SEGMENT_MODEL = ROOT/'models'/'selfie_multiclass_256x256.tflite'
SEGMENT_URL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite'
SEGMENT_SHA256 = 'c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0'


def ensure_model(path, url, digest):
    path.parent.mkdir(exist_ok=True)
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError('Model checksum mismatch')
        temp = path.with_suffix('.download')
        temp.write_bytes(data)
        temp.replace(path)


def main():
    ensure_model(MODEL, URL, SHA256)
    ensure_model(SEGMENT_MODEL, SEGMENT_URL, SEGMENT_SHA256)
    sys.path.insert(0, str(ROOT))
    from src.local.faces import Landmarker
    from src.local.media import ffmpeg
    model = Landmarker()
    model.close()
    from src.local.matting import FaceSegmenter
    segmenter = FaceSegmenter()
    segmenter.close()
    import tkinter
    print('Python:', sys.version.split()[0])
    print('Tk:', tkinter.TkVersion)
    print('FFmpeg:', ffmpeg())
    print('Face model SHA256:', SHA256)
    print('Face/hair segmenter SHA256:', SEGMENT_SHA256)
    print('Local setup OK')


if __name__ == '__main__':
    main()
