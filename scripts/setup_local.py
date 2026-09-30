"""Download the versioned official model and verify its published-in-project digest."""
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/'models'/'face_landmarker.task'
URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
SHA256 = '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'


def main():
    MODEL.parent.mkdir(exist_ok=True)
    if not MODEL.exists() or hashlib.sha256(MODEL.read_bytes()).hexdigest() != SHA256:
        with urllib.request.urlopen(URL, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise RuntimeError('Model checksum mismatch')
        temp = MODEL.with_suffix('.download')
        temp.write_bytes(data)
        temp.replace(MODEL)
    sys.path.insert(0, str(ROOT))
    from src.local.faces import Landmarker
    from src.local.media import ffmpeg
    model = Landmarker()
    model.close()
    import tkinter
    print('Python:', sys.version.split()[0])
    print('Tk:', tkinter.TkVersion)
    print('FFmpeg:', ffmpeg())
    print('Face model SHA256:', SHA256)
    print('Local setup OK')


if __name__ == '__main__':
    main()
