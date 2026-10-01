"""Write local visual QA artifacts from public fixtures; never uploads images."""
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.local.faces import Landmarker, OVAL, bbox, clip_box, composite, extract_face


def checker(face, size=(300, 340)):
    background = np.indices((size[1], size[0])).transpose(1, 2, 0)//16
    tiles = np.where(background.sum(2) % 2, 210, 245).astype(np.uint8)
    canvas = Image.fromarray(np.repeat(tiles[..., None], 3, axis=2)).convert('RGBA')
    face.thumbnail((size[0]-20, size[1]-20))
    canvas.alpha_composite(face, ((size[0]-face.width)//2, (size[1]-face.height)//2))
    return canvas.convert('RGB')


def main():
    output = ROOT/'workspace'/'expression-review'
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT/'workspace'/'test-assets'/'astronaut.png'
    target = Path(matplotlib.get_data_path())/'sample_data'/'grace_hopper.jpg'
    model = Landmarker()
    try:
        asset = extract_face(source, output/'cutout.png', model)
        with Image.open(asset.path) as image:
            cutout = checker(image.convert('RGBA'))
        frame = cv2.cvtColor(np.array(Image.open(target).convert('RGB')), cv2.COLOR_RGB2BGR)
        points = model.detect(frame)[0]
        start = time.perf_counter()
        result = composite(frame, asset, points, landmarker=model)
        elapsed = time.perf_counter()-start
        x, y, w, h = bbox(points[OVAL])
        x, y, w, h = clip_box((x-w*.12, y-h*.12, w*1.24, h*1.24), frame.shape)
        panels = [cutout]
        for data in (frame, result):
            photo = Image.fromarray(cv2.cvtColor(data[y:y+h, x:x+w], cv2.COLOR_BGR2RGB))
            photo.thumbnail((300, 340))
            panel = Image.new('RGB', (300, 340), '#eeeeee')
            panel.paste(photo, ((300-photo.width)//2, (340-photo.height)//2))
            panels.append(panel)
        sheet = Image.new('RGB', (900, 370), 'white')
        drawing = ImageDraw.Draw(sheet)
        for i, (title, panel) in enumerate(zip(('Source cutout', 'Driving face', 'Expression composite'), panels)):
            drawing.text((i*300+10, 8), title, fill='black')
            sheet.paste(panel, (i*300, 30))
        sheet.save(output/'comparison.png')
        (output/'metadata.json').write_text(json.dumps({'source': str(source), 'target': str(target),
            'triangles': len(asset.triangles), 'single_composite_seconds': elapsed,
            'note': 'Public still fixtures; not evidence of real-video expression quality.'}, indent=2), encoding='utf-8')
        print(output/'comparison.png')
        print(f'One frame compositing: {elapsed:.3f}s (includes lazy segmenter state if needed)')
    finally:
        model.close()


if __name__ == '__main__':
    main()
