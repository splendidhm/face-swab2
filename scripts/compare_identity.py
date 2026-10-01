"""Render and inspect three identity presets using identical inputs and selections."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.local.faces import Landmarker, composite, extract_face
from src.local.identity import CompositeSettings, IdentityState
from src.local.media import read_frame, run_ffmpeg, video_info, write_json
from src.local.pipeline import render_video
from src.local.tracking import SelectedTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, help='Normalized 720p 30fps video')
    parser.add_argument('--face', required=True)
    parser.add_argument('--selections', required=True)
    parser.add_argument('--out-dir', required=True, help='New directory for comparison artifacts')
    parser.add_argument('--preview-only', action='store_true')
    parser.add_argument('--presets', nargs='+', choices=('legacy','balanced','strong'),
                        default=['legacy','balanced','strong'], help='Render only these; other MP4s must already exist for comparison')
    args = parser.parse_args()
    folder = Path(args.out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    video = Path(args.video).resolve()
    if (folder/'source_face.png').resolve() == Path(args.face).resolve():
        parser.error('Output cutout must differ from source face image')
    if any((folder/f'{name}.mp4').resolve() == video for name in ('legacy','balanced','strong','comparison')):
        parser.error('Output directory would overwrite input')
    payload = json.loads(Path(args.selections).read_text(encoding='utf-8'))
    selections = {int(k):v for k,v in payload.get('selections', payload).items()}
    names = ('legacy', 'balanced', 'strong')
    model = Landmarker()
    try:
        asset = extract_face(args.face, folder/'source_face.png', model)
        info = video_info(video)
        frames = sorted(set(min(info['frames']-1, round(t*info['fps'])) for t in (0,2,6)))
        sheet = Image.new('RGB', (1200, 300*len(frames)), '#18202d')
        draw = ImageDraw.Draw(sheet)
        for row, index in enumerate(frames):
            frame = read_frame(video, index)
            anchor = max((i for i in selections if i <= index), default=min(selections))
            # These are static selection previews, separate from full temporal render.
            tracker = SelectedTracker(frame, selections[anchor], model)
            x,y,w,h = map(int, tracker.box)
            margin = max(10, round(max(w,h)*.15))
            x1,y1,x2,y2 = max(0,x-margin),max(0,y-margin),min(frame.shape[1],x+w+margin),min(frame.shape[0],y+h+margin)
            panels = [asset.image, frame[y1:y2,x1:x2]]
            for name in names:
                state = IdentityState()
                result = composite(frame, asset, tracker.points, tracker.box, landmarker=model,
                                   settings=CompositeSettings.preset(name), identity_state=state)
                panels.append(result[y1:y2,x1:x2])
                print(f'preview {index}: {name}, applied={state.applied_strengths}', flush=True)
            for col, (title, panel) in enumerate(zip(('UPLOADED FACE','ORIGINAL')+names, panels)):
                image = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
                image.thumbnail((224,260))
                sheet.paste(image,(col*240+(240-image.width)//2,row*300+32))
                draw.text((col*240+8,row*300+8),f'{title} {index/info["fps"]:.1f}s',fill='white')
        sheet.save(folder/'preset_previews.jpg', quality=95)
        if args.preview_only:
            return
        reports = json.loads((folder/'renders.json').read_text(encoding='utf-8')) if (folder/'renders.json').exists() else {}
        for name in args.presets:
            print(f'Start {name}', flush=True)
            start = time.perf_counter()
            def progress(p,s):
                if p >= .92 or round(p*100)%9 == 0:
                    print(f'{name}: {p:.0%} {s}', flush=True)
            report = render_video(video, asset, selections, folder/f'{name}.mp4', model,
                                  progress=progress, settings=CompositeSettings.preset(name))
            reports[name] = {'seconds':time.perf_counter()-start, 'report':report}
            write_json(folder/'renders.json', reports)
        # Four 320-pixel face panels plus source face reference; no stretching.
        first = SelectedTracker(read_frame(video), selections[min(selections)], model)
        x,y,w,h = map(int, first.box)
        size = min(360, video_info(video)['height'])
        cx,cy = x+w//2,y+h//2
        x = max(0,min(1280-size,cx-size//2)); y = max(0,min(720-size,cy-size//2))
        font = 'C\\:/Windows/Fonts/arial.ttf'
        filters = []
        for i, name in enumerate(('ORIGINAL','LEGACY','BALANCED','STRONG')):
            filters.append(f"[{i}:v]crop={size}:{size}:{x}:{y},scale=320:320,pad=320:720:0:230:color=0x18202d,drawtext=fontfile='{font}':text='{name}':x=12:y=190:fontsize=24:fontcolor=white[p{i}]")
        # Uploaded reference sits above the four synchronized target panels.
        filters.append('[p0][p1][p2][p3]hstack=inputs=4[base]')
        filters.append('[4:v]scale=140:140:force_original_aspect_ratio=decrease[face]')
        filters.append(f"[base][face]overlay=20:15,drawtext=fontfile='{font}':text='UPLOADED FACE / SYNTHETIC TEST':x=185:y=65:fontsize=25:fontcolor=white[v]")
        inputs = ['-i', video]
        for name in names:
            inputs += ['-i',folder/f'{name}.mp4']
        inputs += ['-loop','1','-i',folder/'source_face.png']
        run_ffmpeg(['-y',*inputs,'-filter_complex',';'.join(filters),'-map','[v]','-map','2:a:0',
                    '-t',str(info['duration']),'-c:v','libx264','-crf','18','-preset','veryfast',
                    '-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart',folder/'comparison.mp4'])
    finally:
        model.close()


if __name__ == '__main__':
    main()
