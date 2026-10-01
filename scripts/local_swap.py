"""Reproducible CLI for the same engine used by the GUI."""
import argparse
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.local.faces import Landmarker, extract_face
from src.local.media import WORKSPACE, normalize_video
from src.local.pipeline import render_video
from src.local.identity import CompositeSettings


def main():
    parser = argparse.ArgumentParser(description='Selected-face cutout compositing, 720p MP4 stereo')
    parser.add_argument('--video', required=True)
    parser.add_argument('--face', required=True)
    parser.add_argument('--out', required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--roi', nargs=4, type=int, metavar=('X', 'Y', 'W', 'H'), help='Coordinates in normalized 1280x720 video')
    selection.add_argument('--selections', help='JSON report or {frame_index: [x,y,w,h]} in normalized 30fps video')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--region-mode', action='store_true')
    parser.add_argument('--preset', choices=('legacy', 'balanced', 'strong'), default='balanced')
    for name in ('identity', 'lighting', 'skin-color', 'detail'):
        parser.add_argument('--'+name, type=float, help='Override preset coefficient (identity: 0-0.65; others: 0-1)')
    args = parser.parse_args()
    try:
        settings = replace(CompositeSettings.preset(args.preset), **{
            name: getattr(args, name) for name in ('identity', 'lighting', 'skin_color', 'detail')
            if getattr(args, name) is not None})
    except ValueError as error:
        parser.error(str(error))
    if Path(args.out).resolve() in {Path(args.video).resolve(), Path(args.face).resolve()}:
        parser.error('Output must differ from input files')
    selections = {args.start_frame: args.roi}
    if args.selections:
        payload = json.loads(Path(args.selections).read_text(encoding='utf-8'))
        selections = {int(k): v for k, v in payload.get('selections', payload).items()}
    WORKSPACE.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=WORKSPACE) as temp:
        temp = Path(temp)
        video = temp/'normalized.mp4'
        normalize_video(args.video, video)
        model = Landmarker()
        try:
            asset = extract_face(args.face, temp/'cutout.png', model)
            report = render_video(video, asset, selections, args.out, model,
                                  lambda p, message: print(f'{p:.0%} {message}', flush=True),
                                  force_region=args.region_mode, settings=settings)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        finally:
            model.close()


if __name__ == '__main__':
    main()
