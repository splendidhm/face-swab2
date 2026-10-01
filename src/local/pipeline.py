from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import cv2

from .faces import composite
from .media import Cancelled, export_mp4, video_info, write_json
from .tracking import SelectedTracker
from .expression import BlendState
from .identity import CompositeSettings, IdentityState


def render_video(video, asset, selections, destination, landmarker, progress=lambda *args: None,
                 cancel=None, force_region=False, settings=None):
    """Selection keyframes restart tracking; earlier and lost frames remain original."""
    settings = settings or CompositeSettings()
    video, destination = Path(video).resolve(), Path(destination).resolve()
    if video == destination or destination == asset.path.resolve():
        raise ValueError('입력 파일을 출력으로 덮어쓸 수 없습니다.')
    if destination.suffix.lower() != '.mp4':
        raise ValueError('출력 확장자는 .mp4여야 합니다.')
    info = video_info(video)
    if (info['width'], info['height']) != (1280, 720):
        raise ValueError('먼저 영상을 720p로 가져오세요.')
    if not selections or any(not 0 <= int(i) < info['frames'] for i in selections):
        raise ValueError('영상 속 얼굴을 드래그하여 선택하세요.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    replaced = 0
    skipped = []
    modes = set()
    mode_frames = {'mesh': 0, 'region': 0}
    with tempfile.TemporaryDirectory(prefix='faceswab-', dir=destination.parent) as temp:
        temp = Path(temp)
        raw = temp / 'frames.mkv'
        final = temp / 'final.mp4'
        cap = cv2.VideoCapture(str(video))
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'FFV1'), info['fps'], (1280, 720))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError('임시 영상 인코더를 열지 못했습니다.')
        tracker = None
        blend_state = BlendState()
        identity_states = []
        identity_state = IdentityState()
        index = 0
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    raise Cancelled('합성을 취소했습니다.')
                ok, frame = cap.read()
                if not ok:
                    break
                if index in selections:
                    tracker = SelectedTracker(frame, selections[index], landmarker, force_region)
                    blend_state = BlendState()
                    identity_state = IdentityState()
                    identity_states.append(identity_state)
                    state = tracker.current()
                    modes.add(tracker.mode)
                else:
                    state = tracker.update(frame) if tracker else None
                if state is not None:
                    frame = composite(frame, asset, state['points'], state['box'],
                                      landmarker=landmarker, blend_state=blend_state,
                                      settings=settings, identity_state=identity_state)
                    replaced += 1
                    mode_frames[state['mode']] += 1
                elif tracker is not None:
                    if skipped and skipped[-1][1] == index-1:
                        skipped[-1][1] = index
                    else:
                        skipped.append([index, index])
                writer.write(frame)
                index += 1
                if index % 5 == 0:
                    progress(index/info['frames']*.9, f'{index}/{info["frames"]} 프레임 · 합성 {replaced} · 추적 중단 {sum(b-a+1 for a,b in skipped)}')
        finally:
            cap.release()
            writer.release()
        if index != info['frames']:
            raise RuntimeError(f'영상 읽기가 중단되었습니다: {index}/{info["frames"]} 프레임')
        if replaced == 0:
            raise RuntimeError('합성된 프레임이 없습니다. 얼굴을 다시 선택하세요.')
        progress(.92, '720p H.264 MP4 · AAC 스테레오 인코딩 중…')
        export_mp4(raw, video, final, index/info['fps'], cancel)
        check = video_info(final)
        if check['frames'] != index or (check['width'], check['height']) != (1280, 720):
            raise RuntimeError('출력 영상 검증에 실패했습니다.')
        if cancel is not None and cancel.is_set():
            raise Cancelled('합성을 취소했습니다.')
        report = {'output': str(destination), 'frames': index, 'replaced_frames': replaced,
                  'unchanged_before_selection': min(selections), 'skipped_tracking_ranges': skipped,
                  'modes': sorted(modes), 'selections': selections, 'fps': info['fps'],
                  'resolution': [1280, 720], 'video_codec': 'h264', 'audio_codec': 'aac',
                  'audio_channels': 2, 'audio_sample_rate': 48000,
                  'compositor': 'dense-expression-v3', 'mesh_vertices': 468,
                  'mesh_triangles': len(asset.triangles), 'source_hair_segmentation': True,
                  'target_hair_segmentation': mode_frames['mesh'] > 0,
                  'mode_frames': mode_frames,
                  'expression_interiors': 'driving_eyes_and_inner_mouth' if mode_frames['mesh'] else None,
                  'intermediate_codec': 'ffv1',
                  'composite_settings': asdict(settings),
                  'identity_geometry_frames': sum(len(s.applied_strengths) for s in identity_states),
                  'identity_reduced_frames': sum(s.reduced_frames for s in identity_states),
                  'identity_reduction_steps': sum(s.reduction_steps for s in identity_states),
                  'identity_applied_strength_min': min((v for s in identity_states for v in s.applied_strengths), default=0.),
                  'identity_applied_strength_mean': float(sum(v for s in identity_states for v in s.applied_strengths)/max(1, sum(len(s.applied_strengths) for s in identity_states)))}
        write_json(temp/'report.json', report)
        os.replace(final, destination)
        os.replace(temp/'report.json', destination.with_suffix('.report.json'))
    progress(1., '합성 완료')
    return report
