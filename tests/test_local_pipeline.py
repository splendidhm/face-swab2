r"""Regression tests for real model inference, selected-face tracking and export.

Download the public astronaut fixture using the URL in LOCAL_GUIDE.ko.md first.
Run: .venv\Scripts\python -m unittest discover -s tests -p test_local_pipeline.py -v
"""
import json
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.local.faces import MODEL, ROOT, OVAL, Landmarker, bbox, composite, extract_face
from src.local.media import Cancelled, ffmpeg, normalize_video, read_frame, run_ffmpeg, video_info
from src.local.pipeline import render_video
from src.local.tracking import SelectedTracker

FIXTURE = ROOT/'workspace'/'test-assets'/'astronaut.png'


@unittest.skipUnless(MODEL.exists() and FIXTURE.exists(), 'Download model and astronaut fixture first')
class LocalPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Landmarker()
        cls.temp = tempfile.TemporaryDirectory(prefix='얼굴 테스트 ', dir=ROOT/'workspace')
        cls.folder = Path(cls.temp.name)
        cls.asset = extract_face(FIXTURE, cls.folder/'누끼.png', cls.model)
        cls.photo = cv2.cvtColor(np.array(Image.open(FIXTURE).convert('RGB')), cv2.COLOR_RGB2BGR)
        cls.raw = cls.folder/'원본 영상.mp4'
        writer = cv2.VideoWriter(str(cls.raw), cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))
        assert writer.isOpened()
        for i in range(18):
            frame = np.full((720, 1280, 3), 45, np.uint8)
            frame[80:592, 30+i*2:542+i*2] = cls.photo
            frame[80:592, 735:1247] = cls.photo
            writer.write(frame)
        writer.release()
        cls.normalized = cls.folder/'준비.mp4'
        normalize_video(cls.raw, cls.normalized)
        cls.frame = read_frame(cls.normalized)
        cls.points = cls.model.detect(cls.photo)[0] + np.array([30, 80], np.float32)
        cls.box = tuple(map(float, bbox(cls.points[OVAL])))

    @classmethod
    def tearDownClass(cls):
        cls.model.close()
        cls.temp.cleanup()

    def test_frontal_cutout_has_transparency_and_no_background(self):
        with Image.open(self.asset.path) as image:
            self.assertEqual(image.mode, 'RGBA')
        self.assertEqual(self.asset.alpha.min(), 0)
        self.assertEqual(self.asset.alpha.max(), 255)
        self.assertGreater(np.count_nonzero(self.asset.alpha == 0), 50)
        self.assertLess(self.asset.image.shape[0], 200)

    def test_jpeg_and_no_face_validation(self):
        jpeg = self.folder/'정면 사진.jpeg'
        with Image.open(FIXTURE) as image:
            image.convert('RGB').save(jpeg)
        self.assertTrue(extract_face(jpeg, self.folder/'jpeg-face.png', self.model).path.exists())
        blank = self.folder/'빈 이미지.png'
        Image.new('RGB', (400, 400), 'white').save(blank)
        with self.assertRaisesRegex(ValueError, '정면 얼굴'):
            extract_face(blank, self.folder/'bad.png', self.model)

    def test_composite_changes_only_selected_face(self):
        result = composite(self.frame, self.asset, self.points, self.box)
        self.assertEqual(result.shape, self.frame.shape)
        self.assertTrue(np.array_equal(result[:, 700:], self.frame[:, 700:]))
        self.assertGreater(np.abs(result.astype(float)-self.frame).sum(), 1000)

    def test_tracker_follows_moving_selected_face(self):
        tracker = SelectedTracker(self.frame, self.box, self.model)
        self.assertEqual(tracker.mode, 'mesh')
        for i in range(1, 10):
            state = tracker.update(read_frame(self.normalized, i))
            self.assertIsNotNone(state)
            self.assertLess(state['box'][0], 600)
        self.assertGreater(state['box'][0], self.box[0]+5)

    def test_scene_cut_does_not_jump_to_other_face(self):
        tracker = SelectedTracker(self.frame, self.box, self.model)
        cut = np.zeros_like(self.frame)
        cut[:] = (0, 230, 0)
        self.assertIsNone(tracker.update(cut))
        self.assertTrue(tracker.lost)
        self.assertIsNone(tracker.update(self.frame))

    def test_region_mode_for_cartoon(self):
        tracker = SelectedTracker(self.frame, self.box, self.model, force_region=True)
        self.assertEqual(tracker.mode, 'region')
        state = tracker.update(read_frame(self.normalized, 1))
        self.assertIsNotNone(state)
        result = composite(self.frame, self.asset, None, state['box'])
        self.assertFalse(np.array_equal(result, self.frame))

    def test_export_720p_h264_stereo_silent_input_and_anchor(self):
        output = self.folder/'결과 영상.mp4'
        # A late selection leaves earlier frames unchanged; the full duration is kept.
        report = render_video(self.normalized, self.asset, {3: self.box}, output, self.model)
        info = video_info(output)
        self.assertEqual((info['width'], info['height'], info['frames']), (1280, 720, 18))
        self.assertEqual(report['unchanged_before_selection'], 3)
        self.assertEqual(report['replaced_frames'], 15)
        self.assertEqual(report['skipped_tracking_ranges'], [])
        self.assertEqual(json.loads(output.with_suffix('.report.json').read_text(encoding='utf8'))['audio_channels'], 2)
        metadata = subprocess.run([ffmpeg(), '-hide_banner', '-i', str(output)], capture_output=True).stderr.decode('utf8', 'replace')
        self.assertIn('Video: h264', metadata)
        self.assertIn('Audio: aac', metadata)
        self.assertIn('48000 Hz, stereo', metadata)
        decoded = subprocess.run([ffmpeg(), '-v', 'error', '-i', str(output), '-f', 'null', '-'], capture_output=True)
        self.assertEqual(decoded.returncode, 0, decoded.stderr)

    def test_mono_audio_preserved_as_stereo(self):
        mono = self.folder/'mono.mp4'
        run_ffmpeg(['-y', '-i', self.raw, '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000',
                    '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-ac', '1', '-t', '0.6', mono])
        normalized = self.folder/'mono-normalized.mp4'
        normalize_video(mono, normalized)
        output = self.folder/'stereo.mp4'
        render_video(normalized, self.asset, {0: self.box}, output, self.model, force_region=True)
        audio = subprocess.run([ffmpeg(), '-v', 'error', '-i', str(output), '-vn', '-f', 'f32le', '-ac', '2', '-'], capture_output=True)
        self.assertEqual(audio.returncode, 0)
        samples = np.frombuffer(audio.stdout, np.float32).reshape(-1, 2)
        self.assertGreater(np.sqrt(np.mean(samples**2)), .01)
        self.assertLess(np.mean(np.abs(samples[:, 0]-samples[:, 1])), .005)

    def test_cancel_and_output_protection(self):
        event = threading.Event()
        event.set()
        output = self.folder/'cancel.mp4'
        with self.assertRaises(Cancelled):
            render_video(self.normalized, self.asset, {0: self.box}, output, self.model, cancel=event)
        self.assertFalse(output.exists())
        with self.assertRaises(ValueError):
            render_video(self.normalized, self.asset, {0: self.box}, self.normalized, self.model)

    def test_portrait_input_is_letterboxed_without_stretching(self):
        portrait = self.folder/'portrait.mp4'
        writer = cv2.VideoWriter(str(portrait), cv2.VideoWriter_fourcc(*'mp4v'), 30, (180, 320))
        image = np.zeros((320, 180, 3), np.uint8)
        image[:] = (0, 0, 220)
        for _ in range(3):
            writer.write(image)
        writer.release()
        ready = self.folder/'portrait-ready.mp4'
        normalize_video(portrait, ready)
        frame = read_frame(ready)
        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertLess(frame[:, :400].mean(), 2)
        self.assertGreater(frame[:, 450:830, 2].mean(), 200)

    def test_new_selection_resumes_after_scene_cut(self):
        source = self.folder/'cut-and-reselect.mp4'
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))
        for i in range(12):
            if 4 <= i < 8:
                frame = np.full((720, 1280, 3), (0, 230, 0), np.uint8)
            else:
                frame = self.frame
            writer.write(frame)
        writer.release()
        report = render_video(source, self.asset, {0: self.box, 8: self.box}, self.folder/'reselected.mp4', self.model)
        self.assertEqual(report['replaced_frames'], 8)
        self.assertEqual(report['skipped_tracking_ranges'], [[4, 7]])

    def test_nonhuman_cartoon_uses_region_tracking(self):
        image = np.full((720, 1280, 3), 200, np.uint8)
        cv2.rectangle(image, (220, 160), (380, 360), (220, 40, 180), -1)
        cv2.circle(image, (255, 210), 18, (10, 10, 10), -1)
        cv2.rectangle(image, (320, 200), (355, 220), (10, 10, 10), -1)
        cv2.line(image, (250, 315), (345, 290), (10, 10, 10), 8)
        tracker = SelectedTracker(image, (215, 155, 170, 210), self.model)
        self.assertEqual(tracker.mode, 'region')
        moved = cv2.warpAffine(image, np.float32([[1, 0, 8], [0, 1, 0]]), (1280, 720), borderValue=(200, 200, 200))
        state = tracker.update(moved)
        self.assertIsNotNone(state)
        self.assertGreater(state['box'][0], 218)


if __name__ == '__main__':
    unittest.main()
