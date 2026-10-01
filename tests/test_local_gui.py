"""Hidden Tk integration test: coordinates, keyframes, worker completion and close."""
import time
import tkinter as tk
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.local.gui import FaceSwapGUI


class GuiTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = FaceSwapGUI(self.root)
        self.app.info = {'fps': 30, 'frames': 90, 'duration': 3}
        self.app.frame = np.zeros((720, 1280, 3), np.uint8)
        self.root.update_idletasks()

    def tearDown(self):
        if self.root.winfo_exists():
            self.app.close()

    def test_reverse_drag_maps_display_to_video(self):
        self.app.drag_begin(SimpleNamespace(x=260, y=195))
        self.app.drag_move(SimpleNamespace(x=130, y=65))
        self.app.drag_end(SimpleNamespace(x=130, y=65))
        self.assertEqual(self.app.selections[0], (200, 100, 200, 200))
        self.app.index = 30
        self.app.drag_begin(SimpleNamespace(x=0, y=0))
        self.app.drag_end(SimpleNamespace(x=65, y=65))
        self.assertEqual(len(self.app.selections), 2)
        self.app.clear_selection()
        self.assertEqual(list(self.app.selections), [0])

    def test_tiny_drag_does_not_create_selection(self):
        self.app.drag_begin(SimpleNamespace(x=40, y=40))
        self.app.drag_end(SimpleNamespace(x=42, y=42))
        self.assertFalse(self.app.selections)

    def test_worker_completion_runs_through_main_queue(self):
        output = []
        self.app.submit(lambda: 42, output.append, 'test')
        deadline = time.monotonic()+3
        while self.app.busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.02)
        self.assertEqual(output, [42])
        self.assertFalse(self.app.busy)

    def test_presets_region_controls_and_busy_restore(self):
        self.assertEqual(self.app.composite_settings().identity, .35)
        self.app.preset_name.set('특징 강화')
        self.app.apply_preset()
        self.assertEqual(self.app.composite_settings().identity, .55)
        self.app.force_region.set(True)
        self.app.region_changed()
        self.app.set_busy(True)
        self.app.set_busy(False)
        self.assertEqual(str(self.app.setting_controls['identity']['state']), 'disabled')
        self.assertEqual(str(self.app.preset_combo['state']), 'readonly')

    def test_preview_and_export_receive_identical_settings(self):
        self.app.asset = SimpleNamespace()
        self.app.video = Path('test-video.mp4')
        self.app.original = self.app.video.resolve()
        self.app.selections = {0: (100,100,200,200)}
        self.app.setting_vars['identity'].set(.45)
        self.app.mask_level.set(1)
        self.app.mask_changed()
        self.app.submit = lambda work, done, label: work()
        with patch('src.local.gui.Landmarker'), patch('src.local.gui.SelectedTracker'), \
             patch('src.local.gui.composite', return_value=self.app.frame) as preview, \
             patch('src.local.gui.render_video') as render, \
             patch('src.local.gui.filedialog.asksaveasfilename', return_value='test-output.mp4'):
            self.app.refresh_preview()
            self.app.export()
        self.assertEqual(preview.call_args.kwargs['settings'], render.call_args.kwargs['settings'])
        self.assertEqual(render.call_args.kwargs['settings'].identity, .45)
        self.assertEqual(preview.call_args.kwargs['strength'], .5)
        self.assertEqual(render.call_args.kwargs['mask_control'].snapshot(), ('midium', .5))

    def test_mask_mouse_wheel_and_buttons_remain_live_while_busy(self):
        self.app.rendering = True
        self.app.set_busy(True)
        self.app.mask_wheel(SimpleNamespace(delta=-120))
        self.assertEqual(self.app.mask_level.get(),2)
        self.assertEqual(self.app.mask_control.snapshot(),('strong', .75))
        self.app.mask_controls[1].invoke()
        self.assertEqual(self.app.mask_control.snapshot(),('low', .25))
        self.app.mask_wheel(SimpleNamespace(delta=-120))
        self.assertEqual(self.app.mask_level.get(),0)
        self.app.set_mask_enabled(False)
        self.app.mask_wheel(SimpleNamespace(delta=120))
        self.assertEqual(self.app.mask_level.get(),0)
        self.app.set_busy(False)
        self.assertEqual(str(self.app.mask_scale['state']), 'normal')

    def test_mask_preview_is_debounced_and_uses_latest_selection(self):
        with patch.object(self.app,'refresh_preview') as preview:
            for level in (0,1,2):
                self.app.mask_level.set(level)
                self.app.mask_changed()
            deadline=time.monotonic()+.5
            while time.monotonic()<deadline:
                self.root.update()
                time.sleep(.02)
            self.assertEqual(preview.call_count,1)
            self.assertEqual(self.app.mask_control.snapshot(),('strong', .75))


if __name__ == '__main__':
    unittest.main()
