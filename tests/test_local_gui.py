"""Hidden Tk integration test: coordinates, keyframes, worker completion and close."""
import time
import tkinter as tk
import unittest
from types import SimpleNamespace

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
            self.root.destroy()

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


if __name__ == '__main__':
    unittest.main()
