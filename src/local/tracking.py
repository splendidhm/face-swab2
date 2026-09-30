from __future__ import annotations

import cv2
import numpy as np

from .faces import OVAL, bbox, clip_box, iou


def face_in_region(frame, box, landmarker):
    x, y, w, h = clip_box(box, frame.shape)
    # Context margin improves detection without selecting an unrelated face.
    margin = .25
    x0, y0, cw, ch = clip_box((x-w*margin, y-h*margin, w*(1+2*margin), h*(1+2*margin)), frame.shape)
    faces = landmarker.detect(frame[y0:y0+ch, x0:x0+cw])
    candidates = []
    for face in faces:
        face = face + np.array([x0, y0], np.float32)
        fb = bbox(face[OVAL])
        cx, cy = fb[:2]+fb[2:]/2
        if x <= cx <= x+w and y <= cy <= y+h and iou(fb, box) > .15:
            candidates.append((iou(fb, box), face))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


class SelectedTracker:
    """Track only the user-selected ROI; never search the full frame on loss."""
    def __init__(self, frame, box, landmarker, force_region=False):
        self.landmarker = landmarker
        self.box = np.array(clip_box(box, frame.shape), np.float32)
        self.points = None if force_region else face_in_region(frame, self.box, landmarker)
        self.mode = 'mesh' if self.points is not None else 'region'
        if self.points is not None:
            self.box = bbox(self.points[OVAL])
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, tuple(map(int, clip_box(self.box, frame.shape))))
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.hist = self.histogram(frame, self.box)
        self.global_hist = self.global_histogram(frame)
        self.lost = False
        self.mesh_missing = 0

    @staticmethod
    def histogram(frame, box):
        x, y, w, h = clip_box(box, frame.shape)
        hsv = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
        return cv2.normalize(cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256]), None).flatten()

    @staticmethod
    def global_histogram(frame):
        hsv = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2HSV)
        return cv2.normalize(cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256]), None).flatten()

    def update(self, frame):
        if self.lost:
            return None
        current_hist = self.global_histogram(frame)
        if cv2.compareHist(self.global_hist, current_hist, cv2.HISTCMP_BHATTACHARYYA) > .72:
            self.lost = True
            return None
        self.global_hist = current_hist
        success, new_box = self.tracker.update(frame)
        if not success:
            self.lost = True
            return None
        new_box = np.array(new_box, np.float32)
        try:
            new_box = np.array(clip_box(new_box, frame.shape), np.float32)
            appearance = self.histogram(frame, new_box)
        except ValueError:
            self.lost = True
            return None
        movement = np.linalg.norm((new_box[:2]+new_box[2:]/2) - (self.box[:2]+self.box[2:]/2))
        ratio = np.prod(new_box[2:])/max(1, np.prod(self.box[2:]))
        if (movement > max(self.box[2:])*.75 or not .4 < ratio < 2.5 or
                cv2.compareHist(self.hist, appearance, cv2.HISTCMP_BHATTACHARYYA) > .8):
            self.lost = True
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.mode == 'mesh':
            predicted = (self.points-self.box[:2]) * (new_box[2:]/self.box[2:]) + new_box[:2]
            # Optical flow provides a motion-aligned reference for landmark smoothing.
            prev = self.points.astype(np.float32).reshape(-1, 1, 2)
            flow, status, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, prev, None)
            if flow is not None:
                back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, flow, None)
                if back is not None:
                    good = ((status.ravel() > 0) & (back_status.ravel() > 0) &
                            (np.linalg.norm(back[:, 0]-prev[:, 0], axis=1) < 2))
                    predicted[good] = flow[:, 0][good]
            detected = face_in_region(frame, new_box, self.landmarker)
            if detected is None:
                self.mesh_missing += 1
                # Skip immediately when occluded; short gaps can recover in the same ROI.
                self.gray, self.box = gray, new_box
                self.points = predicted
                if self.mesh_missing > 5:
                    self.lost = True
                return None
            if iou(bbox(detected[OVAL]), new_box) < .25:
                self.lost = True
                return None
            self.mesh_missing = 0
            self.points = detected*.8 + predicted*.2
        self.gray, self.box = gray, new_box
        return {'points': self.points, 'box': self.box, 'mode': self.mode}

    def current(self):
        return {'points': self.points, 'box': self.box, 'mode': self.mode}
