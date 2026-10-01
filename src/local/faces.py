from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps
from .media import ROOT
from .matting import FaceSegmenter, face_matte, contour_mask, inward_feather
from .expression import anatomical_triangles, aperture_mask, dense_warp, match_lighting, multiband_blend
from .identity import CompositeSettings, IdentityState

MODEL = ROOT / 'models' / 'face_landmarker.task'
# Anatomical face boundary; semantic segmentation separately removes hair.
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
        365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
        132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


class Landmarker:
    def __init__(self, max_faces=5):
        if not MODEL.is_file():
            raise FileNotFoundError('얼굴 모델이 없습니다. setup_local.bat를 실행하세요.')
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=MODEL.read_bytes()),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=max_faces, min_face_detection_confidence=0.45,
            min_face_presence_confidence=0.45)
        self.task = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self.segmenter = None

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        result = self.task.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        return [np.array([(p.x * w, p.y * h) for p in face], np.float32)
                for face in result.face_landmarks]

    def close(self):
        self.task.close()
        if self.segmenter is not None:
            self.segmenter.close()

    def segment(self, bgr):
        if self.segmenter is None:
            self.segmenter = FaceSegmenter()
        return self.segmenter.probabilities(bgr)


def bbox(points):
    lo, hi = points.min(axis=0), points.max(axis=0)
    return np.array([*lo, *(hi-lo)], np.float32)


def clip_box(box, shape):
    x, y, w, h = map(float, box)
    height, width = shape[:2]
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(width, int(np.ceil(x+w))), min(height, int(np.ceil(y+h)))
    if x2-x1 < 8 or y2-y1 < 8:
        raise ValueError('얼굴 영역이 너무 작거나 화면 밖에 있습니다.')
    return x1, y1, x2-x1, y2-y1


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    lo = np.maximum(a[:2], b[:2])
    hi = np.minimum(a[:2]+a[2:], b[:2]+b[2:])
    intersection = np.prod(np.maximum(0, hi-lo))
    return float(intersection / max(1, np.prod(a[2:])+np.prod(b[2:])-intersection))


def is_frontal(points):
    left, right, nose = points[33], points[263], points[1]
    axis = right-left
    length = np.linalg.norm(axis)
    if length < 20:
        return False
    # Rotation-independent lateral position of the nose between the eyes.
    fraction = np.dot(nose-left, axis) / (length*length)
    offset = nose-(left+right)/2
    vertical = abs(axis[0]*offset[1]-axis[1]*offset[0]) / (length*length)
    return 0.25 <= fraction <= 0.75 and 0.08 < vertical < 1.1


@dataclass
class FaceAsset:
    image: np.ndarray
    alpha: np.ndarray
    points: np.ndarray
    triangles: np.ndarray
    path: Path


def semantic_for_box(frame, box, landmarker):
    x, y, w, h = box
    bx, by, bw, bh = clip_box((x-w*.5, y-h*.5, w*2, h*2), frame.shape)
    probabilities = landmarker.segment(frame[by:by+bh, bx:bx+bw])
    return probabilities[y-by:y-by+h, x-bx:x-bx+w]


def extract_face(path, output, landmarker):
    path, output = Path(path), Path(output)
    if path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
        raise ValueError('얼굴 이미지는 JPEG 또는 PNG를 선택하세요.')
    with Image.open(path) as opened:
        if opened.format not in {'JPEG', 'PNG'}:
            raise ValueError('실제 JPEG/PNG 이미지가 아닙니다.')
        image = ImageOps.exif_transpose(opened).convert('RGBA')
        image.thumbnail((1600, 1600))
        rgba = np.array(image)
    bgr = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR)
    faces = [p for p in landmarker.detect(bgr) if is_frontal(p)]
    if not faces:
        raise ValueError('정면 얼굴을 찾지 못했습니다. 얼굴이 크고 선명한 정면 사진을 선택하세요.')
    if len(faces) > 1:
        raise ValueError('정면 얼굴이 여러 개입니다. 사용할 사람 한 명만 있는 사진을 선택하세요.')
    points = faces[0]
    face_box = bbox(points[OVAL])
    padding = max(3, min(face_box[2:])*.045)
    x, y, w, h = clip_box((*(face_box[:2]-padding), *(face_box[2:]+2*padding)), bgr.shape)
    crop = bgr[y:y+h, x:x+w]
    probabilities = semantic_for_box(bgr, (x, y, w, h), landmarker)
    alpha = face_matte(crop, points[OVAL]-[x, y], probabilities, rgba[y:y+h, x:x+w, 3])
    if np.count_nonzero(alpha > 128) < w*h*.15:
        raise ValueError('헤어를 제외한 얼굴 피부 영역이 충분하지 않습니다. 가림 없는 정면 사진을 선택하세요.')
    scale = min(1., 512/max(w, h))
    size = (max(1, round(w*scale)), max(1, round(h*scale)))
    crop, alpha = cv2.resize(crop, size), cv2.resize(alpha, size)
    points = (points-np.array([x, y])) * np.array([size[0]/w, size[1]/h])
    points = points.astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.dstack([cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), alpha])).save(output)
    return FaceAsset(crop, alpha, points, anatomical_triangles(), output)


def composite(frame, asset, target_points=None, target_box=None, strength=1.0,
              landmarker=None, blend_state=None, settings=None, identity_state=None):
    """Dense expression mesh with semantic matting and preserved eye/mouth interiors."""
    settings = settings or CompositeSettings()
    box = bbox(target_points[OVAL]) if target_points is not None else target_box
    try:
        x, y, w, h = clip_box(box, frame.shape)
    except ValueError:
        return frame
    roi = frame[y:y+h, x:x+w].astype(np.float32)
    interiors = None
    interior_protection = np.ones((h, w), np.float32)
    if target_points is None:
        warped = cv2.resize(asset.image, (w, h)).astype(np.float32)
        alpha = cv2.resize(asset.alpha, (w, h)).astype(np.float32)/255
    else:
        target = (target_points - [x, y]).astype(np.float32)
        identity_state = identity_state if identity_state is not None else IdentityState()
        destination = identity_state.deform(asset.points, target, asset.triangles, settings.identity)
        warped, alpha = dense_warp(asset, destination, (h, w))
        alpha *= inward_feather(contour_mask((h, w), target[OVAL]), max(1.5, w*.02))
        if landmarker is not None:
            probabilities = semantic_for_box(frame, (x, y, w, h), landmarker)
            interior_protection = face_matte(frame[y:y+h, x:x+w], target[OVAL], probabilities)/255
            alpha *= interior_protection
        # Warp the uploaded lips/eyelids; retain the driving pupils, teeth and tongue.
        openings = aperture_mask((h, w), destination)
        openings[openings > .999] = 1.
        if settings.identity > 0 and not np.array_equal(target, destination):
            # Move driving pupils/teeth with their surrounding eyelids/lips.
            driving = FaceAsset(roi, np.full((h, w), 255, np.uint8), target, asset.triangles, asset.path)
            interiors, coverage = dense_warp(driving, destination, (h, w))
            interiors = np.where((coverage > .5)[..., None], interiors, roi)
        alpha *= 1-openings
    alpha = np.clip(alpha*strength, 0, 1)
    warped = match_lighting(warped, roi, alpha, blend_state, settings)
    # Extend colors into transparent pixels before pyramid filtering to avoid dark halos.
    warped = np.where((alpha > .01)[..., None], warped, roi)
    output = frame.copy()
    blended = multiband_blend(warped, roi, alpha)
    if interiors is not None:
        support = openings*interior_protection*np.clip(strength, 0, 1)
        blended = blended*(1-support[..., None])+interiors*support[..., None]
    output[y:y+h, x:x+w] = np.rint(blended).astype(np.uint8)
    return output
