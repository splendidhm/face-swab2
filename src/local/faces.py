from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps
from scipy.spatial import Delaunay

from .media import ROOT

MODEL = ROOT / 'models' / 'face_landmarker.task'
# MediaPipe face oval. This excludes hair, neck, and background.
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
        365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
        132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
FEATURES = [33, 133, 160, 159, 158, 144, 145, 153, 362, 263, 385, 386,
            387, 380, 374, 373, 70, 63, 105, 66, 107, 336, 296, 334, 293,
            300, 1, 4, 6, 168, 2, 98, 327, 61, 291, 0, 17, 13, 14, 78,
            308, 37, 267, 84, 314, 205, 425, 50, 280, 187, 411, 199]
KEYS = np.array(sorted(set(OVAL + FEATURES)))


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

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        result = self.task.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        return [np.array([(p.x * w, p.y * h) for p in face], np.float32)
                for face in result.face_landmarks]

    def close(self):
        self.task.close()


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
    x, y, w, h = clip_box(bbox(points[OVAL]), bgr.shape)
    mask = np.zeros(bgr.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points[OVAL].astype(np.int32)), 255)
    mask = cv2.GaussianBlur(mask, (7, 7), 1.5)
    mask = np.minimum(mask, rgba[..., 3])
    crop, alpha = bgr[y:y+h, x:x+w], mask[y:y+h, x:x+w]
    scale = min(1., 512/max(w, h))
    size = (max(1, round(w*scale)), max(1, round(h*scale)))
    crop, alpha = cv2.resize(crop, size), cv2.resize(alpha, size)
    points = (points-np.array([x, y])) * np.array([size[0]/w, size[1]/h])
    points = points.astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.dstack([cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), alpha])).save(output)
    return FaceAsset(crop, alpha, points, Delaunay(points[KEYS]).simplices, output)


def composite(frame, asset, target_points=None, target_box=None, strength=1.0):
    """Warp only the local face ROI; all alpha/image coordinates stay aligned."""
    box = bbox(target_points[OVAL]) if target_points is not None else target_box
    try:
        x, y, w, h = clip_box(box, frame.shape)
    except ValueError:
        return frame
    if target_points is None:
        warped = cv2.resize(asset.image, (w, h)).astype(np.float32)
        alpha = cv2.resize(asset.alpha, (w, h)).astype(np.float32)/255
    else:
        source = asset.points[KEYS].astype(np.float32)
        target = (target_points[KEYS] - [x, y]).astype(np.float32)
        # Premultiplied alpha avoids dark borders around the cutout.
        premult = np.dstack([asset.image.astype(np.float32) * (asset.alpha[..., None]/255),
                            asset.alpha.astype(np.float32)/255])
        merged = np.zeros((h, w, 4), np.float32)
        for indices in asset.triangles:
            src, dst = source[indices], target[indices]
            bx, by, bw, bh = cv2.boundingRect(dst)
            x1, y1, x2, y2 = max(0, bx), max(0, by), min(w, bx+bw), min(h, by+bh)
            if x2 <= x1 or y2 <= y1:
                continue
            local = dst - [x1, y1]
            transform = cv2.getAffineTransform(src, local.astype(np.float32))
            piece = cv2.warpAffine(premult, transform, (x2-x1, y2-y1), flags=cv2.INTER_LINEAR)
            triangle_mask = np.zeros((y2-y1, x2-x1), np.uint8)
            cv2.fillConvexPoly(triangle_mask, np.rint(local).astype(np.int32), 255)
            region = merged[y1:y2, x1:x2]
            region[triangle_mask > 0] = piece[triangle_mask > 0]
        alpha = np.clip(merged[..., 3], 0, 1)
        warped = merged[..., :3] / np.maximum(alpha[..., None], 1e-5)
    roi = frame[y:y+h, x:x+w].astype(np.float32)
    interior = alpha > .8
    if interior.sum() > 20:
        # Restrained illumination correction preserves the uploaded face's color.
        shift = roi[interior].mean(0) - warped[interior].mean(0)
        warped = np.clip(warped + shift*.3, 0, 255)
    a = np.clip(alpha * strength, 0, 1)[..., None]
    output = frame.copy()
    output[y:y+h, x:x+w] = np.clip(warped*a + roi*(1-a), 0, 255).astype(np.uint8)
    return output
