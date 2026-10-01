"""Semantic face/hair separation and inward-only, subpixel face mattes."""
import cv2
import mediapipe as mp
import numpy as np
from scipy.interpolate import CubicSpline

from .media import ROOT

SEGMENT_MODEL = ROOT/'models'/'selfie_multiclass_256x256.tflite'


def smoothstep(value, low=0., high=1.):
    t = np.clip((value-low)/(high-low), 0, 1)
    return t*t*(3-2*t)


def contour_mask(shape, contour):
    """Interpolate the ordered jaw/forehead contour, without a convex hull."""
    points = np.asarray(contour, np.float32)
    closed = np.vstack([points, points[0]])
    curve = CubicSpline(np.arange(len(closed)), closed, bc_type='periodic')
    samples = curve(np.linspace(0, len(points), len(points)*8, endpoint=False))
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.rint(samples*16).astype(np.int32)], 255, lineType=cv2.LINE_AA, shift=4)
    return mask.astype(np.float32)/255


def inward_feather(mask, radius):
    """Keep pixels outside the matte exactly zero; no hair/background bleed."""
    binary = (mask > .05).astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.asarray(mask, np.float32)*smoothstep(distance, 0., max(1., radius))


class FaceSegmenter:
    def __init__(self):
        if not SEGMENT_MODEL.is_file():
            raise FileNotFoundError('얼굴/헤어 분할 모델이 없습니다. setup_local.bat를 실행하세요.')
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=SEGMENT_MODEL.read_bytes()),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            output_confidence_masks=True, output_category_mask=False)
        self.task = mp.tasks.vision.ImageSegmenter.create_from_options(options)

    def probabilities(self, bgr):
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        result = self.task.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        # Official class order: background, hair, body skin, face skin, clothes, accessories.
        return np.stack([mask.numpy_view().copy().reshape(bgr.shape[:2])
                         for mask in result.confidence_masks], axis=-1)

    def close(self):
        self.task.close()


def face_matte(bgr, contour, probabilities, original_alpha=None):
    geometry = contour_mask(bgr.shape[:2], contour)
    face = probabilities[..., 3].astype(np.float32)
    # Edge-aware refinement keeps low-resolution semantic boundaries aligned to image edges.
    guide = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    radius = max(2, round(min(bgr.shape[:2])*.012))
    face = cv2.ximgproc.guidedFilter(guide, face, radius, 1e-3)
    semantic = smoothstep(face, .22, .72)
    semantic[probabilities[..., 1] >= .65] = 0
    alpha = inward_feather(geometry*semantic, max(1.5, min(bgr.shape[:2])*.018))
    if original_alpha is not None:
        alpha *= original_alpha.astype(np.float32)/255
    return np.rint(np.clip(alpha, 0, 1)*255).astype(np.uint8)
