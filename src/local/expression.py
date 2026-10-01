"""Anatomical dense mesh warping and expression-preserving photometric blending."""
import cv2
import numpy as np
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

from .matting import contour_mask, inward_feather

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
INNER_MOUTH = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LIPS = sorted({v for e in FaceLandmarksConnections.FACE_LANDMARKS_LIPS for v in (e.start, e.end)})
EXPRESSION_INDICES = np.array(sorted(set(LEFT_EYE+RIGHT_EYE+LIPS)))


def anatomical_triangles():
    """Fixed topology avoids Delaunay flipping as a mouth closes or eyes blink."""
    edges = FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
    triangles = []
    for i in range(0, len(edges), 3):
        indices = sorted({v for e in edges[i:i+3] for v in (e.start, e.end)})
        if len(indices) != 3 or max(indices) >= 468:
            raise RuntimeError('Unexpected MediaPipe face mesh topology')
        triangles.append(indices)
    return np.array(triangles, np.int32)


def aperture_mask(shape, points):
    """Preserve driving eye apertures and inner mouth, never whole lips/eyelids."""
    mask = np.zeros(shape[:2], np.float32)
    width = np.linalg.norm(points[33]-points[263])
    for ring in (LEFT_EYE, RIGHT_EYE, INNER_MOUTH):
        contour = points[ring]
        # Closed eyes/lips remain a thin moving line rather than a source texture hole.
        opening = contour_mask(shape, contour)
        if cv2.contourArea(contour.astype(np.float32)) > max(1., width*.012):
            opening = cv2.GaussianBlur(opening, (0, 0), max(.45, width*.004))
        mask = np.maximum(mask, opening)
    return np.clip(mask, 0, 1)


def dense_warp(asset, points, shape):
    """Local triangle rasterization with premultiplied alpha and overlap averaging."""
    h, w = shape[:2]
    premult = np.dstack([asset.image.astype(np.float32)*(asset.alpha[..., None]/255),
                        asset.alpha.astype(np.float32)/255])
    merged = np.zeros((h, w, 4), np.float32)
    weights = np.zeros((h, w), np.float32)
    for ids in asset.triangles:
        src, dst = asset.points[ids].astype(np.float32), points[ids].astype(np.float32)
        src_edges, dst_edges = src[1:]-src[0], dst[1:]-dst[0]
        source_area, target_area = np.linalg.det(src_edges), np.linalg.det(dst_edges)
        if abs(source_area) < .02 or abs(target_area) < .02 or source_area*target_area < 0:
            continue  # Occluded/inverted triangles must not smear across adjacent features.
        bx, by, bw, bh = cv2.boundingRect(dst)
        x1, y1, x2, y2 = max(0, bx), max(0, by), min(w, bx+bw), min(h, by+bh)
        if x2 <= x1 or y2 <= y1:
            continue
        local = dst-[x1, y1]
        matrix = cv2.getAffineTransform(src, local.astype(np.float32))
        piece = cv2.warpAffine(premult, matrix, (x2-x1, y2-y1), flags=cv2.INTER_LINEAR)
        coverage = np.zeros((y2-y1, x2-x1), np.uint8)
        cv2.fillConvexPoly(coverage, np.rint(local*16).astype(np.int32), 255, lineType=cv2.LINE_AA, shift=4)
        weight = coverage.astype(np.float32)/255
        merged[y1:y2, x1:x2] += piece*weight[..., None]
        weights[y1:y2, x1:x2] += weight
    merged /= np.maximum(weights[..., None], 1e-6)
    alpha = np.clip(merged[..., 3], 0, 1)
    return merged[..., :3]/np.maximum(alpha[..., None], 1e-6), alpha


def stabilize_landmarks(detected, predicted):
    """Motion-aligned adaptive filter; expression landmarks respond without fixed lag."""
    width = max(20., np.linalg.norm(detected[33]-detected[263]))
    motion = np.linalg.norm(detected-predicted, axis=1)/width
    gain = np.clip(.35 + motion*28, .35, .96)
    gain[EXPRESSION_INDICES] = np.maximum(gain[EXPRESSION_INDICES], .9)
    return (detected*gain[:, None]+predicted*(1-gain[:, None])).astype(np.float32)


class BlendState:
    def __init__(self):
        self.shift = None

    def color_shift(self, shift):
        self.shift = shift if self.shift is None else self.shift*.75+shift*.25
        return self.shift


def match_lighting(warped, target, alpha, state=None):
    """Transfer local illumination, retaining uploaded skin texture and bounded color."""
    inside = alpha > .8
    if np.count_nonzero(inside) < 30:
        return warped
    source_lab = cv2.cvtColor(np.clip(warped, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    shift = np.clip(np.median(target_lab[inside], axis=0)-np.median(source_lab[inside], axis=0), [-45, -18, -18], [45, 18, 18])
    if state is not None:
        shift = state.color_shift(shift)
    source_lab += shift*np.array([.65, .45, .45])
    # Local mid-frequency light variations carry cheek/nasolabial expression cues.
    sigma = max(2., min(alpha.shape)*.025)
    local = target_lab[..., 0]-cv2.GaussianBlur(target_lab[..., 0], (0, 0), sigma)
    source_lab[..., 0] += np.clip(local, -12, 12)*.25
    return cv2.cvtColor(np.clip(source_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)


def multiband_blend(source, target, alpha, levels=3):
    """Blend coarse illumination over wider bands and retain fine skin detail."""
    source_pyramid, target_pyramid, mask_pyramid = [source.astype(np.float32)], [target.astype(np.float32)], [alpha.astype(np.float32)]
    for _ in range(levels):
        if min(source_pyramid[-1].shape[:2]) < 8:
            break
        source_pyramid.append(cv2.pyrDown(source_pyramid[-1]))
        target_pyramid.append(cv2.pyrDown(target_pyramid[-1]))
        mask_pyramid.append(cv2.pyrDown(mask_pyramid[-1]))
    result = source_pyramid[-1]*mask_pyramid[-1][..., None]+target_pyramid[-1]*(1-mask_pyramid[-1][..., None])
    for level in range(len(source_pyramid)-2, -1, -1):
        size = source_pyramid[level].shape[1], source_pyramid[level].shape[0]
        a = source_pyramid[level]-cv2.pyrUp(source_pyramid[level+1], dstsize=size)
        b = target_pyramid[level]-cv2.pyrUp(target_pyramid[level+1], dstsize=size)
        mask = mask_pyramid[level][..., None]
        result = cv2.pyrUp(result, dstsize=size)+a*mask+b*(1-mask)
    # Pyramid supports cannot change the background or preserved expression interiors.
    support = np.clip(alpha*8, 0, 1)[..., None]
    return np.clip(result*support+target*(1-support), 0, 255)
