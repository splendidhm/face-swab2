"""Bounded source morphology without transferring a still photo's expression."""
from dataclasses import asdict, dataclass

import numpy as np

from .expression import LEFT_EYE, RIGHT_EYE, LIPS

OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,
        400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
NOSE = [1,2,4,5,6,19,94,97,98,168,195,197,326,327]


def feature_field(points, ring, field):
    """Affine feature core with a smooth spatial falloff, including neighboring skin."""
    center = (points[ring].min(0)+points[ring].max(0))*.5
    radius = np.maximum(np.abs(points[ring]-center).max(0)+.025, [.05,.06])
    distance = np.max(np.abs(points-center)/radius, axis=1)
    fade = np.clip((distance-1)/1.5, 0, 1)
    weight = 1-fade*fade*(3-2*fade)
    return field, weight


@dataclass(frozen=True)
class CompositeSettings:
    identity: float = 0.0
    lighting: float = 0.65
    skin_color: float = 0.45
    detail: float = 0.25

    def __post_init__(self):
        for name, value in asdict(self).items():
            maximum = .65 if name == 'identity' else 1.
            if not np.isfinite(value) or not 0 <= value <= maximum:
                raise ValueError(f'{name} must be finite and between 0 and {maximum}')

    @classmethod
    def preset(cls, name):
        return cls(*{'legacy': (0., .65, .45, .25),
                     'balanced': (.35, .50, .25, .10),
                     'strong': (.55, .40, .15, .05)}[name])


def canonical(points):
    axis = points[263]-points[33]
    axis = axis / max(float(np.linalg.norm(axis)), 1e-6)
    basis = np.array([axis, [-axis[1], axis[0]]], np.float32)
    origin = (points[33]+points[263])*.5
    scale = max(float(np.linalg.norm(points[454]-points[234])), 1.)
    return (points-origin) @ basis.T / scale, basis, scale


def safe_geometry(points, displacement, triangles, strength):
    """Never introduce a fold or collapse into previously usable target triangles."""
    def areas(p):
        v = p[triangles]
        a, b = v[:, 1]-v[:, 0], v[:, 2]-v[:, 0]
        return a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
    original = areas(points)
    valid = np.abs(original) >= .02
    reductions = 0
    while strength > 1e-4:
        candidate = points+displacement*strength
        changed = areas(candidate)
        if np.all(original[valid]*changed[valid] > 0) and np.all(np.abs(changed[valid]) >= np.abs(original[valid])*.15):
            return candidate.astype(np.float32), strength, reductions
        strength *= .5
        reductions += 1
    return points.copy(), 0., reductions


class IdentityState:
    def __init__(self):
        self.reference = None
        self.reduced_frames = 0
        self.reduction_steps = 0
        self.applied_strengths = []
        self.safe_limit = None

    def deform(self, source, target, triangles, strength):
        if strength == 0:
            return target.copy()
        src, _, _ = canonical(source)
        current, basis, scale = canonical(target)
        if self.reference is None:
            self.reference = current.copy()
        ref = self.reference
        fields = []
        for ring in (LEFT_EYE, RIGHT_EYE):
            # Only horizontal spacing: no source blink, gaze, or eyebrow expression.
            shift = np.clip(src[ring].mean(0)[0]-ref[ring].mean(0)[0], -.08, .08)
            field = np.zeros_like(current)
            field[:, 0] = shift
            fields.append(feature_field(current, ring, field))
        for ring, ends, vertical in (
            (NOSE, (98,327), False),
            (LIPS, (61,291), True),
        ):
            a, b = ends
            ratio = np.clip(np.linalg.norm(src[a]-src[b])/max(np.linalg.norm(ref[a]-ref[b]), .01), .75, 1.25)
            center = (current[a]+current[b])*.5
            field = (current-center)*(ratio-1)
            if not vertical:
                field[:, 1] = 0
            fields.append(feature_field(current, ring, field))
        weights = np.array([w for _,w in fields])
        # A feature's affine core dominates nearby falloffs without abrupt ring overrides.
        exclusive = weights**8
        delta = np.sum(np.array([f for f,_ in fields])*exclusive[...,None], axis=0)
        total = exclusive.sum(0)
        core = weights.max(0)
        delta *= (core/np.maximum(total, 1e-6))[:,None]
        # Skin displacement fades to zero at the fixed driving silhouette.
        distance = np.linalg.norm(current[:,None]-current[OVAL][None], axis=2).min(1)
        fade = np.clip(distance/.09, 0, 1)
        delta *= (fade*fade*(3-2*fade))[:,None]
        delta[OVAL] = 0
        length = np.linalg.norm(delta, axis=1)
        delta *= np.minimum(1., .08/np.maximum(length, 1e-6))[:, None]
        displacement = delta @ basis*scale
        # Evaluate the same safety envelope for all presets, so increasing the slider
        # cannot paradoxically decrease morphology because of different halving steps.
        _, limit, reductions = safe_geometry(target, displacement, triangles, .65)
        # Drop immediately when required for safety, recover gradually to avoid pumping.
        self.safe_limit = limit if self.safe_limit is None else min(limit, self.safe_limit*.9+limit*.1)
        limit = self.safe_limit
        applied = strength*limit/.65
        result = (target+displacement*applied).astype(np.float32)
        self.applied_strengths.append(applied)
        self.reduced_frames += applied < strength-1e-9
        self.reduction_steps += reductions
        return result
