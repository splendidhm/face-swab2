"""Identity controls, geometric safety, and driving aperture registration."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.local.faces import Landmarker, MODEL, ROOT, OVAL, FaceAsset, composite, extract_face
from src.local.matting import SEGMENT_MODEL
from src.local.expression import LEFT_EYE, RIGHT_EYE, INNER_MOUTH, LIPS, aperture_mask, dense_warp
from src.local.identity import CompositeSettings, IdentityState, safe_geometry

FIXTURE = ROOT/'workspace/test-assets/astronaut.png'


class SettingsTests(unittest.TestCase):
    def test_invalid_settings_fail_before_render(self):
        for values in ({'identity': .66}, {'detail': -1}, {'lighting': float('nan')}, {'skin_color': float('inf')}):
            with self.assertRaises(ValueError):
                CompositeSettings(**values)

    def test_fold_and_collapse_reduce_strength(self):
        points = np.array([[0, 0], [10, 0], [0, 10]], np.float32)
        delta = np.array([[0, 0], [-30, 0], [0, 0]], np.float32)
        result, applied, reductions = safe_geometry(points, delta, np.array([[0, 1, 2]]), 1.)
        self.assertGreater(reductions, 0)
        self.assertLess(applied, 1)
        self.assertGreater(result[1, 0], 1.5)


@unittest.skipUnless(MODEL.exists() and SEGMENT_MODEL.exists() and FIXTURE.exists(), 'Models and face fixture required')
class IdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Landmarker()
        cls.temp = tempfile.TemporaryDirectory(dir=ROOT/'workspace')
        cls.asset = extract_face(FIXTURE, Path(cls.temp.name)/'face.png', cls.model)
        cls.target = cls.asset.points.copy()
        cls.source = cls.target.copy()
        center = (cls.source[61]+cls.source[291])*.5
        cls.source[LIPS] = center+(cls.source[LIPS]-center)*.85

    @classmethod
    def tearDownClass(cls):
        cls.model.close()
        cls.temp.cleanup()

    def test_legacy_setting_is_pixel_identical_to_omitted_setting(self):
        frame = self.asset.image
        a = composite(frame, self.asset, self.target)
        b = composite(frame, self.asset, self.target, settings=CompositeSettings.preset('legacy'))
        np.testing.assert_array_equal(a, b)

    def test_more_identity_moves_mouth_width_toward_source_and_pins_outline(self):
        results = [IdentityState().deform(self.source, self.target, self.asset.triangles, x) for x in (0, .35, .55)]
        widths = [np.linalg.norm(p[61]-p[291]) for p in results]
        self.assertGreater(widths[0], widths[1])
        self.assertGreater(widths[1], widths[2])
        for p in results:
            np.testing.assert_array_equal(p[OVAL], self.target[OVAL])

    def test_still_photo_smile_and_blink_do_not_enter_geometry(self):
        modified = self.source.copy()
        modified[159, 1] += 3
        modified[145, 1] -= 3
        modified[LIPS, 1] += 5
        modified[13, 1] += 2
        modified[14, 1] -= 2
        a = IdentityState().deform(self.source, self.target, self.asset.triangles, .35)
        b = IdentityState().deform(modified, self.target, self.asset.triangles, .35)
        np.testing.assert_allclose(a, b, atol=1e-5)

    def test_closed_eyes_and_mouth_follow_driver_and_remain_finite(self):
        state = IdentityState()
        state.deform(self.source, self.target, self.asset.triangles, .35)
        target = self.target.copy()
        for ring in (LEFT_EYE, RIGHT_EYE, INNER_MOUTH):
            center = target[ring].mean(0)
            target[ring, 1] = center[1]+(target[ring, 1]-center[1])*.02
        result = state.deform(self.source, target, self.asset.triangles, .35)
        for a, b, c, d in ((159,145,33,133), (386,374,362,263), (13,14,78,308)):
            ratio = lambda p: np.linalg.norm(p[a]-p[b])/np.linalg.norm(p[c]-p[d])
            self.assertAlmostEqual(ratio(result), ratio(target), places=4)
        self.assertTrue(np.isfinite(result).all())

    def test_driving_interiors_move_to_new_aperture_positions(self):
        source = self.asset.points.copy()
        source[LEFT_EYE, 0] -= 4
        source[RIGHT_EYE, 0] += 4
        asset = FaceAsset(self.asset.image, self.asset.alpha, source, self.asset.triangles, self.asset.path)
        frame = self.asset.image.copy()
        # Colored aperture is identifiable after transport; background is a different color.
        original_mask = aperture_mask(frame.shape, self.target)
        frame[original_mask > .5] = (13, 231, 77)
        settings = CompositeSettings.preset('strong')
        output = composite(frame, asset, self.target, settings=settings)
        # Same ROI coordinate origin as compositor.
        from src.local.faces import bbox, clip_box
        x,y,w,h = clip_box(bbox(self.target[OVAL]), frame.shape)
        local = self.target-[x,y]
        destination = IdentityState().deform(source, local, asset.triangles, settings.identity)
        driver = FaceAsset(frame[y:y+h,x:x+w], np.full((h,w),255,np.uint8), local, asset.triangles, asset.path)
        expected, coverage = dense_warp(driver, destination, (h,w))
        mask = (aperture_mask((h,w), destination) > .999) & (coverage > .5)
        self.assertGreater(mask.sum(), 0)
        np.testing.assert_allclose(output[y:y+h,x:x+w][mask], np.rint(expected[mask]), atol=1)

    def test_region_mode_ignores_identity_geometry(self):
        frame = self.asset.image
        a = composite(frame, self.asset, target_box=(0,0,frame.shape[1],frame.shape[0]),
                      settings=CompositeSettings(identity=0))
        b = composite(frame, self.asset, target_box=(0,0,frame.shape[1],frame.shape[0]),
                      settings=CompositeSettings(identity=.55))
        np.testing.assert_array_equal(a, b)

    def test_moved_apertures_cannot_paint_over_target_hair(self):
        class HairSegmenter:
            def segment(self, image):
                probabilities = np.zeros((*image.shape[:2], 6), np.float32)
                probabilities[..., 1] = 1
                return probabilities
        source = self.asset.points.copy()
        source[LEFT_EYE, 0] -= 4
        source[RIGHT_EYE, 0] += 4
        asset = FaceAsset(self.asset.image, self.asset.alpha, source, self.asset.triangles, self.asset.path)
        frame = self.asset.image.copy()
        output = composite(frame, asset, self.target, settings=CompositeSettings.preset('strong'), landmarker=HairSegmenter())
        np.testing.assert_array_equal(output, frame)
