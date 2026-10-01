"""Tests for hair mattes, dense deformation, expression apertures and temporal lag."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.local.expression import (BlendState, INNER_MOUTH, LEFT_EYE, RIGHT_EYE,
                                  anatomical_triangles, aperture_mask, dense_warp,
                                  multiband_blend, stabilize_landmarks)
from src.local.faces import MODEL, ROOT, Landmarker, OVAL, extract_face, composite
from src.local.matting import SEGMENT_MODEL, contour_mask, face_matte, inward_feather

FIXTURE = ROOT/'workspace/test-assets/astronaut.png'


class MatteTests(unittest.TestCase):
    def test_hair_is_removed_even_inside_geometric_face(self):
        image = np.full((120, 100, 3), (100, 150, 190), np.uint8)
        contour = np.array([[10, 15], [50, 5], [90, 15], [90, 90], [50, 115], [10, 90]], np.float32)
        probabilities = np.zeros((120, 100, 6), np.float32)
        probabilities[..., 3] = 1
        probabilities[:35, :, 3] = 0
        probabilities[:35, :, 1] = 1
        image[:35] = 20
        alpha = face_matte(image, contour, probabilities)
        self.assertEqual(alpha[:35].max(), 0)
        self.assertEqual(alpha[60, 50], 255)
        self.assertTrue(np.any((alpha > 0) & (alpha < 255)))

    def test_feather_does_not_expand_into_background(self):
        contour = np.array([[15, 15], [45, 10], [70, 20], [65, 70], [20, 65]])
        mask = contour_mask((90, 90), contour)
        matte = inward_feather(mask, 4)
        self.assertTrue(np.all(matte[mask == 0] == 0))
        self.assertLess(matte.sum(), mask.sum())

    def test_original_png_transparency_is_not_filled(self):
        image = np.full((60, 60, 3), 150, np.uint8)
        probabilities = np.zeros((60, 60, 6), np.float32)
        probabilities[..., 3] = 1
        original = np.full((60, 60), 255, np.uint8)
        original[25:35, 25:35] = 0
        alpha = face_matte(image, np.array([[5, 5], [55, 5], [55, 55], [5, 55]]), probabilities, original)
        self.assertEqual(alpha[25:35, 25:35].max(), 0)

    def test_blend_preserves_zero_alpha_pixels(self):
        source = np.full((65, 71, 3), 210, np.float32)
        target = np.full((65, 71, 3), 80, np.float32)
        alpha = np.zeros((65, 71), np.float32)
        alpha[15:50, 15:55] = 1
        result = multiband_blend(source, target, alpha)
        np.testing.assert_array_equal(result[alpha == 0], target[alpha == 0])
        self.assertGreater(result[30, 30].mean(), 150)

    def test_color_state_limits_frame_to_frame_change(self):
        state = BlendState()
        state.color_shift(np.zeros(3))
        np.testing.assert_allclose(state.color_shift(np.array([40, 12, 8])), [10, 3, 2])


@unittest.skipUnless(MODEL.exists() and SEGMENT_MODEL.exists() and FIXTURE.exists(), 'Both models and face fixture required')
class ExpressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Landmarker()
        cls.temp = tempfile.TemporaryDirectory(dir=ROOT/'workspace')
        cls.asset = extract_face(FIXTURE, Path(cls.temp.name)/'face.png', cls.model)
        cls.photo = cv2.cvtColor(np.array(Image.open(FIXTURE).convert('RGB')), cv2.COLOR_RGB2BGR)
        cls.points = cls.model.detect(cls.photo)[0]

    @classmethod
    def tearDownClass(cls):
        cls.model.close()
        cls.temp.cleanup()

    def test_fixed_topology_has_full_anatomical_mesh(self):
        triangles = anatomical_triangles()
        self.assertEqual(triangles.shape, (852, 3))
        self.assertEqual(len(np.unique(triangles)), 468)
        self.assertGreater(len(self.asset.triangles), 800)

    def test_blink_filter_tracks_expression_without_skin_jitter(self):
        predicted = self.points.copy()
        detected = predicted.copy()
        detected[LEFT_EYE, 1] += 2
        detected[205, 1] += .1
        filtered = stabilize_landmarks(detected, predicted)
        self.assertGreaterEqual(float(np.mean(filtered[LEFT_EYE, 1]-predicted[LEFT_EYE, 1])), 1.79)
        self.assertLess(float(filtered[205, 1]-predicted[205, 1]), .09)

    def test_open_mouth_and_eye_interiors_remain_driving_video(self):
        points = self.points.copy()
        mouth = points[INNER_MOUTH]
        middle = mouth[:, 1].mean()
        points[INNER_MOUTH, 1] = middle+(mouth[:, 1]-middle)*3
        mask = aperture_mask(self.photo.shape[:2], points)
        self.assertGreater(np.count_nonzero(mask > .999), 20)
        frame = self.photo.copy()
        frame[mask > .999] = (20, 55, 220)
        output = composite(frame, self.asset, points)
        np.testing.assert_array_equal(output[mask > .999], frame[mask > .999])

    def test_lip_and_cheek_deformation_changes_warped_texture(self):
        source = self.asset.points.copy()
        # Non-rigid changes, not a global translation: smile corners and cheek raise.
        expression = source.copy()
        expression[61] += [-4, -3]
        expression[291] += [4, -3]
        expression[205] += [-2, -3]
        expression[425] += [2, -3]
        neutral, neutral_alpha = dense_warp(self.asset, source, self.asset.image.shape)
        smile, smile_alpha = dense_warp(self.asset, expression, self.asset.image.shape)
        valid = (neutral_alpha > .8) & (smile_alpha > .8)
        self.assertGreater(np.abs(neutral[valid]-smile[valid]).mean(), .05)
        self.assertTrue(np.isfinite(smile).all())

    def test_near_closed_eyes_do_not_produce_nonfinite_pixels(self):
        points = self.asset.points.copy()
        for ring in (LEFT_EYE, RIGHT_EYE):
            center = points[ring].mean(0)
            points[ring, 1] = center[1]+(points[ring, 1]-center[1])*.02
        warped, alpha = dense_warp(self.asset, points, self.asset.image.shape)
        self.assertTrue(np.isfinite(warped).all())
        self.assertTrue(np.isfinite(alpha).all())
        self.assertTrue(((alpha >= 0) & (alpha <= 1)).all())


if __name__ == '__main__':
    unittest.main()
