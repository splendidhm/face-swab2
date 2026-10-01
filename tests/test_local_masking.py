"""Discrete mask selection and perceptually monotonic full-result crossfades."""
import unittest
from pathlib import Path

import numpy as np

from src.local.faces import FaceAsset, composite
from src.local.identity import CompositeSettings
from src.local.masking import MASK_LEVELS, MASK_STRENGTHS, MaskStrengthControl


class MaskingTests(unittest.TestCase):
    def test_four_levels_and_default_preserve_existing_strength(self):
        control = MaskStrengthControl()
        self.assertEqual(control.snapshot(), ('very strong', 1.))
        for index, name in enumerate(MASK_LEVELS):
            control.set_level(index)
            self.assertEqual(control.snapshot(), (name, MASK_STRENGTHS[index]))
        for invalid in (-1,4,1.5,True,'low'):
            with self.assertRaises(ValueError):
                control.set_level(invalid)

    def test_levels_crossfade_finished_face_and_keep_background(self):
        frame = np.full((100,100,3), 80, np.uint8)
        face = np.full((50,50,3), 220, np.uint8)
        alpha = np.full((50,50),255,np.uint8)
        alpha[:5] = 0
        asset = FaceAsset(face,alpha,np.zeros((468,2),np.float32),np.zeros((0,3),np.int32),Path('unused.png'))
        settings = CompositeSettings(lighting=0,skin_color=0,detail=0)
        full = composite(frame,asset,target_box=(25,25,50,50),settings=settings)
        previous = 0
        for value in MASK_STRENGTHS:
            result = composite(frame,asset,target_box=(25,25,50,50),settings=settings,strength=value)
            expected = np.rint(frame.astype(float)*(1-value)+full.astype(float)*value)
            np.testing.assert_allclose(result,expected,atol=1)
            np.testing.assert_array_equal(result[:30],frame[:30])
            amount = np.abs(result.astype(float)-frame).sum()
            self.assertGreater(amount,previous)
            previous=amount
        np.testing.assert_array_equal(composite(frame,asset,target_box=(25,25,50,50),strength=0),frame)

    def test_invalid_opacity_is_rejected(self):
        for value in (-.1,1.1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                composite(np.zeros((10,10,3),np.uint8),None,strength=value)
