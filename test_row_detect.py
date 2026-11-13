"""Small behavioral checks; run: python -m unittest -v."""
import unittest

import cv2
import numpy as np

from row_detect import detect


class GuidanceTests(unittest.TestCase):
    def test_steering_sign_and_centered_deadband(self):
        for shift, expected in [(-80, 'left'), (0, 'centered'), (80, 'right')]:
            with self.subTest(shift=shift):
                image = np.full((600, 800, 3), (65, 85, 110), np.uint8)
                for side in [-1, 1]:
                    cv2.line(image, (400 + shift + side * 45, 210),
                             (400 + shift + side * 260, 599), (30, 160, 30), 24)
                annotated, result, _ = detect(image)
                self.assertEqual(result['hint'], expected)
                self.assertEqual(annotated.shape, image.shape)
                if shift:
                    self.assertGreater(result['offset_px'] * shift, 0)

    def test_blank_and_non_green_have_no_guidance(self):
        for color in [(0, 0, 0), (70, 80, 120), (255, 255, 255)]:
            image = np.full((480, 640, 3), color, np.uint8)
            _, result, _ = detect(image)
            self.assertEqual(result['status'], 'unknown')
            self.assertIsNone(result['offset_px'])

    def test_single_row_is_insufficient(self):
        image = np.full((600, 800, 3), (65, 85, 110), np.uint8)
        cv2.line(image, (350, 210), (100, 599), (30, 160, 30), 24)
        self.assertEqual(detect(image)[1]['status'], 'unknown')


if __name__ == '__main__':
    unittest.main()
