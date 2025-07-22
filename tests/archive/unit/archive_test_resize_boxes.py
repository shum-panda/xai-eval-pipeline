import unittest

import numpy as np

from archive.resize_box_np_array import resize_boxes


class TestResizeBoxes(unittest.TestCase):
    """Test Suite für resize_boxes Funktion."""

    def test_resize_boxes_upscale(self) -> None:
        """Test Bounding Box Skalierung nach oben."""
        boxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        orig_size = (100, 100)
        new_size = (200, 200)

        resized = resize_boxes(boxes, orig_size, new_size)
        expected = np.array([[20, 40, 60, 80]], dtype=np.float32)

        np.testing.assert_array_almost_equal(resized, expected)

    def test_resize_boxes_downscale(self) -> None:
        """Test Bounding Box Skalierung nach unten."""
        boxes = np.array([[20, 40, 60, 80]], dtype=np.float32)
        orig_size = (200, 200)
        new_size = (100, 100)

        resized = resize_boxes(boxes, orig_size, new_size)
        expected = np.array([[10, 20, 30, 40]], dtype=np.float32)

        np.testing.assert_array_almost_equal(resized, expected)

    def test_resize_empty_boxes(self) -> None:
        """Test Skalierung leerer Bounding Box Liste."""
        boxes = np.array([]).reshape(0, 4)
        orig_size = (100, 100)
        new_size = (200, 200)

        resized = resize_boxes(boxes, orig_size, new_size)
        self.assertEqual(len(resized), 0)
        self.assertEqual(resized.ndim, 2)
        self.assertEqual(resized.shape[1], 4)

    def test_resize_boxes_clipping(self) -> None:
        """Test dass Boxen innerhalb der Bildgrenzen bleiben."""
        boxes = np.array([[90, 90, 110, 110]], dtype=np.float32)
        orig_size = (100, 100)
        new_size = (50, 50)

        resized = resize_boxes(boxes, orig_size, new_size)

        # Koordinaten sollten nicht größer als neue Bildgröße sein
        self.assertTrue(np.all(resized[:, [0, 2]] <= 50))  # x-Koordinaten
        self.assertTrue(np.all(resized[:, [1, 3]] <= 50))  # y-Koordinaten
        self.assertTrue(np.all(resized >= 0))  # Nicht negativ
