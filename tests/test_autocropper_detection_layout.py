"""Keep chamber identities stable when Hough reports out-of-layout circles."""
import unittest

import numpy as np

from autobehaver.preinf.autocropper.detection import (
    calculate_theoretical_centers,
    match_circles_to_chambers,
)


class ChamberLayoutTests(unittest.TestCase):
    def setUp(self):
        self.centers = calculate_theoretical_centers(2016, 1520, 3, 2)
        # Reproduced detector output from the size-nutrition trial5/rpi19 movie.
        self.circles = np.array([
            [541, 170, 135], [221, 171, 133], [859, 173, 132],
            [220, 487, 134], [856, 491, 133], [538, 490, 132],
        ])
        points = np.asarray(list(self.centers.values()), dtype=float)
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        self.bound = 0.45 * float(distances.min())

    def match(self, circles):
        return match_circles_to_chambers(circles, self.centers, 2,
                                         max_center_distance_px=self.bound)

    def test_false_circle_between_chambers_does_not_change_assignments(self):
        expected = self.match(self.circles)
        self.assertEqual(expected, self.match(np.vstack([self.circles, [663, 584, 130]])))
        self.assertEqual(expected['chamber1'], (442, 342))
        self.assertEqual(expected['chamber6'], (1712, 982))

    def test_missing_real_chamber_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError, 'chamber-region detections'):
            self.match(self.circles[:-1])

    def test_duplicate_within_chamber_region_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError, 'chamber-region detections'):
            self.match(np.vstack([self.circles, self.circles[0] + [5, 5, 0]]))

    def test_nonfinite_or_nonpositive_detections_are_rejected(self):
        for radius in (0, np.nan):
            invalid = self.circles.astype(float)
            invalid[0, 2] = radius
            with self.assertRaisesRegex(ValueError, 'positive radii'):
                self.match(invalid)

    def test_ambiguous_assignment_remains_rejected(self):
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            match_circles_to_chambers(np.array([[0, 0, 1], [0, 0, 1]]),
                                     {'a': (0, 0), 'b': (0, 0)}, 1,
                                     max_center_distance_px=1)


if __name__ == '__main__':
    unittest.main()
