import numpy as np
from snowmix.modis import quality_mask, subset_quality_mask


def test_excluded_bad_quality_band_does_not_disable_valid_subset():
    raw = np.full((4, 7), 5000, np.int16)
    qc = np.full(4, 1 << 30, np.uint32)
    qc[0] |= 8 << (2 + 4 * 4)
    raw[0, 4] = -28672
    state = np.array([0, 1 << 13, 1 << 10, 1 << 2], np.uint16)
    assert not quality_mask(raw, qc, state)[0]
    np.testing.assert_array_equal(subset_quality_mask(raw, qc, state), [True, False, False, False])
    np.testing.assert_array_equal(
        subset_quality_mask(raw, qc, state, allow_adjacent=True), [True, True, False, False]
    )
