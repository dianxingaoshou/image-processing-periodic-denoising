"""Meaningful data-integrity and degradation tests; no denoiser truth access."""
import json
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

from data_pipeline import KINDS, _extract_images, load_manifest, sample_seed, sha256, synthesize


class SynthesisTests(unittest.TestCase):
    def setUp(self):
        self.clean = np.full((128, 160), 0.5)

    def test_deterministic_and_does_not_touch_global_rng(self):
        np.random.seed(77)
        state = np.random.get_state()
        a, meta, masks = synthesize(self.clean, "local", .12, 123)
        b, meta2, masks2 = synthesize(self.clean, "local", .12, 123)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(meta, meta2)
        np.testing.assert_array_equal(state[1], np.random.get_state()[1])
        json.dumps(meta)
        self.assertTrue(np.all(a[masks["clean"]] == self.clean[masks["clean"]]))
        for key in masks:
            np.testing.assert_array_equal(masks[key], masks2[key])

    def test_masks_partition_and_compact_support(self):
        noisy, meta, masks = synthesize(self.clean, "local", .12, 123)
        count = masks["clean"].astype(int) + masks["contaminated"] + masks["transition"]
        np.testing.assert_array_equal(count, np.ones_like(count))
        self.assertGreater(meta["clean_fraction"], 0.35)
        self.assertGreater(meta["contaminated_fraction"], 0.10)
        np.testing.assert_allclose(masks["true_noise"], noisy-self.clean)

    def test_changing_frequencies_are_phase_gradients(self):
        for kind in KINDS:
            noisy, meta, masks = synthesize(self.clean, kind, .18, 99)
            self.assertEqual(noisy.shape, self.clean.shape)
            self.assertTrue(np.all(np.isfinite(noisy)))
            self.assertTrue(np.all((noisy >= 0) & (noisy <= 1)))
            self.assertEqual(meta["clipping_fraction"], 0)
            self.assertLess(np.max(np.hypot(masks["true_fy"], masks["true_fx"])), .3)
        _, _, chirp = synthesize(self.clean, "chirp", .12, 99)
        self.assertGreater(np.ptp(chirp["true_fx"]), .03)
        _, _, rotation = synthesize(self.clean, "orientation", .12, 99)
        self.assertGreater(np.ptp(rotation["true_fy"]), .06)
        self.assertGreater(np.ptp(rotation["true_fx"]), .06)

    def test_clipping_is_recorded_and_actual_noise_is_postclip(self):
        edge = np.zeros((64, 64))
        output, metadata, masks = synthesize(edge, "global", .18, 7)
        self.assertGreater(metadata["clipping_fraction"], .4)
        np.testing.assert_array_equal(masks["true_noise"], output)
        self.assertLess(metadata["postclip_noise_rms"], metadata["preclip_noise_rms"])

    def test_stable_seed_and_invalid_input(self):
        self.assertEqual(sample_seed("123", "chirp", .12), sample_seed("123", "chirp", .12))
        self.assertNotEqual(sample_seed("123", "chirp", .12), sample_seed("124", "chirp", .12))
        with self.assertRaises(ValueError):
            synthesize(self.clean, "unknown", .12, 4)
        with self.assertRaises(ValueError):
            synthesize(np.full((64, 64), np.nan), "global", .12, 4)

    def test_archive_rejects_traversal_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            archive = base / "malicious.tgz"
            with tarfile.open(archive, "w:gz") as package:
                member = tarfile.TarInfo("../escaped.jpg")
                member.size = 3
                package.addfile(member, io.BytesIO(b"bad"))
            with self.assertRaises(ValueError):
                _extract_images(archive, base / "safe")
            self.assertFalse((base / "escaped.jpg").exists())


@pytest.mark.requires_bsds(dataset='legacy')
class DatasetIntegrityTests(unittest.TestCase):
    def test_official_split_and_unique_images(self):
        manifest = load_manifest()
        rows = manifest["images"]
        self.assertEqual(len({row["image_id"] for row in rows}), len(rows))
        self.assertEqual(sum(row["split"] == "dev" for row in rows), 30)
        self.assertEqual(sum(row["split"] == "test" for row in rows), 60)
        for row in rows:
            self.assertIn(row["official_split"], {"train", "val"} if row["split"] == "dev" else {"test"})
            image = np.load(Path("data") / row["clean_npy"])
            self.assertEqual(image.shape, (256, 256))
            self.assertTrue(np.all((image >= 0) & (image <= 1)))
            self.assertEqual(sha256(Path("data") / row["clean_npy"]), row["clean_sha256"])


if __name__ == "__main__":
    unittest.main()
