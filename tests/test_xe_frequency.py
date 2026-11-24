"""Read-only Xe primary-GT sysfs paths; no GPU, root or driver writes are required."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gputaskman import drm


class XeFrequencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.device = Path(self.tmp.name) / 'device'
        self.freq = self.device / 'tile0' / 'gt0' / 'freq0'
        self.freq.mkdir(parents=True)
        self.card = drm.DrmCard(key=1, pdev='0000:04:00.0', sysfs=str(self.device), driver='xe',
                                vendor_id=0x8086, device_id=0xe222, name='Intel',
                                vram_total=32 << 30, integrated=False, intel=True)

    def read(self):
        with patch.object(drm, '_intel_pmu_rates', return_value=(None, None)):
            return drm.read_device(self.card, 0)

    def write(self, name, value):
        file = self.freq / name
        file.parent.mkdir(exist_ok=True)
        file.write_text(str(value))

    def test_actual_zero_is_preserved_and_other_gts_do_not_replace_primary_clock(self):
        self.write('act_freq', 0)
        self.write('cur_freq', 400)
        self.write('rp0_freq', 2400)
        other = self.device / 'tile0' / 'gt1' / 'freq0'
        other.mkdir(parents=True)
        (other / 'act_freq').write_text('1500')
        metrics = self.read()
        self.assertEqual((metrics.sm_clock, metrics.gfx_clock, metrics.sm_clock_max), (0, 400, 2400))
        (self.freq / 'act_freq').unlink()
        self.assertEqual(self.read().sm_clock, 400)
        (self.freq / 'cur_freq').write_text('invalid')
        self.assertIsNone(self.read().sm_clock)
        self.assertIsNone(self.read().throttle)

    def test_aggregate_reasons_preferred_and_new_driver_labels_preserved(self):
        self.write('throttle/reasons', 'pl1 thermal mem_thermal')
        self.write('throttle/reason_pl2', 1)  # separate files must not override the snapshot
        self.assertEqual(self.read().throttle, ['PL1', 'thermal', 'mem_thermal'])
        self.write('throttle/reasons', 'none')
        self.assertEqual(self.read().throttle, [])
        self.write('throttle/reasons', 'none pl1')
        self.assertIsNone(self.read().throttle)

    def test_legacy_reason_files_distinguish_unknown_idle_and_active(self):
        for node, _ in drm._INTEL_THROTTLE:
            self.write('throttle/' + node.replace('throttle_reason_', 'reason_'), 0)
        self.assertEqual(self.read().throttle, [])
        self.write('throttle/reason_thermal', 'invalid')
        self.assertIsNone(self.read().throttle)
        self.write('throttle/reason_pl1', 1)
        self.assertEqual(self.read().throttle, ['PL1'])

    def test_xe_does_not_borrow_i915_paths_when_its_primary_gt_is_unavailable(self):
        i915 = self.device.parent / 'gt' / 'gt0'
        i915.mkdir(parents=True)
        (i915 / 'rps_act_freq_mhz').write_text('2200')
        (i915 / 'rps_cur_freq_mhz').write_text('2300')
        (i915 / 'throttle_reason_thermal').write_text('1')
        metrics = self.read()
        self.assertIsNone(metrics.sm_clock)
        self.assertIsNone(metrics.gfx_clock)
        self.assertIsNone(metrics.throttle)
