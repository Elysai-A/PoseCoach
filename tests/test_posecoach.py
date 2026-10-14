"""离线回归和有已知答案的数学测试：python -m unittest discover -s tests -v。"""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from analyze_reps_3d import ROOT, analyze_data, load_angle_csv, segment_reps
from lat_pulldown_3d_demo import calculate_angle_3d, process_video
from motion_metrics import assess_rep, normalize_motion_curve, validate_config


class BaselineTests(unittest.TestCase):
    def test_existing_15_reps_and_all_original_metrics(self):
        results, curves, motion = analyze_data(*load_angle_csv(ROOT / 'lat_pulldown_2d_3d.csv'))
        with (ROOT / 'tests/data/baseline_reps.csv').open() as file:
            expected = list(csv.DictReader(file))
        self.assertEqual(len(results), 15)
        self.assertEqual(len(curves), 15 * 101)
        for old, new in zip(expected, results):
            for field in old:
                self.assertAlmostEqual(float(old[field]), new[field], places=11, msg=field)
            self.assertEqual(new['data_status'], 'ok')
            self.assertTrue(0 <= new['quality_score'] <= 100)
            self.assertTrue(0 <= new['synchronization_score'] <= 100)
        self.assertEqual(motion['tracking_gap_count'], 0)
        np.testing.assert_allclose([r['progress_percent'] for r in curves[:101]], range(101), atol=1e-12)

    def test_original_count_script_still_counts_15(self):
        # 原计数逻辑保留，仅增加低幅度保护；无窗口模式实际运行。
        result = subprocess.run([sys.executable, str(ROOT / 'count_reps_3d.py')],
                                cwd=ROOT, capture_output=True, text=True,
                                env=self.headless_env(), timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('高位下拉次数: 15', result.stdout)

    @staticmethod
    def headless_env():
        import os
        return {**os.environ, 'MPLBACKEND': 'Agg', 'MPLCONFIGDIR': str(ROOT / 'work/mpl')}

    def test_cli_writes_reports_and_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, str(ROOT / 'analyze_reps_3d.py'),
                                     '--output-dir', folder], cwd=folder, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (Path(folder) / 'rep_analysis.csv').open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 15)
            self.assertIn('quality_score', rows[0])
            self.assertEqual(json.loads((Path(folder) / 'analysis_settings.json').read_text())['rep_count'], 15)


class CurveTests(unittest.TestCase):
    def setUp(self):
        self.times = np.linspace(0, 2, 61).tolist()
        self.left = (150 - 100 * (1 - np.abs(np.asarray(self.times) - 1))).tolist()

    def test_identical_motion_is_100(self):
        metrics, _ = assess_rep(self.times, self.left, self.left, 1)
        self.assertAlmostEqual(metrics['synchronization_score'], 100)
        self.assertAlmostEqual(metrics['quality_score'], 100)
        self.assertEqual(metrics['asymmetry_result'], 'symmetric')

    def test_time_normalization_is_invariant_to_rep_duration(self):
        grid1, raw1, shape1 = normalize_motion_curve(self.times, self.left)
        grid2, raw2, shape2 = normalize_motion_curve([10 + 2*t for t in self.times], self.left)
        np.testing.assert_allclose(grid1, grid2)
        np.testing.assert_allclose(raw1, raw2)
        np.testing.assert_allclose(shape1, shape2, atol=1e-12)

    def test_interpolation_uses_elapsed_time_not_row_number(self):
        grid, raw, _ = normalize_motion_curve([0, 0.1, 1], [0, 10, 100])
        self.assertAlmostEqual(raw[50], 50)

    def test_rom_difference_is_separate_from_shape_score(self):
        right = [30 + angle/2 for angle in self.left]
        metrics, _ = assess_rep(self.times, self.left, right, 1)
        self.assertAlmostEqual(metrics['synchronization_score'], 100)
        self.assertAlmostEqual(metrics['rom_balance_score'], 50)
        self.assertAlmostEqual(metrics['quality_score'], 85)
        self.assertEqual(metrics['asymmetry_reasons'], 'rom_difference')

    def test_delayed_bottom_is_detected_without_aligning_sides(self):
        right = np.interp(self.times, [0, 1.3, 2], [150, 50, 150]).tolist()
        metrics, _ = assess_rep(self.times, self.left, right, 1)
        self.assertLess(metrics['synchronization_score'], 100)
        self.assertLess(metrics['quality_score'], 100)
        self.assertAlmostEqual(metrics['bottom_time_difference_ratio'], 0.15)
        self.assertIn('bottom_timing', metrics['asymmetry_reasons'])

    def test_return_mismatch_is_reported_in_its_phase(self):
        right = np.interp(self.times, [0, 1, 1.8, 2], [150, 50, 60, 150]).tolist()
        metrics, _ = assess_rep(self.times, self.left, right, 1)
        self.assertAlmostEqual(metrics['pull_synchronization_score'], 100)
        self.assertLess(metrics['return_synchronization_score'], 80)
        self.assertIn('return_sync', metrics['asymmetry_reasons'])

    def test_flat_or_low_rom_is_not_perfect_sync(self):
        for right in ([80]*61, [80 + value/100 for value in self.left]):
            metrics, _ = assess_rep(self.times, self.left, right, 1)
            self.assertIsNone(metrics['quality_score'])
            self.assertIsNone(metrics['synchronization_score'])
            self.assertEqual(metrics['asymmetry_result'], 'insufficient_data')

    def test_gap_and_sparse_data_are_not_scored(self):
        metrics, _ = assess_rep([0, 0.03, 0.06, 0.5, 0.53], [150, 100, 50, 100, 150],
                                [150, 100, 50, 100, 150], .06)
        self.assertIn('tracking_gap', metrics['data_warning'])
        self.assertIsNone(metrics['quality_score'])
        sparse, _ = assess_rep([0, 1, 2], [150, 50, 150], [150, 50, 150], 1)
        self.assertIn('too_few_samples', sparse['data_warning'])

    def test_bad_config_is_rejected(self):
        for config in ({'sync_weight': 0.7}, {'curve_mae': float('nan')},
                       {'minimum_samples': 2}, {'unknown': 1}):
            with self.assertRaises(ValueError):
                validate_config(config)


class SegmentationAndInputTests(unittest.TestCase):
    def test_requires_top_bottom_top(self):
        times = [i*.2 for i in range(5)]
        self.assertEqual(segment_reps(times, [.9, .6, .1, .5, .9], 1), [(0, 4)])
        self.assertEqual(segment_reps(times, [.1, .1, .5, .9, .9], 1), [])
        self.assertEqual(segment_reps(times, [.9, .6, .1, .1, .1], 1), [])

    def test_gap_resets_rep_and_cooldown_suppresses_duplicates(self):
        self.assertEqual(segment_reps([0, .03, 1], [.9, .1, .9], .2), [])
        self.assertEqual(segment_reps([0, .1, .2, .3, .4, .5, .8],
                                     [.9, .1, .9, .1, .9, .1, .9], 1), [(0, 2), (4, 6)])

    def test_flat_input_is_zero_reps(self):
        results, curves, _ = analyze_data([0, .03, .06], [90]*3, [90]*3)
        self.assertEqual(results, [])
        self.assertEqual(curves, [])

    def test_small_tracking_jitter_is_not_counted(self):
        times = np.arange(0, 6, 1/30).tolist()
        angles = (90 + np.abs(np.asarray(times) % 2 - 1)).tolist()
        results, _, motion = analyze_data(times, angles, angles)
        self.assertEqual(results, [])
        self.assertTrue(motion['low_session_motion'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'lat_pulldown_2d_3d.csv'
            with path.open('w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['time', 'left_3d', 'right_3d'])
                writer.writerows(zip(times, angles, angles))
            result = subprocess.run([sys.executable, str(ROOT/'count_reps_3d.py')],
                                    cwd=folder, env=BaselineTests.headless_env(),
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('高位下拉次数: 0', result.stdout)

    def test_invalid_csv_fails_clearly(self):
        invalid = ['time,left_3d,right_3d\n', 'time,left_3d,right_3d\n0,,90\n',
                   'time,left_3d,right_3d\n0,nan,90\n1,90,90\n',
                   'time,left_3d,right_3d\n0,90,90\n0,90,90\n',
                   'time,left_3d,right_3d\n0,190,90\n1,90,90\n', 'other\n1\n']
        with tempfile.TemporaryDirectory() as folder:
            for text in invalid:
                path = Path(folder)/'invalid.csv'
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_angle_csv(path)

    def test_invalid_requested_rep_has_clear_cli_error(self):
        result = subprocess.run([sys.executable, str(ROOT/'compare_rep_shape.py'), '--rep', '99'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('requested 99', result.stderr)
        self.assertNotIn('Traceback', result.stderr)

    def test_missing_video_keeps_existing_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'preserved.csv'
            output.write_text('keep this data')
            result = subprocess.run([sys.executable, str(ROOT/'lat_pulldown_3d_demo.py'),
                                     '--video', str(Path(folder)/'missing.mp4'),
                                     '--output', str(output), '--headless'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn('视频不存在', result.stderr)
            self.assertEqual(output.read_text(), 'keep this data')

    def test_3d_vector_angles_and_zero_length(self):
        self.assertAlmostEqual(calculate_angle_3d((1,0,0), (0,0,0), (0,0,1)), 90)
        self.assertAlmostEqual(calculate_angle_3d((1,0,0), (0,0,0), (-1,0,0)), 180)
        self.assertAlmostEqual(calculate_angle_3d((1,0,0), (0,0,0), (1,0,0)), 0)
        self.assertIsNone(calculate_angle_3d((0,0,0), (0,0,0), (1,0,0)))


class VideoIOTests(unittest.TestCase):
    def test_real_video_io_with_simulated_pose_results(self):
        # 真正读写12帧视频；仅模拟MediaPipe关键点，不冒充真人识别测试。
        import cv2
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            video = folder/'frames.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 30, (64,64))
            self.assertTrue(writer.isOpened())
            for _ in range(12):
                writer.write(np.zeros((64,64,3), dtype=np.uint8))
            writer.release()
            model = folder/'placeholder.task'
            model.write_bytes(b'simulated for I/O test only')
            landmarks = [SimpleNamespace(x=.5,y=.5,z=0,visibility=1) for _ in range(33)]
            for shoulder, elbow, wrist in ((11,13,15),(12,14,16)):
                landmarks[shoulder] = SimpleNamespace(x=.3,y=.5,z=0,visibility=1)
                landmarks[wrist] = SimpleNamespace(x=.5,y=.3,z=0,visibility=1)
            visible = SimpleNamespace(pose_landmarks=[landmarks], pose_world_landmarks=[landmarks])
            absent = SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])
            fake = SimpleNamespace(timestamps=[], calls=0, closed=False)
            def detect(image, timestamp):
                fake.timestamps.append(timestamp)
                fake.calls += 1
                return absent if fake.calls == 4 else visible
            def close():
                fake.closed = True
            fake.detect_for_video = detect
            fake.close = close
            with patch('lat_pulldown_3d_demo.mp.tasks.vision.PoseLandmarker.create_from_options', return_value=fake), \
                 patch('cv2.imshow', side_effect=AssertionError('Headless called GUI')), \
                 patch('cv2.waitKey', side_effect=AssertionError('Headless called GUI')):
                counts = process_video(video, model, folder/'angles.csv', headless=True)
            self.assertEqual(counts, (12,11))
            self.assertTrue(fake.closed)
            self.assertTrue(all(b>a for a,b in zip(fake.timestamps,fake.timestamps[1:])))
            with (folder/'angles.csv').open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 11)
            self.assertAlmostEqual(float(rows[-1]['time']), .366)
            for row in rows:
                self.assertAlmostEqual(float(row['left_3d']), 90)
                self.assertAlmostEqual(float(row['right_3d']), 90)

    def test_exception_closes_landmarker_and_preserves_csv(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            video, model, output = folder/'invalid.avi', folder/'model.task', folder/'keep.csv'
            video.write_bytes(b'not a video')
            model.write_bytes(b'mock model')
            output.write_text('existing result')
            fake = SimpleNamespace(closed=False)
            def close():
                fake.closed = True
            fake.close = close
            with patch('lat_pulldown_3d_demo.mp.tasks.vision.PoseLandmarker.create_from_options', return_value=fake):
                with self.assertRaises(ValueError):
                    process_video(video, model, output, headless=True)
            self.assertTrue(fake.closed)
            self.assertEqual(output.read_text(), 'existing result')


if __name__ == '__main__':
    unittest.main()
