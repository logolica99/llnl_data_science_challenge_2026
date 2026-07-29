import io
import json
import sys
import threading
import unittest
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tifffile


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server


def synthetic_volume():
    volume = np.zeros((64, 48, 48), dtype=np.uint16)
    yy, xx = np.indices((48, 48))
    disk = (xx - 28) ** 2 + (yy - 24) ** 2 <= 3 ** 2
    for z in range(5, 59):
        volume[z][disk] = 1000
    return volume


class StandaloneViewerTests(unittest.TestCase):
    def tearDown(self):
        server.STATE.clear()

    def test_profile_tracks_shifted_cylinder(self):
        volume = synthetic_volume()
        result = server.extract_profile(
            volume,
            np.array([24.0, 24.0, 5.0]),
            np.array([24.0, 24.0, 58.0]),
            threshold=500.0,
        )
        self.assertGreater(result["coverage"], 0.85)
        self.assertAlmostEqual(result["median_radius_voxels"], 3.0, delta=0.65)
        tracked = [
            item for item in result["profile"]
            if item["center_u_voxels"] is not None
        ]
        self.assertTrue(tracked)
        self.assertGreater(
            float(np.median([
                abs(item["center_u_voxels"]) + abs(item["center_v_voxels"])
                for item in tracked
            ])),
            3.0,
        )
        self.assertIn("centerline_deviation_max_voxels", result)
        self.assertTrue(all(
            "deviation_voxels" in item for item in result["profile"]
        ))

    def test_catalog_merges_analysis_json_by_strut_id(self):
        state = server.AppState()
        try:
            state.set_registration({
                "junctions": [
                    {"id": 1, "position": [1, 2, 3]},
                    {"id": 2, "position": [4, 5, 6]},
                ],
                "struts": [
                    {"id": 10, "junction0": 1, "junction1": 2},
                ],
            })
            state.set_result_json({
                "defect_class": "thin",
                "measurement_provenance": {"ct_threshold": 40129},
                "findings": [
                    {
                        "strut_id": 10,
                        "classification": "thin",
                        "confidence": 0.92,
                        "measurement_profile": {
                            "source": "thin_thick_bent_pipeline",
                            "ct_threshold": 40129,
                            "section_measurements_sha256": "same-run",
                            "tracking_coverage": 0.5,
                            "median_radius_voxels": 2.5,
                            "centerline_deviation_max_voxels": 1.25,
                            "samples": [
                                {
                                    "sample_index": 0,
                                    "fraction": 0.25,
                                    "radius_voxels": 2.5,
                                    "center_x_voxels": 4.0,
                                    "center_y_voxels": 5.0,
                                    "center_z_voxels": 6.0,
                                    "center_u_voxels": 1.0,
                                    "center_v_voxels": 2.0,
                                    "sampling_plane_center_x_voxels": 3.8,
                                    "sampling_plane_center_y_voxels": 5.1,
                                    "sampling_plane_center_z_voxels": 6.0,
                                    "local_tangent_x": 0.2,
                                    "local_tangent_y": 0.3,
                                    "local_tangent_z": 0.9,
                                    "tracking_method": (
                                        "3d_centerline_local_tangent"
                                    ),
                                    "deviation_voxels": 1.25,
                                    "confidence": 0.9,
                                    "valid": True,
                                },
                            ],
                        },
                    },
                    {"strut_id": 99, "classification": "thin"},
                ],
            }, "findings_thin.json")
            state.set_result_json({
                "defect_class": "bent",
                "findings": [
                    {"strut_id": 10, "classification": "bent", "reason": "curved"},
                ],
            }, "findings_bent.json")
            catalog = state.catalog()
            self.assertEqual([item["strut_id"] for item in catalog["entries"]], [10])
            self.assertEqual(catalog["unmatched_ids"], [99])
            self.assertEqual(catalog["entries"][0]["classifications"], ["thin", "bent"])
            self.assertEqual(catalog["entries"][0]["fields"]["reason"], "curved")
            self.assertEqual(catalog["class_counts"]["thin"], 1)
            self.assertEqual(catalog["class_counts"]["bent"], 1)
            self.assertEqual(catalog["threshold"], 40129)
            self.assertTrue(
                catalog["entries"][0]["has_embedded_measurements"]
            )
            profile = state.profile(10, threshold=123)
            self.assertEqual(profile["profile_source"], "embedded_pipeline")
            self.assertEqual(profile["threshold"], 40129)
            self.assertEqual(profile["coverage"], 0.5)
            self.assertEqual(profile["profile"][0]["radius_voxels"], 2.5)
            self.assertEqual(
                profile["profile"][0][
                    "sampling_plane_center_x_voxels"
                ],
                3.8,
            )
            self.assertEqual(
                profile["profile"][0]["local_tangent_z"], 0.9
            )
            self.assertEqual(
                profile["profile"][0]["tracking_method"],
                "3d_centerline_local_tangent",
            )
        finally:
            state.clear()

    def test_crop_keeps_big_endian_uint16_as_compact_uint16(self):
        state = server.AppState()
        try:
            state.volume = synthetic_volume().astype(">u2")
            state.set_registration({
                "junctions": [
                    {"id": 1, "position": [24, 24, 5]},
                    {"id": 2, "position": [24, 24, 58]},
                ],
                "struts": [{"id": 10, "junction0": 1, "junction1": 2}],
            })
            crop = state.crop(10)
            self.assertEqual(crop["dtype"], "uint16")
            self.assertEqual(len(crop["body"]), int(np.prod(crop["shape"])) * 2)
        finally:
            state.clear()

    def test_http_upload_profile_volume_and_cleanup(self):
        httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"

        def request(path, method="GET", body=None):
            req = urllib.request.Request(
                base + path,
                data=body,
                method=method,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-File-Name": "findings_thin.json",
                },
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status, response.headers, response.read()

        try:
            tiff_buffer = io.BytesIO()
            tifffile.imwrite(tiff_buffer, synthetic_volume())
            status, _, payload = request(
                "/api/tiff", "POST", tiff_buffer.getvalue()
            )
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(payload)["shape_zyx"], [64, 48, 48])

            registration = json.dumps({
                "junctions": [
                    {"id": 1, "position": [24, 24, 5]},
                    {"id": 2, "position": [24, 24, 58]},
                ],
                "struts": [{"id": 10, "junction0": 1, "junction1": 2}],
            }).encode()
            request("/api/registration", "POST", registration)
            findings = json.dumps({
                "defect_class": "thin",
                "measurement_provenance": {"ct_threshold": 40129},
                "findings": [{
                    "strut_id": 10,
                    "classification": "thin",
                    "confidence": 0.9,
                    "measurement_profile": {
                        "source": "thin_thick_bent_pipeline",
                        "ct_threshold": 40129,
                        "section_measurements_sha256": "http-fixture",
                        "tracking_coverage": 0.5,
                        "median_radius_voxels": 2.75,
                        "centerline_deviation_max_voxels": 1.5,
                        "samples": [{
                            "sample_index": 0,
                            "fraction": 0.5,
                            "radius_voxels": 2.75,
                            "center_u_voxels": 4.0,
                            "center_v_voxels": 0.0,
                            "deviation_voxels": 1.5,
                            "confidence": 0.9,
                            "valid": True,
                        }],
                    },
                }],
            }).encode()
            request("/api/results", "POST", findings)

            _, _, catalog_body = request("/api/catalog")
            catalog = json.loads(catalog_body)
            self.assertTrue(catalog["ready"])
            self.assertEqual(catalog["entries"][0]["strut_id"], 10)

            threshold = catalog["threshold"]
            _, _, profile_body = request(
                f"/api/profile/10?threshold={threshold}"
            )
            profile = json.loads(profile_body)
            self.assertEqual(profile["profile_source"], "embedded_pipeline")
            self.assertEqual(profile["coverage"], 0.5)
            self.assertEqual(profile["threshold"], 40129)
            self.assertIn("centerline_deviation_max_voxels", profile)

            _, headers, crop_body = request("/api/volume/10")
            shape = [int(value) for value in headers["X-Volume-Shape"].split(",")]
            self.assertEqual(headers["X-Volume-Dtype"], "uint16")
            self.assertEqual(len(crop_body), int(np.prod(shape)) * 2)

            temp_dir = server.STATE.temp_dir
            self.assertTrue(Path(temp_dir).exists())
            request("/api/session", "DELETE")
            self.assertFalse(Path(temp_dir).exists())
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
