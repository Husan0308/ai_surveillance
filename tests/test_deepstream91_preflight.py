import unittest
from scripts.preflight_deepstream91 import gpu_check, stack_matches


class PlatformGateTests(unittest.TestCase):
    def setUp(self):
        self.platform = {
            "gpu_name": "NVIDIA GeForce RTX 3060", "minimum_vram_mib": 12000,
            "minimum_driver": "595.58.03", "deepstream": "9.1.0",
            "cuda_runtime": "13.2", "tensorrt": "10.16.1.11",
        }

    def test_current_driver_blocked_with_correct_gpu(self):
        r = gpu_check("0, NVIDIA GeForce RTX 3060, 580.178.04, 12288, GPU-x", self.platform, 0)
        self.assertTrue(r["gpu_ok"])
        self.assertFalse(r["driver_ok"])

    def test_supported_driver(self):
        r = gpu_check("0, NVIDIA GeForce RTX 3060, 595.91.07, 12288, GPU-x", self.platform, 0)
        self.assertTrue(r["driver_ok"])

    def test_wrong_gpu_and_memory(self):
        for name, memory in [("NVIDIA GeForce RTX 3060 Ti", 12288), ("NVIDIA GeForce RTX 3060", 8192)]:
            r = gpu_check(f"0, {name}, 595.91.07, {memory}, GPU-x", self.platform, 0)
            self.assertFalse(r["gpu_ok"])

    def test_missing_gpu(self):
        with self.assertRaises(ValueError):
            gpu_check("1, NVIDIA GeForce RTX 3060, 595.91.07, 12288, GPU-x", self.platform, 0)

    def test_stack_mismatch(self):
        text = "DeepStreamSDK 9.1.0\nCUDA Runtime Version: 13.2\nTRT_PACKAGE=10.16.1.11-1+cuda13.2\n"
        self.assertTrue(stack_matches(text, self.platform))
        for before, after in [("9.1.0", "7.1.0"), ("13.2", "12.6"), ("10.16.1.11", "11.2.1.2")]:
            self.assertFalse(stack_matches(text.replace(before, after), self.platform))
        self.assertFalse(stack_matches("", self.platform))


if __name__ == "__main__":
    unittest.main()
