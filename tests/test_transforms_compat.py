import unittest

from transforms_compat import FallbackCompose, build_compose, render_torchvision_import_error


class TestTransformsCompat(unittest.TestCase):
    def test_fallback_compose_applies_transforms_in_order(self):
        compose = FallbackCompose(
            [
                lambda x: x + [1],
                lambda x: x + [2],
            ]
        )
        self.assertEqual(compose([]), [1, 2])

    def test_build_compose_returns_callable_pipeline(self):
        compose = build_compose(
            [
                lambda x: x + 3,
                lambda x: x * 2,
            ]
        )
        self.assertEqual(compose(4), 14)

    def test_render_torchvision_import_error_contains_windows_hint(self):
        msg = render_torchvision_import_error(
            ImportError("DLL load failed while importing _imaging: 找不到指定的模块。")
        )
        self.assertIn("DLL load failed while importing _imaging", msg)
        self.assertIn("reinstall Pillow and torchvision", msg)


if __name__ == "__main__":
    unittest.main()
