"""Tests for SelectorWheel."""

import numpy as np

from scopesim.effects import SelectorWheel
from scopesim.optics.image_plane import ImagePlane
from scopesim.tests.mocks.py_objects.imagehdu_objects import _image_hdu_square


def make_image_plane():
    image_plane = ImagePlane(_image_hdu_square().header)
    image_plane.hdu.data = np.ones((10, 10), dtype=float)
    return image_plane


def test_selector_wheel_can_override_child_z_order():
    wheel = SelectorWheel(
        selector_key="image_plane_id",
        z_order=[760],
        wheel=[
            {
                "selector_value": 0,
                "effect_class": "ImagePlaneBackground",
                "effect_kwargs": {"value": 1.0},
            },
        ],
    )

    assert wheel.z_order == (760,)


def test_selector_wheel_inherits_child_z_order_by_default():
    wheel = SelectorWheel(
        selector_key="image_plane_id",
        wheel=[
            {
                "selector_value": 0,
                "effect_class": "ImagePlaneBackground",
                "effect_kwargs": {"value": 1.0},
            },
        ],
    )

    assert wheel.z_order == (760,)


def test_selector_wheel_applies_image_plane_effect_by_id():
    image_plane = make_image_plane()
    wheel = SelectorWheel(
        selector_key="image_plane_id",
        wheel=[
            {
                "selector_value": 0,
                "effect_class": "ImagePlaneBackground",
                "effect_kwargs": {"value": 2.0},
            },
            {
                "selector_value": 1,
                "effect_class": "ImagePlaneBackground",
                "effect_kwargs": {"value": 10.0},
            },
        ],
    )

    wheel.apply_to(image_plane)

    assert np.all(image_plane.hdu.data == 3.0)
