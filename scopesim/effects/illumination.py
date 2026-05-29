# -*- coding: utf-8 -*-
"""Image-plane illumination effects."""

from collections.abc import Callable, Mapping
from typing import ClassVar

import numpy as np
from astropy import units as u
from astropy.modeling.functional_models import Gaussian2D

from . import Effect
from ..optics.image_plane import ImagePlane
from ..utils import figure_factory, from_currsys


__all__ = [
    "Illumination",
    "ImagePlaneBackground",
    "gaussian2d",
    "quadratic_vignetting",
]


def gaussian2d(
    shape: tuple[int, int],
    amp: float = 1.0,
    mu: tuple[float, float] = (0.0, 0.0),
    sigma: tuple[float, float] = (2000.0, 2000.0),
    theta: u.Quantity[u.deg] | float = 0.0 * u.deg,
) -> np.ndarray:
    """Return a 2D elliptical Gaussian illumination map."""
    nx, ny = reversed(shape)
    y, x = np.ogrid[:ny, :nx]
    x = x - nx / 2
    y = y - ny / 2

    model = Gaussian2D(
        amplitude=amp,
        x_mean=mu[0],
        y_mean=mu[1],
        x_stddev=sigma[0],
        y_stddev=sigma[1],
        theta=theta << u.deg,
    )
    return model(x, y)


def quadratic_vignetting(
    shape: tuple[int, int],
    falloff: float = 0.01,
    r_ref: float | None = None,
    mu: tuple[float, float] = (0.0, 0.0),
    stretch: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> np.ndarray:
    """Return a quadratic vignetting pattern."""
    nx, ny = reversed(shape)

    yy, xx = np.ogrid[:ny, :nx]
    dx = xx - (nx / 2 + mu[0])
    dy = yy - (ny / 2 + mu[1])

    sx = np.where(dx >= 0, stretch[0], stretch[1])
    sy = np.where(dy >= 0, stretch[2], stretch[3])

    r2 = (dx / sx) ** 2 + (dy / sy) ** 2
    r2_ref = r2.max() if r_ref is None else r_ref ** 2

    return np.clip(1.0 - falloff * r2 / r2_ref, 0.0, 1.0)


class Illumination(Effect):
    """Large-scale multiplicative illumination variation on the image plane."""

    z_order: ClassVar[tuple[int, ...]] = (750,)

    def __init__(
        self,
        model: Callable = gaussian2d,
        modelargs: Mapping | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.meta.setdefault("include", "!DET.include_illumination")
        self._model = model
        self._modelargs = modelargs or {}
        self._map = None
        self._map_shape = None

    def apply_to(self, obj, **kwargs):
        if not isinstance(obj, ImagePlane):
            return obj

        shape = obj.hdu.data.shape
        if self._map is None or shape != self._map_shape:
            self._map = self._make_map(shape)
            self._map_shape = shape

        obj.hdu.data *= self._map
        return obj

    def _make_map(self, shape):
        illumination_map = self._model(shape, **self._modelargs)
        return illumination_map.astype(np.float32)

    def plot(self):
        """Plot the cached illumination map."""
        if self._map is None:
            raise RuntimeError("No illumination map cached - run a simulation first.")

        fig, ax = figure_factory()
        im = ax.imshow(
            self._map, origin="lower", vmin=0.98, vmax=1.0, cmap="gray_r",
        )
        fig.colorbar(im, ax=ax, label="Relative illumination")
        ax.set_title("Illumination")
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
        return fig


class ImagePlaneBackground(Effect):
    """Add an already integrated diffuse background to the image plane.

    The value added by this effect is in ScopeSim image-plane units,
    ``ph s-1 pixel-1``. Use this for post-disperser diffuse backgrounds that
    should not be converted into a source cube and sent through the spectral
    trace list. The supplied value or model should already include downstream
    throughput and detector QE. For tapered QE maps, use an average or otherwise
    representative positional QE when deriving this non-dispersed background.
    """

    z_order: ClassVar[tuple[int, ...]] = (760,)

    def __init__(
        self,
        value: float = 0.0,
        model: Callable | None = None,
        modelargs: Mapping | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.meta.setdefault("include", "!DET.include_image_plane_background")
        self.meta.setdefault("value", value)
        self._model = model
        self._modelargs = modelargs or {}
        self._map = None
        self._map_shape = None

    def apply_to(self, obj, **kwargs):
        if not isinstance(obj, ImagePlane):
            return obj

        shape = obj.hdu.data.shape
        if self._map is None or shape != self._map_shape:
            self._map = self._make_map(shape)
            self._map_shape = shape

        obj.hdu.data = obj.hdu.data + self._map
        return obj

    def _make_map(self, shape):
        if self._model is None:
            value = from_currsys(self.meta["value"], self.cmds)
            background = np.full(shape, value, dtype=np.float32)
        else:
            background = self._model(shape, **self._modelargs)
        return np.asarray(background, dtype=np.float32)

    def plot(self):
        """Plot the cached background map."""
        if self._map is None:
            raise RuntimeError("No background map cached - run a simulation first.")

        fig, ax = figure_factory()
        im = ax.imshow(self._map, origin="lower")
        fig.colorbar(im, ax=ax, label="ph s-1 pixel-1")
        ax.set_title("Image-plane background")
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
        return fig
