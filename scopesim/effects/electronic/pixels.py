# -*- coding: utf-8 -*-
"""Any effects regarding pixels, like binning and border effects."""

from typing import ClassVar

from numpy.typing import ArrayLike, NDArray

from ...utils import from_currsys, figure_factory
from . import ElectronicEffect, Detector, logger


class ReferencePixelBorder(ElectronicEffect):
    """Remove signal from reference pixels.

    Detectors often have a number of rows and columns around the edges masked.
    These pixels serve as reference pixels for various purposes. They do not
    get any signal, but have all the detector effects, such as dark current
    and readout noise.

    .. versionchanged:: 0.11.2

       Re-implemented the effect with a new YAML syntax, see #840 for details.

    Parameters
    ----------
    border : list(4)
       a list with the number of rows and columns to be masked. The sequence
       should be [bottom, left, top, right]
    """

    z_order: ClassVar[tuple[int, ...]] = (861,)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.meta["border_sequence"] = "bottom left top right"
        if "border" not in self.meta:
            self.meta["border"] = [0, 0, 0, 0]
        else:
            self.meta["border"] = from_currsys(self.meta["border"], self.cmds)
        if isinstance(self.meta["border"], dict):
            for val in self.meta["border"].values():
                if len(val) != 4:
                    raise ValueError("All entries for 'border' must have exactly four values.")
        else:
            if len(self.meta["border"]) != 4:
                raise ValueError(
                    "Parameter 'border' must have exactly four values.")

    def __call__(self, data: ArrayLike, border: dict[int, int]) -> NDArray:
        if border[0] > 0:
            data[:border[0], :] = 0
        if border[1] > 0:
            data[:, :border[1]] = 0
        if border[2] > 0:
            data[-border[2]:, :] = 0
        if border[3] > 0:
            data[:, -border[3]:] = 0
        return data

    def _get_border(self, det_id: int) -> dict[int, int]:
        if hasattr(self.meta["border"], "dic"):
            return self.meta["border"].dic[det_id]
        if isinstance(self.meta["border"], list):
            return self.meta["border"]
        raise TypeError(
            f"{self.__class__.__name__}.meta['border'] must be either "
            f"dict or list, but is {self.meta['border']}")

    def _apply_to_det(self, det: Detector) -> None:
        """Subclasses can override if more params needed in call."""
        logger.debug("Apply %s to %s", self.display_name, det)
        logger.info(f"Applying border {from_currsys(self.meta['border'])}")
        det.data = self(det.data, self._get_border(det.det_id))

    def plot(self, det, **kwargs):
        """Show the masked detector image."""
        det = self.apply_to(det)
        _, ax = figure_factory()
        ax.imshow(det.data, origin="bottom", **kwargs)

    def __str__(self) -> str:
        """Return str(self)."""
        msg = (
            f"{self.__class__.__name__}: \"{self.display_name}\"\n"
            f"    {from_currsys(self.meta['border'], self.cmds)}"
            f"   ({self.meta['border_sequence']})\n"
        )
        return msg


class BinnedImageBase(ElectronicEffect):
    """Base class for binning effects."""

    z_order: ClassVar[tuple[int, ...]] = (870,)

    def __call__(self, data: ArrayLike) -> NDArray:
        bx, by = self.bin_factors
        ny, nx = data.shape
        return data.reshape(ny//by, by, nx//bx, bx).sum(axis=(1, 3))

    def bin_header(self, native_header):
        """WCS of the summed pixel blocks, also used by trace coordinates."""
        bx, by = self.bin_factors
        if not isinstance(bx, int) or not isinstance(by, int) or min(bx, by) < 1:
            raise ValueError('Bin factors must be positive integers')
        if 'allowed_bins' in self.meta and [bx, by] not in self.meta['allowed_bins']:
            raise ValueError(f'Unsupported binning: {bx} x {by}')
        header = native_header.copy()
        for axis, factor in ((1, bx), (2, by)):
            if header[f'NAXIS{axis}'] % factor:
                raise ValueError('Binning must divide the full detector dimensions')
            header[f'NAXIS{axis}'] //= factor
        for suffix in [''] + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            if f'CRPIX1{suffix}' not in header:
                continue
            for axis, factor in ((1, bx), (2, by)):
                header[f'CRPIX{axis}{suffix}'] = (header[f'CRPIX{axis}{suffix}'] - .5)/factor + .5
            if f'CD1_1{suffix}' in header:
                for i in (1, 2):
                    for j, factor in ((1, bx), (2, by)):
                        header[f'CD{i}_{j}{suffix}'] *= factor
            else:
                header[f'CDELT1{suffix}'] *= bx
                header[f'CDELT2{suffix}'] *= by
                if f'PC1_2{suffix}' in header:
                    header[f'PC1_2{suffix}'] *= by/bx
                if f'PC2_1{suffix}' in header:
                    header[f'PC2_1{suffix}'] *= bx/by
        header['BINX'], header['BINY'] = bx, by
        return header

    def _apply_to_det(self, det):
        header = self.bin_header(det.header)
        det.data = self(det.data)
        det.header.update(header)
        det.meta.update(header)


class BinnedImage(BinnedImageBase):
    required_keys = {"bin_size"}

    @property
    def bin_factors(self):
        return self.bin_size, self.bin_size

    @property
    def bin_size(self) -> int:
        return from_currsys(self.meta["bin_size"], self.cmds)


class UnequalBinnedImage(BinnedImageBase):
    required_keys = {"binx","biny"}

    @property
    def bin_factors(self):
        return self.binx, self.biny

    @property
    def binx(self) -> int:
        return from_currsys(self.meta["binx"], self.cmds)

    @property
    def biny(self) -> int:
        return from_currsys(self.meta["biny"], self.cmds)


class OnChipBinning(UnequalBinnedImage):
    """Sum charge in FITS x/y pixel blocks before adding read noise.

    Unlike a software sum after readout, each output pixel receives one read
    noise draw. Configure this effect between charge integration and read noise.
    WCS reference pixels refer to the centers of the new pixel blocks.
    """

    z_order = (815,)
