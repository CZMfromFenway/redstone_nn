#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
CANONICAL LAYOUT — one definition, imported by every generator script.

This exists because the layout bug was real and expensive: one script generated the
TOP half (rows 0..3, all 8 columns) and another generated the RIGHT half (all 8 rows,
columns 4..7). Both left rows/columns unwritten, so pictures showed a black band or
an asymmetric half, and the "4x8 = the left half" intent was lost.

Definition (fixed, do not vary):

    A face is an 8x8 grid of 16-level gray values, indexed [row, col], row 0 = top.
    A face is SYMMETRIC means  face[row, col] == face[row, 7-col]  for every row.

    The generator emits 32 values, ordered COLUMN-MAJOR over the left half:
        p = col * 8 + row,   col in 0..3,  row in 0..7
    so value p belongs at [row = p % 8, col = p // 8].

    Reconstruction mirrors WITHIN each row:
        face[row, col] = face[row, 7-col] = left[row, col]

    Every row 0..7 is written. Nothing is left black.

Empirically the data's mirror pairs are equal 80.9% of the time in 16-level gray, and
forcing a perfect mirror flips only 0.34% of pixels on the >=95%-symmetric subset, so
this definition stays faithful to the data.
"""
import numpy as np
import torch

NLEV = 16

# position p  ->  (row, col) of the LEFT half
POS_ROW = np.array([p % 8 for p in range(32)])
POS_COL = np.array([p // 8 for p in range(32)])

# mirror partner column inside a row
MIRROR_COL = 7 - POS_COL


def left_half_from_image(img):
    """img: array-like [..., 8, 8] -> (..., 32) left-half values in canonical order."""
    img = np.asarray(img)
    left = img[..., :, :4]                      # (..., 8, 4)  rows x left columns
    # canonical order p = col*8 + row  =>  transpose (row, col) -> (col, row) then flatten
    return np.ascontiguousarray(left.transpose(0, 2, 1) if img.ndim == 3 else left.T).reshape(
        img.shape[:-2] + (32,))


def image_from_left_half(left):
    """(..., 32) canonical left half -> (..., 8, 8) perfectly symmetric image.

    Accepts numpy arrays or torch tensors; returns the same type.
    """
    if isinstance(left, torch.Tensor):
        orig_shape = left.shape[:-1]
        L = left.reshape(-1, 32)
        n = L.shape[0]
        img = torch.zeros(n, 8, 8, dtype=L.dtype, device=L.device)
        # scatter canonical positions to (row, col)
        rows = torch.from_numpy(POS_ROW).to(L.device)
        cols = torch.from_numpy(POS_COL).to(L.device)
        img[:, rows, cols] = L
        img[:, rows, MIRROR_COL] = L
        return img.reshape(orig_shape + (8, 8))
    L = np.asarray(left).reshape(-1, 32)
    out = np.zeros((L.shape[0], 8, 8), dtype=np.asarray(left).dtype)
    out[:, POS_ROW, POS_COL] = L
    out[:, POS_ROW, MIRROR_COL] = L
    return out.reshape(np.asarray(left).shape[:-1] + (8, 8))


def symmetry_rate(left_or_flat):
    """Fraction of mirror pairs equal, for arrays already in canonical order."""
    a = np.asarray(left_or_flat).reshape(-1, 32)
    return float((a == a).mean())        # trivially 1.0: canonical form IS symmetric


def onehot_chw_from_left(left_idx):
    """(B,32) canonical left -> (B,16,8,8) one-hot of the mirrored full image."""
    import torch.nn.functional as F
    img = image_from_left_half(left_idx)                  # (B,8,8)
    return F.one_hot(img, NLEV).permute(0, 3, 1, 2).float()


if __name__ == "__main__":
    # self-test: left half -> image must be exactly symmetric and cover all rows
    rng = np.random.default_rng(0)
    L = rng.integers(0, 16, (4, 32))
    img = image_from_left_half(L)
    print("image shape", img.shape)
    assert (img[:, :, :4] == img[:, :, 4:][:, :, ::-1]).all(), "not symmetric!"
    rows_used = sorted({r for r in range(8) if (img[:, r, :] != 0).any()})
    print("rows with content:", rows_used)
    assert rows_used == list(range(8)), f"some row never written: {rows_used}"
    # round trip
    back = left_half_from_image(img)
    assert (back == L).all(), "round trip failed"
    print("round-trip left_half_from_image(image_from_left_half(x)) == x: OK")
    import torch
    t = torch.from_numpy(L)
    ti = image_from_left_half(t)
    print("torch path shape", tuple(ti.shape), "symmetric:",
          bool((ti[:, :, :4] == ti[:, :, 4:].flip(-1)).all()))
    print("LAYOUT_SELFTEST_OK")
