"""ExG + Hough/RANSAC crop-row guidance. No learned weights or network calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def vegetation_mask(image: np.ndarray, roi_top: float) -> np.ndarray:
    """Normalized excess green: 2g-r-b, with Otsu and a positive-green floor."""
    b, g, r = cv2.split(image.astype(np.float32))
    total = np.maximum(r + g + b, 1)
    exg = (2 * g - r - b) / total
    positive = np.uint8(np.clip(exg, 0, 1) * 255)
    y0 = int(image.shape[0] * roi_top)
    threshold, _ = cv2.threshold(positive[y0:], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = np.uint8((positive > max(15, threshold)) & (g > r * 1.04) & (g > b * 1.02)) * 255
    mask[:y0] = 0
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def fit_rows(mask: np.ndarray, roi_top: float) -> list[dict]:
    """Hough seeds, local RANSAC x=a*y+b, then least-squares on inliers."""
    h, w = mask.shape
    y0, y1 = int(h * roi_top), int(h * 0.95)
    edges = cv2.Canny(mask, 50, 150)
    edges[:y0 + 3] = 0
    edges[y1:] = 0
    segments = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=30,
                              minLineLength=max(25, int(h * 0.13)), maxLineGap=int(h * 0.06))
    if segments is None:
        return []
    ys, xs = np.nonzero(edges)
    # Bound work on large, leafy images; deterministic across invocations.
    rng = np.random.default_rng(7)
    if len(xs) > 35000:
        take = rng.choice(len(xs), 35000, replace=False)
        xs, ys = xs[take], ys[take]
    candidates = []
    band = max(3, w * 0.009)
    for x0, sy0, x1, sy1 in segments.reshape(-1, 4):
        if abs(sy1 - sy0) < h * 0.12:
            continue
        a = (x1 - x0) / (sy1 - sy0)
        b = x0 - a * sy0
        if abs(a) > 2:
            continue
        near = np.abs(xs - (a * ys + b)) < band * 2
        xx, yy = xs[near], ys[near]
        if len(xx) < 25:
            continue
        best = None
        for _ in range(35):
            i, j = rng.choice(len(xx), 2, replace=False)
            if abs(float(yy[i]) - yy[j]) < h * 0.10:
                continue
            ra = (float(xx[j]) - xx[i]) / (float(yy[j]) - yy[i])
            rb = xx[i] - ra * yy[i]
            inliers = np.abs(xx - (ra * yy + rb)) < band
            if best is None or inliers.sum() > best.sum():
                best = inliers
        if best is None or best.sum() < 25:
            continue
        a, b = np.polyfit(yy[best], xx[best], 1)
        coverage = len(np.unique(np.clip(((yy[best] - y0) / (y1 - y0) * 12).astype(int), 0, 11))) / 12
        if coverage < 0.58:
            continue
        bottom, top = a * y1 + b, a * y0 + b
        if not (-0.2 * w < bottom < 1.2 * w and -0.1 * w < top < 1.1 * w):
            continue
        score = coverage * min(1, best.sum() / (y1 - y0))
        candidates.append(dict(a=float(a), b=float(b), bottom=float(bottom), top=float(top), score=float(score)))
    rows = []
    for line in sorted(candidates, key=lambda x: x['score'], reverse=True):
        if all(abs(line['bottom'] - other['bottom']) > w * 0.035 or
               abs(line['top'] - other['top']) > w * 0.035 for other in rows):
            rows.append(line)
    return sorted(rows[:24], key=lambda x: x['bottom'])


def select_pair(rows: list[dict], h: int, w: int, roi_top: float):
    pairs = []
    for left in rows:
        for right in rows:
            width = right['bottom'] - left['bottom']
            far_width = right['top'] - left['top']
            if not (w * 0.08 < width < w * 0.85 and 0 < far_width < width * 0.92):
                continue
            if not left['bottom'] < w / 2 < right['bottom']:
                continue
            da = right['a'] - left['a']
            if da <= 0:
                continue
            vy = (left['b'] - right['b']) / da
            vx = left['a'] * vy + left['b']
            if not (-h * 0.5 < vy < h * roi_top and w * 0.05 < vx < w * 0.95):
                continue
            # Prefer nearby, supported boundaries around the current optical axis.
            center_error = abs((left['bottom'] + right['bottom']) / 2 - w / 2) / w
            score = left['score'] + right['score'] - 1.5 * center_error - 0.15 * width / w
            pairs.append((score, left, right))
    if not pairs:
        return None
    _, left, right = max(pairs, key=lambda x: x[0])
    return left, right


def detect(image: np.ndarray, roi_top: float = 0.40, deadband: float = 0.04):
    original_h, original_w = image.shape[:2]
    scale = min(1, 1000 / original_w)
    work = cv2.resize(image, (round(original_w * scale), round(original_h * scale)))
    h, w = work.shape[:2]
    mask = vegetation_mask(work, roi_top)
    rows = fit_rows(mask, roi_top)
    pair = select_pair(rows, h, w, roi_top)
    canvas = work.copy()
    green = np.zeros_like(canvas)
    green[:, :, 1] = mask
    canvas = cv2.addWeighted(canvas, 1, green, 0.16, 0)
    y0, y1 = int(h * roi_top), int(h * 0.95)
    cv2.line(canvas, (0, y0), (w - 1, y0), (160, 160, 160), 1)
    result = dict(status='unknown', hint='unknown', offset_px=None, offset_fraction=None,
                  candidate_rows=len(rows), lookahead_y_fraction=0.70)
    if pair:
        left, right = pair
        lookahead = int(h * 0.70)
        target = sum(row['a'] * lookahead + row['b'] for row in pair) / 2
        offset = target - w / 2
        hint = 'centered' if abs(offset / w) <= deadband else ('right' if offset > 0 else 'left')
        result.update(status='detected', hint=hint, offset_px=round(offset / scale, 1),
                      offset_fraction=round(offset / w, 4))
        overlay = canvas.copy()
        polygon = np.array([[left['top'], y0], [right['top'], y0],
                            [right['bottom'], y1], [left['bottom'], y1]], np.int32)
        cv2.fillPoly(overlay, [polygon], (70, 190, 80))
        canvas = cv2.addWeighted(canvas, 0.82, overlay, 0.18, 0)
        for row in pair:
            cv2.line(canvas, (round(row['top']), y0), (round(row['bottom']), y1), (0, 230, 255), 4, cv2.LINE_AA)
        cv2.line(canvas, (w // 2, y1), (w // 2, lookahead), (255, 255, 255), 1)
        cv2.arrowedLine(canvas, (w // 2, y1), (round(target), lookahead), (255, 80, 255), 5, cv2.LINE_AA, tipLength=0.15)
        cv2.circle(canvas, (round(target), lookahead), 6, (255, 255, 255), -1)
        label = f'{hint.upper()} | target offset {result["offset_px"]:+.1f} px ({offset / w:+.1%} width)'
    else:
        label = 'UNKNOWN | no supported perspective row pair'
    cv2.rectangle(canvas, (0, 0), (w, 77), (24, 31, 25), -1)
    font_scale = min(0.66, w / 1050)
    cv2.putText(canvas, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'ExG + Hough / RANSAC | yellow: selected rows | magenta: guidance',
                (15, 58), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.82, (195, 220, 200), 1, cv2.LINE_AA)
    return cv2.resize(canvas, (original_w, original_h)), result, mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('-o', '--output', type=Path, help='Default: <input>_annotated.jpg')
    parser.add_argument('--roi-top', type=float, default=0.40, help='Start of ground ROI, fraction of height (default 0.40)')
    parser.add_argument('--deadband', type=float, default=0.04, help='Centered tolerance as image-width fraction')
    parser.add_argument('--mask-output', type=Path, help='Optional diagnostic vegetation-mask PNG')
    args = parser.parse_args()
    if not 0.1 <= args.roi_top <= 0.75 or not 0 <= args.deadband <= 0.25:
        parser.error('roi-top must be 0.1..0.75; deadband must be 0..0.25')
    output = args.output or args.image.with_name(args.image.stem + '_annotated.jpg')
    if output.resolve() == args.image.resolve() or (args.mask_output and
            args.mask_output.resolve() in (args.image.resolve(), output.resolve())):
        parser.error('input, output, and mask paths must be distinct')
    try:
        image = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or min(image.shape[:2]) < 100:
            raise ValueError('Input must be a readable image at least 100x100 pixels')
        annotated, result, mask = detect(image, args.roi_top, args.deadband)
        for path, pixels in [(output, annotated)] + ([(args.mask_output, mask)] if args.mask_output else []):
            ok, encoded = cv2.imencode(path.suffix, pixels)
            if not ok:
                raise ValueError(f'Cannot encode {path}')
            encoded.tofile(path)
        print(json.dumps(dict(input=str(args.image), output=str(output), **result)))
        return 0 if result['status'] == 'detected' else 2
    except (OSError, ValueError, cv2.error) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
