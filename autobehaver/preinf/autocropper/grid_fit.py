"""Score circle boundaries and fit the relative six-chamber layout to a movie."""
import itertools
import cv2
import numpy as np


def circle_support(gray, circles):
    """Angular edge coverage in [0,1], a ranking score, not a probability."""
    edges = cv2.Canny(gray, 25, 50)
    near = cv2.dilate(edges, np.ones((5, 5), np.uint8)) > 0
    angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    scores = []
    for x, y, radius in circles:
        xx = np.rint(x + radius * np.cos(angles)).astype(int)
        yy = np.rint(y + radius * np.sin(angles)).astype(int)
        valid = (xx >= 0) & (xx < gray.shape[1]) & (yy >= 0) & (yy < gray.shape[0])
        hit = np.zeros(len(angles), dtype=bool)
        hit[valid] = near[yy[valid], xx[valid]]
        scores.append(float(hit.mean()))
    return np.asarray(scores), edges


def _similarity(source, target):
    # x' = a*x - b*y + tx; y' = b*x + a*y + ty. No reflection.
    a = np.zeros((2 * len(source), 4))
    a[0::2, 0] = source[:, 0]; a[0::2, 1] = -source[:, 1]; a[0::2, 2] = 1
    a[1::2, 0] = source[:, 1]; a[1::2, 1] = source[:, 0]; a[1::2, 3] = 1
    return np.linalg.lstsq(a, target.ravel(), rcond=None)[0]


def _transform(points, params):
    a, b, tx, ty = params
    return points @ np.array([[a, b], [-b, a]]) + [tx, ty]


def fit_grid(circles, template, down_factor, frame_shape, scores=None):
    """Select a unique supported grid; infer missing centers from >=3 noncollinear matches.

    Translation is free. Scale 0.8..1.2 and rotation +/-20 degrees preserve the
    supported camera geometry and chamber numbering. All six chamber discs must
    fit in the image. These are geometry bounds, not learned confidence limits.
    """
    raw = np.asarray(circles, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) < 3 or not np.isfinite(raw).all() or np.any(raw[:, 2] <= 0):
        raise ValueError('Grid fitting needs at least three finite circle candidates with positive radii.')
    if down_factor <= 0:
        raise ValueError('down_factor must be positive.')
    ids = list(template)
    ref = np.asarray(list(template.values()), dtype=float)
    if ref.shape != (6, 2):
        raise ValueError('Grid fitting requires the supported 3x2 layout.')
    points = raw[:, :2] * down_factor
    scores = np.ones(len(raw)) if scores is None else np.asarray(scores, dtype=float)
    if scores.shape != (len(raw),) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Invalid circle support scores.')
    # Very weak image candidates cannot establish an inferred chamber layout.
    usable = np.flatnonzero(scores >= .20)
    if len(usable) < 3:
        raise ValueError('Fewer than three candidates have adequate boundary support.')
    if len(usable) > 40:
        raise ValueError('Too many supported circle candidates; inspect the diagnostic image.')
    spacing = np.min(np.linalg.norm(ref[:, None] - ref[None, :], axis=2) + np.eye(6) * 1e9)
    tolerance = .10 * spacing
    radius = float(np.median(raw[usable, 2]) * down_factor)
    height, width = frame_shape[:2]
    hypotheses = {}
    for i, j in itertools.combinations(range(6), 2):
        delta = ref[j] - ref[i]
        length = np.dot(delta, delta)
        for u, v in itertools.permutations(usable, 2):
            target = points[v] - points[u]
            a = np.dot(delta, target) / length
            b = (delta[0] * target[1] - delta[1] * target[0]) / length
            scale, angle = np.hypot(a, b), np.degrees(np.arctan2(b, a))
            if not (.8 <= scale <= 1.2 and abs(angle) <= 20):
                continue
            params = np.r_[a, b, points[u] - _transform(ref[i:i+1], [a, b, 0, 0])[0]]
            for _ in range(3):
                fitted = _transform(ref, params)
                distances = np.linalg.norm(fitted[:, None] - points[usable][None, :], axis=2)
                nearest = distances.argmin(axis=1)
                matched = np.flatnonzero(distances[np.arange(6), nearest] <= tolerance)
                picked = usable[nearest[matched]]
                if len(matched) < 3 or len(set(picked)) != len(picked) or len(set(matched // 3)) < 2 or len(set(matched % 3)) < 2 or np.linalg.matrix_rank(ref[matched] - ref[matched].mean(axis=0)) < 2:
                    break
                params = _similarity(ref[matched], points[picked])
            else:
                fitted = _transform(ref, params)
                if not (.8 <= np.hypot(*params[:2]) <= 1.2 and abs(np.degrees(np.arctan2(params[1], params[0]))) <= 20):
                    continue
                if np.any(fitted[:, 0] < radius) or np.any(fitted[:, 0] > width-radius) or np.any(fitted[:, 1] < radius) or np.any(fitted[:, 1] > height-radius):
                    continue
                residual = np.linalg.norm(fitted[matched] - points[picked], axis=1)
                if np.max(residual) > tolerance:
                    continue
                key = tuple(zip(matched.tolist(), picked.tolist()))
                quality = len(matched) + .3 * float(scores[picked].mean()) - .3 * float(np.mean(residual) / tolerance)
                hypotheses[key] = (quality, fitted, matched, picked, residual)
    if not hypotheses:
        raise ValueError('No supported full-frame 3x2 grid; need three noncollinear chambers.')
    ranked = sorted(hypotheses.values(), key=lambda item: item[0], reverse=True)
    best = ranked[0]
    # Competing placements or candidate identities must be clearly worse.
    for other in ranked[1:]:
        if len(other[2]) == len(best[2]) and best[0] - other[0] < .05:
            if np.max(np.linalg.norm(other[1] - best[1], axis=1)) > 2 or set(other[3]) != set(best[3]):
                raise ValueError('Ambiguous grid fit; inspect the candidates rather than guessing chamber identities.')
    _, fitted, matched, picked, residual = best
    coords = fitted.copy()
    coords[matched] = points[picked]  # Preserve measured centers; infer only missing ones.
    report = {'method': 'scored_similarity_grid_v1', 'candidate_scores': scores.tolist(),
              'candidates_full_resolution': (raw * down_factor).tolist(),
              'fitted_centers': dict(zip(ids, fitted.tolist())),
              'selected_candidates': {ids[k]: int(v) for k, v in zip(matched, picked)},
              'inferred_chambers': [ids[k] for k in range(6) if k not in matched],
              'max_residual_px': float(max(residual)), 'score_kind': 'angular edge coverage; not a probability'}
    return {key: tuple(np.rint(value).astype(int)) for key, value in zip(ids, coords)}, report
