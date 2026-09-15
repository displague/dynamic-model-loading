"""Frozen output-only ridge specialists; no model weights or KV are modified."""
import numpy as np

DOMAINS = ('code', 'prose', 'math')
RANK = 16
RIDGE = 32.0
SEED = 20260916


def finite_matrix(value):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or not a.size or not np.isfinite(a).all():
        raise ValueError('Expected a finite nonempty matrix')
    return a


def projection(width):
    if type(width) is not int or width < RANK:
        raise ValueError('Insufficient hidden width')
    rng = np.random.default_rng(SEED)
    return (2 * rng.integers(0, 2, (width, RANK)) - 1).astype(np.float64) / np.sqrt(width)


def features(hidden):
    h = finite_matrix(hidden)
    h = h / np.sqrt(np.mean(h*h, axis=1, keepdims=True) + 1e-12)
    return np.concatenate((h @ projection(h.shape[1]), np.ones((len(h), 1))), axis=1)


def fit_head(x, target, draft):
    x, target, draft = map(finite_matrix, (x, target, draft))
    if target.shape != draft.shape or len(x) != len(target):
        raise ValueError('Training shapes differ')
    # Logit offsets have no effect on argmax. Remove each row's scalar offset.
    residual = target - draft
    residual -= residual.mean(axis=1, keepdims=True)
    return np.linalg.solve(x.T @ x + RIDGE * np.eye(x.shape[1]), x.T @ residual)


def fit_bank(hidden, target, draft, domain, fit):
    x = features(hidden)
    domain, fit = np.asarray(domain), np.asarray(fit)
    if (domain.shape != (len(x),) or fit.shape != domain.shape or fit.dtype != bool
            or not set(domain.tolist()) <= set(range(3)) or not fit.any() or fit.all()):
        raise ValueError('Invalid fit/domain partition')
    rows = [fit] + [fit & (domain == i) for i in range(3)]
    if any(not r.any() for r in rows):
        raise ValueError('Missing domain training rows')
    return np.stack([fit_head(x[r], target[r], draft[r]) for r in rows])


def evaluate(hidden, target, draft, domain, bank):
    x, t, d = features(hidden), finite_matrix(target), finite_matrix(draft)
    domain = np.asarray(domain)
    if (t.shape != d.shape or len(t) != len(x) or domain.shape != (len(x),)
            or bank.shape != (4, x.shape[1], t.shape[1]) or not np.isfinite(bank).all()
            or not set(domain.tolist()) <= set(range(3))):
        raise ValueError('Invalid evaluation shapes')
    ids = [t.argmax(1), d.argmax(1), (d + x @ bank[0]).argmax(1)]
    for shift in (0, 1):
        chosen = np.empty(len(x), dtype=np.int64)
        for i in range(3):
            rows = domain == i
            chosen[rows] = (d[rows] + x[rows] @ bank[1 + (i + shift) % 3]).argmax(1)
        ids.append(chosen)
    return np.stack(ids, axis=1)


def summarize(ids, domain, fit):
    ids, domain, fit = np.asarray(ids), np.asarray(domain), np.asarray(fit)
    if ids.ndim != 2 or ids.shape[1] != 5 or domain.shape != (len(ids),) or fit.shape != domain.shape:
        raise ValueError('Invalid score dimensions')
    reports = {}
    for split, mask in [('fit', fit), ('diagnostic', ~fit)]:
        count = int(mask.sum())
        if not count:
            raise ValueError('Empty score partition')
        correct = ids[:, 1:] == ids[:, :1]
        reports[split] = dict(positions=count, agreement={name:int(correct[mask, i].sum())
            for i, name in enumerate(('base', 'general', 'specialist', 'wrong_domain'))},
            by_domain={DOMAINS[k]:dict(positions=int((mask & (domain == k)).sum()),
                agreement={name:int(correct[mask & (domain == k), i].sum())
                    for i, name in enumerate(('base', 'general', 'specialist', 'wrong_domain'))})
                for k in range(3)})
    diag = reports['diagnostic']; a = diag['agreement']
    # Absolute counts were frozen for 48 diagnostic positions; do not retune.
    general = a['general'] >= a['base'] + 2
    specialist = (a['specialist'] >= max(a['base'], a['general'], a['wrong_domain']) + 3
        and all(row['agreement']['specialist'] >= row['agreement']['general']
                for row in diag['by_domain'].values()))
    return dict(scores=reports, gates=dict(Hgeneral=general, Hspecialization=specialist),
        decision='eligible_for_specialist_physical_protocol' if specialist else 'stop_this_output_specialist_candidate',
        generated_acceptance_measured=False, physical_paging_measured=False,
        capacity_measured=False, native_admission_evaluated=False)
