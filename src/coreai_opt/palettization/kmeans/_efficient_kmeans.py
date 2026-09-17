# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import torch as _torch


def _kmeanspp_init_batched(x: _torch.Tensor, n_clusters: int) -> _torch.Tensor:
    """Greedy (farthest-point) kmeans++ init, batched over the leading axis.

    ``x`` is ``(P, N, D)``; returns ``(P, K, D)`` initial centroids, chosen
    independently per partition: a random first centroid, then repeatedly the
    point farthest from the chosen set (argmax of the min distance).
    """
    P, N, _ = x.shape
    p_idx = _torch.arange(P, device=x.device)
    centroids = _torch.empty((P, n_clusters, x.shape[-1]), device=x.device, dtype=x.dtype)

    first = _torch.randint(0, N, (P,), device=x.device)
    centroids[:, 0] = x[p_idx, first]
    closest = _torch.cdist(x, centroids[:, :1]).squeeze(-1).square()  # (P, N)
    for k in range(1, n_clusters):
        far = closest.argmax(dim=-1)  # (P,)
        centroids[:, k] = x[p_idx, far]
        d = _torch.cdist(x, centroids[:, k : k + 1]).squeeze(-1).square()
        closest = _torch.minimum(closest, d)
    return centroids


def _update_centroids_batched(
    x: _torch.Tensor,
    labels: _torch.Tensor,
    n_clusters: int,
    weights: _torch.Tensor | None,
    prev: _torch.Tensor,
) -> _torch.Tensor:
    """Batched (weighted) mean of the points assigned to each cluster.

    ``x`` is ``(P, N, D)``, ``labels`` is ``(P, N)``, ``prev`` is ``(P, K, D)``.
    Empty clusters keep their previous centroid (no collapse to the origin).
    """
    P, N, D = x.shape
    wl = weights.squeeze(-1) if weights is not None else x.new_ones((P, N))  # (P, N)
    sums = x.new_zeros((P, n_clusters, D)).scatter_add_(
        1, labels.unsqueeze(-1).expand(-1, -1, D), x * wl.unsqueeze(-1)
    )
    counts = x.new_zeros((P, n_clusters)).scatter_add_(1, labels, wl)  # (P, K)
    empty = counts == 0
    centroids = sums / counts.masked_fill(empty, 1.0).unsqueeze(-1)
    return _torch.where(empty.unsqueeze(-1), prev, centroids)


def _batched_kmeans(
    vectors: _torch.Tensor,
    n_clusters: int,
    n_init: int = 5,
    max_iter: int = 300,
    tol: float = 1e-4,
    sample_weight: _torch.Tensor | None = None,
) -> tuple[_torch.Tensor, _torch.Tensor]:
    """Batched vector k-means over the leading partition axis.

    Runs kmeans++ init + Lloyd's independently for each of the ``P`` partitions,
    fully batched (no Python loop over ``P``), keeping the best of ``n_init``
    restarts per partition. Device-preserving (runs on ``vectors.device``).

    Args:
        vectors (torch.Tensor): Points to cluster, shape ``(P, N, D)``.
        n_clusters (int): Number of centroids ``K`` per partition.
        n_init (int): Number of random restarts; the lowest-inertia result is
            kept independently per partition.
        max_iter (int): Maximum Lloyd iterations per restart.
        tol (float): Relative inertia-improvement threshold for early stopping.
        sample_weight (torch.Tensor | None): Optional per-point weights,
            shape ``(P, N, 1)``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Centroids ``(P, K, D)`` (input dtype)
        and integer labels ``(P, N)``.
    """
    P, N, D = vectors.shape
    x = vectors.float()
    w = sample_weight.float() if sample_weight is not None else None

    def inertia_of(min_d: _torch.Tensor) -> _torch.Tensor:
        sq = min_d.square()
        return (sq * w.squeeze(-1)).sum(dim=-1) if w is not None else sq.sum(dim=-1)

    best_inertia = x.new_full((P,), float("inf"))
    best_centroids = x.new_zeros((P, n_clusters, D))
    best_labels = x.new_zeros((P, N), dtype=_torch.long)

    for _ in range(n_init):
        centroids = _kmeanspp_init_batched(x, n_clusters)
        prev_inertia: _torch.Tensor | None = None
        for _ in range(max_iter):
            min_d, labels = _torch.cdist(x, centroids).min(dim=-1)
            centroids = _update_centroids_batched(x, labels, n_clusters, w, centroids)
            inertia = inertia_of(min_d)
            if prev_inertia is not None and bool(
                _torch.all(prev_inertia - inertia <= tol * prev_inertia)
            ):
                break
            prev_inertia = inertia

        # Final assignment so labels are consistent with the returned centroids.
        min_d, labels = _torch.cdist(x, centroids).min(dim=-1)
        inertia = inertia_of(min_d)
        improved = inertia < best_inertia
        best_inertia = _torch.where(improved, inertia, best_inertia)
        best_centroids = _torch.where(improved.view(P, 1, 1), centroids, best_centroids)
        best_labels = _torch.where(improved.view(P, 1), labels, best_labels)

    return best_centroids.to(vectors.dtype), best_labels
