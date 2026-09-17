# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the batched vector k-means used by palettization init."""

import pytest
import torch

from coreai_opt.palettization.kmeans._efficient_kmeans import _batched_kmeans


def _blobs(centers: torch.Tensor, copies: int) -> torch.Tensor:
    """Build ``(K*copies, D)`` points by repeating each center ``copies`` times."""
    return centers.repeat_interleave(copies, dim=0)


def _sort_by_first_col(centroids: torch.Tensor) -> torch.Tensor:
    """Order ``(K, D)`` centroids by their first coordinate for permutation-invariant compare."""
    return centroids[torch.argsort(centroids[:, 0])]


def test_batched_kmeans_recovers_separated_clusters_per_partition():
    """Each partition independently recovers its own well-separated clusters."""
    centers_p0 = torch.tensor([[0.0, 0.0], [10.0, 5.0], [20.0, -5.0]])
    centers_p1 = torch.tensor([[1.0, 1.0], [30.0, 0.0], [50.0, 2.0]])
    vectors = torch.stack([_blobs(centers_p0, 4), _blobs(centers_p1, 4)])  # (2, 12, 2)

    centroids, labels = _batched_kmeans(vectors, n_clusters=3, n_init=5, max_iter=50)

    assert centroids.shape == (2, 3, 2)
    assert labels.shape == (2, 12)
    assert torch.allclose(_sort_by_first_col(centroids[0]), centers_p0, atol=1e-4)
    assert torch.allclose(_sort_by_first_col(centroids[1]), centers_p1, atol=1e-4)


def test_batched_kmeans_labels_are_nearest_centroid():
    """Returned labels are the nearest-centroid assignment (a k-means invariant)."""
    vectors = torch.randn(3, 40, 2)
    centroids, labels = _batched_kmeans(vectors, n_clusters=4, n_init=2, max_iter=20)
    expected = torch.cdist(vectors, centroids).argmin(dim=-1)
    assert torch.equal(labels, expected)


def test_batched_kmeans_converges_to_fixed_point():
    """Each returned centroid is the mean of its assigned points (Lloyd's converged).

    Uniform data needs several Lloyd iterations to settle; if the loop stopped
    after one iteration the centroids would not equal the mean of their finally
    assigned points.
    """
    vectors = torch.arange(30, dtype=torch.float32).reshape(1, 30, 1)  # 0..29
    centroids, labels = _batched_kmeans(vectors, n_clusters=4, n_init=3, max_iter=100)
    for k in range(4):
        mask = labels[0] == k
        if mask.any():
            assert torch.allclose(centroids[0, k], vectors[0][mask].mean(dim=0), atol=1e-4)


def test_batched_kmeans_sample_weight_shifts_centroid():
    """A heavier point pulls the (single) centroid toward it (weighted mean)."""
    points = torch.tensor([[[0.0], [10.0]]])  # (1, 2, 1)
    weights = torch.tensor([[[1.0], [9.0]]])  # weight the second point 9x

    unweighted, _ = _batched_kmeans(points, n_clusters=1, n_init=1, max_iter=5)
    weighted, _ = _batched_kmeans(points, n_clusters=1, n_init=1, max_iter=5, sample_weight=weights)

    assert torch.allclose(unweighted[0, 0], torch.tensor([5.0]), atol=1e-4)  # plain mean
    assert torch.allclose(weighted[0, 0], torch.tensor([9.0]), atol=1e-4)  # weighted mean


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_batched_kmeans_preserves_shape_and_dtype(dtype):
    """Output centroids match the input dtype; labels are integer; shapes are (P,K,D)/(P,N)."""
    vectors = torch.randn(2, 20, 3).to(dtype)
    centroids, labels = _batched_kmeans(vectors, n_clusters=4, n_init=1, max_iter=5)
    assert centroids.shape == (2, 4, 3)
    assert centroids.dtype == dtype
    assert labels.shape == (2, 20)
    assert labels.dtype == torch.long


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batched_kmeans_runs_on_input_device():
    """Clustering runs on the input's device and returns results there."""
    vectors = torch.randn(2, 30, 2, device="cuda")
    centroids, labels = _batched_kmeans(vectors, n_clusters=3, n_init=2, max_iter=10)
    assert centroids.device.type == "cuda"
    assert labels.device.type == "cuda"
