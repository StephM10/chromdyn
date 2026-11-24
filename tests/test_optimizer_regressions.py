import importlib

import numpy as np
import pytest

from chromdyn.optimization import EnergyLandscapeOptimizer


def test_nadam_matches_dozat_constant_momentum_reference(tmp_path):
    opt = EnergyLandscapeOptimizer(
        method="nadam",
        beta1=0.7,
        beta2=0.9,
        eta=0.01,
        epsilon=0,
        save_dir=str(tmp_path / "params"),
    )
    opt.phi_exp = np.zeros((1, 1))
    opt.init_optimization_params()
    # Dozat, Algorithm 8, mu_t=beta1. Golden values evaluated independently
    # with 50-digit Decimal arithmetic, including the t+1 bias correction.
    expected = [-0.01411764705882353, -0.026817512635088575, -0.02532649170736596]
    w = np.zeros((1, 1))
    for g, reference in zip([1.0, 2.0, -1.0], expected):
        w = opt.update_step(np.array([[g]]), w)
        np.testing.assert_allclose(w, [[reference]], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("readonly", [False, True])
def test_optimization_does_not_modify_callers_contact_matrix(tmp_path, readonly):
    p = tmp_path / "hic.txt"
    np.savetxt(p, [[1.0, 0.5], [0.5, 1.0]])
    opt = EnergyLandscapeOptimizer(method="gd", save_dir=str(tmp_path / "params"))
    opt.load_HiC(str(p))
    phi = np.array([[2.0, 3.0], [3.0, 4.0]])
    expected = phi.copy()
    if readonly:
        phi.flags.writeable = False
    opt.get_updated_params(np.zeros((2, 2)), phi)
    np.testing.assert_array_equal(phi, expected)


def test_importing_optimizer_does_not_construct_hic_manager(monkeypatch):
    import chromdyn.optimization as module
    from chromdyn import hic_utils

    def unexpected(*args, **kwargs):
        raise AssertionError("HiCManager constructed during import")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(hic_utils, "HiCManager", unexpected)
            importlib.reload(module)
    finally:
        importlib.reload(module)


def test_nadam_resumes_moments_with_explicit_next_iteration(tmp_path):
    opts = dict(method="nadam", beta1=0.7, beta2=0.9, eta=0.01, epsilon=0)
    opt = EnergyLandscapeOptimizer(**opts, save_dir=str(tmp_path / "first"))
    opt.phi_exp = np.zeros((1, 1))
    opt.init_optimization_params()
    w = opt.update_step(np.array([[1.0]]), np.zeros((1, 1)))
    restored = EnergyLandscapeOptimizer(
        **opts, it=2, save_dir=str(tmp_path / "restored")
    )
    restored.set_optimization_params(str(tmp_path / "first" / "nadam_params_1.h5"))
    actual = restored.update_step(np.array([[2.0]]), w)
    expected = opt.update_step(np.array([[2.0]]), w)
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_unknown_optimization_method_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported optimization method"):
        EnergyLandscapeOptimizer(method="bogus", save_dir=str(tmp_path / "bad"))

    opt = EnergyLandscapeOptimizer(method="gd", save_dir=str(tmp_path / "ok"))
    with pytest.raises(ValueError, match="Unsupported optimization method"):
        opt.set_opt_method("bogus")
