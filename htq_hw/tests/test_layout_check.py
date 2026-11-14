"""gauss-check: refuse a layout whose packet sites break Gauss' law at t = 0."""
import numpy as np
import pytest

from htq_hw import layout_check as LC
from htq_hw.__main__ import main
from htq_hw.campaign import pub_name
from htq_hw.model import Lattice

NS = 50


def _gauss_state_bits(lat, shots, rng, flip=None):
    """0/1 readout of states obeying G_n = +1, with matter bits of the sites in
    ``flip`` {site: probability} flipped at random (a bad qubit there)."""
    x = rng.choice([-1.0, 1.0], size=(shots, lat.ns))
    z = np.stack([(-1) ** n * x[:, (n - 1) % lat.ns] * x[:, n] for n in range(lat.ns)], axis=1)
    for n, p in (flip or {}).items():
        z[rng.random(shots) < p, n] *= -1
    s = np.ones((shots, lat.n_wires))
    s[:, [lat.site_qubit(n) for n in range(lat.ns)]] = z
    s[:, [lat.link_qubit(n) for n in range(lat.ns)]] = x
    return ((1 - s) / 2).astype(np.uint8)


def test_profile_is_one_on_a_gauss_state_and_flags_a_bad_site():
    lat, rng = Lattice(NS), np.random.default_rng(1)
    g, err = LC.gauss_profile(_gauss_state_bits(lat, 4000, rng), lat)
    assert np.allclose(g, 1.0) and np.allclose(err, 0.0)
    bits = _gauss_state_bits(lat, 4000, rng, flip={n: 0.2 for n in range(NS)} | {27: 0.45})
    g, err = LC.gauss_profile(bits, lat)
    res = LC.check(g, err, range(20, 29))
    assert res["far_median"] == pytest.approx(0.6, abs=0.05)
    assert res["status"] == "FAIL" and res["failing"] == [27]
    assert LC.check(g, err, range(20, 27))["status"] == "PASS"
    assert LC.check(g, err, range(20, 29), min_ratio=0.1)["status"] == "PASS"


def test_far_sites_wrap_around_the_ring():
    far = LC.far_sites(NS, 24)
    assert 16 in far and 32 in far and 17 not in far and 31 not in far and 0 in far and 49 in far
    assert LC.far_sites(NS, 2)[0] == 10 and 44 in LC.far_sites(NS, 2) and 45 not in LC.far_sites(NS, 2)


def test_cli_finds_the_reference_and_exits_nonzero_on_failure(tmp_path, capsys):
    lat, rng = Lattice(NS), np.random.default_rng(2)
    good = _gauss_state_bits(lat, 2000, rng, flip={n: 0.2 for n in range(NS)})
    bad = _gauss_state_bits(lat, 2000, rng, flip={n: 0.2 for n in range(NS)} | {22: 0.45})
    ref = "prod-bridge.prod_k1.26_s0.75_ns50:" + pub_name("j0", 0.0, False, "Z")
    other = "prod-bridge.prod_k1.26_s0.75_ns50:" + pub_name("j0", 0.0, False, "XYA")
    for tag, arr in (("good", good), ("bad", bad)):
        np.savez_compressed(tmp_path / f"{tag}.npz", job_id=tag, backend="x",
                            pub_names=np.array([other, ref]), **{other: arr, ref: arr})
    main(["gauss-check", str(tmp_path / "good.npz"), "--record", str(tmp_path / "good.json")])
    assert "PASS" in capsys.readouterr().out
    with pytest.raises(SystemExit) as ex:
        main(["gauss-check", str(tmp_path / "bad.npz")])
    assert ex.value.code == 1 and "sites [22]" in capsys.readouterr().out
    np.savez_compressed(tmp_path / "none.npz", job_id="n", backend="x", pub_names=np.array([other]),
                        **{other: good})
    with pytest.raises(SystemExit, match="no j0_t0.0_Z pub"):
        main(["gauss-check", str(tmp_path / "none.npz")])
