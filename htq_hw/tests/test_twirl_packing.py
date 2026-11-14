"""--twirl-per-job: several pre-twirl instances per job.

The 2026-11-11 ibm_chicago production pass sent K = 4 instances as four jobs
each: 508 jobs, 2,876 QPU s against 1,776 s untwirled.  A fit to its usage
gives ~3.4 s per job and ~0.1 s per pub, so K = 32 as one job per instance
would spend most of its time on job overhead.  Four draws also proved too
few: the slow-particle W00 asymmetry read 0.54 / 1.06 / 1.03 / 1.24 from the
single instances.  Packing keeps the job count of the untwirled run.
"""
import argparse
import json

import numpy as np
import pytest

from htq_hw import analyze as A
from htq_hw import campaign as CP
from htq_hw import target as T
from htq_hw.__main__ import main


def _insts(k=4):
    from htq_hw.tests.test_campaign import _small_pubs
    card, lat, center, be, emb, specs, pubs, info, bundles = _small_pubs(basis="rzz", ns=6, times=(0.5,),
                                                                          target="grid:4x5")
    names = [s.name for s in specs]
    return names, CP.pre_twirl_job(pubs, info, {n: 100 for n in names}, names, k, T.SEED, job_index=3)


def test_packing_keeps_the_draws_and_orders_pubs_instance_major():
    names, insts = _insts(4)
    one = CP.pack_twirl_instances(insts, 1)
    assert [p["instance_list"] for p in one] == [[0], [1], [2], [3]]
    assert [p["names"] for p in one] == [i["names"] for i in insts]
    two = CP.pack_twirl_instances(insts, 2)
    assert [p["instance_list"] for p in two] == [[0, 1], [2, 3]] and [p["instance"] for p in two] == [0, 2]
    assert two[0]["names"] == insts[0]["names"] + insts[1]["names"]
    for p in two:                      # the same circuits as one-per-job: packing never re-draws
        for i in p["instance_list"]:
            for n in insts[i]["names"]:
                assert p["pubs"][n] == insts[i]["pubs"][n] and p["shots"][n] == insts[i]["shots"][n]
    three = CP.pack_twirl_instances(insts, 3)
    assert [p["instance_list"] for p in three] == [[0, 1, 2], [3]]
    with pytest.raises(ValueError):
        CP.pack_twirl_instances(insts, 0)
    assert CP.twirl_instances_of({"twirl": {"instance": 2}}) == [2]
    assert CP.twirl_instances_of({"twirl": {"instance": 2, "instance_list": [2, 3]}}) == [2, 3]


@pytest.mark.parametrize("per_job", [2])
def test_packed_submission_round_trip(tmp_path, per_job):
    """Local testing mode on FakeNighthawk: K = 4 instances in 4 / per_job jobs
    per job; fetch merges them in instance order, records the per-instance row
    counts, and the bits read back like an unpacked group's."""
    from htq_hw.tests.test_campaign import _small_pubs
    card, lat, center, be, emb, specs, pubs, info, bundles = _small_pubs(basis="rzz", ns=6, times=(0.5,),
                                                                          target="fake:nighthawk")
    shots = {n: 64 for n in pubs}
    jobs = CP.group_jobs(specs, max_pubs=8)
    recs = CP.submit(be, pubs, info, shots, jobs, lat, emb, "rzz", str(tmp_path), tag="tw",
                     pre_twirl=4, twirl_per_job=per_job)
    assert len(recs) == (4 // per_job) * len(jobs)
    groups = {}
    for r in recs:
        meta = json.load(open(r["path"]))
        tw = meta["twirl"]
        assert meta["options"]["pre_twirl"] == 4 and meta["options"]["twirl_per_job"] == per_job
        il = tw["instance_list"]
        assert len(il) == per_job and tw["instance"] == il[0] and tw["per_job"] == per_job
        assert meta["pub_names"] == [CP.twirl_name(n, i) for i in il for n in tw["of"]]
        assert all(meta["shots"][n] == 16 for n in meta["pub_names"])
        groups.setdefault(tw["group"], []).append(meta)
    assert len(groups) == len(jobs)
    for metas in groups.values():
        assert sorted(i for m in metas for i in CP.twirl_instances_of(m)) == [0, 1, 2, 3]
    paths = CP.fetch_all([(r["job"], r["meta"]) for r in recs], str(tmp_path))
    assert len(paths) == len(jobs)
    for path, names in zip(paths, jobs):
        z = np.load(path, allow_pickle=True)
        assert list(z["pub_names"]) == names and len(z["twirl_jobs"]) == 4 // per_job
        assert int(z["twirl_instances"]) == 4
        counts = json.loads(str(z["twirl_counts"]))
        assert counts == {n: [16, 16, 16, 16] for n in names}
        for n in names:
            assert z[n].shape == (64, lat.n_wires)
        assert set(A.load_job_bits(path, lat).bits) == set(names)
    # one record finds its packed siblings on disk; a missing one is refused
    one = recs[0]
    metas = CP.twirl_group_metas(one["meta"], str(tmp_path))
    assert sorted(i for m in metas for i in CP.twirl_instances_of(m)) == [0, 1, 2, 3]
    if per_job < 4:
        with pytest.raises(ValueError, match="instances .* of 4"):
            CP.fetch(one["job"], one["meta"], str(tmp_path / "again"), meta_dir=str(tmp_path / "nowhere"))


def test_fetch_orders_instances_even_when_a_job_lists_them_out_of_order(tmp_path):
    """The merged rows must be instance 0, 1, ... whatever order the pubs came in."""
    class Bits:
        def __init__(self, a):
            self.a = a

        def to_bool_array(self, order="little"):
            return self.a.astype(bool)

    class Pub:
        def __init__(self, a):
            self.data = type("D", (), {"c": Bits(a)})()

    class Job:
        def __init__(self, arrs):
            self.arrs = arrs

        def result(self):
            return [Pub(a) for a in self.arrs]

    w = 3
    a = {i: np.full((2, w), i % 2, dtype=np.uint8) + np.eye(2, w, dtype=np.uint8) * 0 for i in range(4)}
    a = {i: np.c_[np.full((2, 1), i, dtype=np.uint8) % 2, np.full((2, w - 1), i // 2, dtype=np.uint8)]
         for i in range(4)}
    base = {"backend": "x", "n_logical": w, "pubs": {}}
    m0 = dict(base, job_id="j0", pub_names=["p#1", "p#0"],
              twirl={"instances": 4, "instance": 0, "instance_list": [0, 1], "group": "j0"})
    m1 = dict(base, job_id="j1", pub_names=["p#3", "p#2"],
              twirl={"instances": 4, "instance": 2, "instance_list": [2, 3], "group": "j0"})
    path = CP.fetch(Job([a[1], a[0]]), m0, str(tmp_path), group=[(Job([a[1], a[0]]), m0), (Job([a[3], a[2]]), m1)])
    z = np.load(path, allow_pickle=True)
    assert np.array_equal(z["p"], np.concatenate([a[0], a[1], a[2], a[3]]))


def test_cli_refuses_twirl_per_job_without_enough_instances(capsys):
    for argv in (["--twirl-per-job", "2", "audit"], ["--pre-twirl", "2", "--twirl-per-job", "4", "audit"],
                 ["--pre-twirl", "4", "--twirl-per-job", "0", "audit"]):
        with pytest.raises(SystemExit, match="twirl-per-job"):
            main(argv)
