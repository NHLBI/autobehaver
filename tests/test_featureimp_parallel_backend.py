import os

import numpy as np
import pandas as pd
import pytest

from autobehaver.postinf.featureimp import committee, mp_utils


class _DummyBrokenFuture:
    def result(self):
        raise mp_utils.BrokenProcessPool("simulated worker crash")


class _DummyExecutor:
    def __init__(self, *args, **kwargs):
        self.shutdown_calls = []

    def submit(self, fn, task):
        return _DummyBrokenFuture()

    def shutdown(self, wait=True, cancel_futures=False):
        self.shutdown_calls.append((wait, cancel_futures))


def test_choose_safe_start_method_prefers_spawn_over_fork(monkeypatch):
    monkeypatch.setattr(mp_utils.mp, "get_all_start_methods", lambda: ["fork", "spawn"])
    monkeypatch.setattr(mp_utils.mp, "get_start_method", lambda allow_none=True: "fork")

    assert mp_utils.choose_safe_start_method() == "spawn"


def test_cap_worker_thread_env_preserves_existing(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    for key in (
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        monkeypatch.delenv(key, raising=False)

    changed = mp_utils.cap_worker_thread_env()

    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert "OMP_NUM_THREADS" not in changed
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert "MKL_NUM_THREADS" in changed


def test_iter_parallel_results_serial_path_runs_initializer():
    state = {"offset": 0}

    def init_worker(offset):
        state["offset"] = offset

    def worker(value):
        return value + state["offset"]

    out = list(
        mp_utils.iter_parallel_results(
            worker,
            [1, 2, 3],
            num_workers=1,
            initializer=init_worker,
            initargs=(10,),
            desc="serial test",
            stage_name="serial stage",
        )
    )

    assert out == [11, 12, 13]


def test_iter_parallel_results_raises_clear_error_on_broken_pool(monkeypatch):
    from autobehaver.postinf.featureimp import bounded_pool
    monkeypatch.setattr(bounded_pool, "ProcessPoolExecutor", _DummyExecutor)
    monkeypatch.setattr(bounded_pool, "wait", lambda futures, **kwargs: (set(futures), set()))
    monkeypatch.setattr(mp_utils, "_main_module_is_importable", lambda: True)
    monkeypatch.setattr(mp_utils.mp, "get_context", lambda method: object())
    monkeypatch.setattr(mp_utils.mp, "get_all_start_methods", lambda: ["spawn"])
    monkeypatch.setattr(mp_utils.mp, "get_start_method", lambda allow_none=True: None)

    with pytest.raises(RuntimeError, match="worker process crashed"):
        list(
            mp_utils.iter_parallel_results(
                str,
                [1],
                num_workers=2,
                desc="parallel test",
                stage_name="FeatureImp test stage",
            )
        )


def test_train_subset_fold_worker_returns_structured_error_dict():
    X = pd.DataFrame({"f0": [0.0, 1.0], "f1": [1.0, 0.0]})
    y = np.array([0, 1], dtype=int)
    le = committee.FixedBinaryLabelEncoder("A", "B")
    committee._init_pool_worker(X, X, y, le)

    args = (
        1,
        0,
        0,
        ["missing_feature"],
        np.array([0]),
        np.array([1]),
        None,
        None,
        None,
        None,
        None,
        None,
        2,
        None,
        1.0,
        1,
        1.0,
        1,
        {},
        False,
        None,
        None,
        None,
    )

    result = committee._train_subset_fold_worker(args)

    assert isinstance(result, dict)
    assert result.get("__worker_error__") is True
    assert result.get("error_type") == "KeyError"
    assert "missing_feature" in result.get("traceback", "")


def test_iter_parallel_results_falls_back_to_serial_when_main_is_not_importable(monkeypatch):
    monkeypatch.setattr(mp_utils.mp, "get_all_start_methods", lambda: ["spawn"])
    monkeypatch.setattr(mp_utils.mp, "get_start_method", lambda allow_none=True: None)
    monkeypatch.setattr(mp_utils, "_main_module_is_importable", lambda: False)

    state = {"offset": 0}

    def init_worker(offset):
        state["offset"] = offset

    def worker(value):
        return value + state["offset"]

    out = list(
        mp_utils.iter_parallel_results(
            worker,
            [1, 2],
            num_workers=2,
            initializer=init_worker,
            initargs=(5,),
            desc="serial fallback",
            stage_name="fallback stage",
        )
    )

    assert out == [6, 7]
