"""Verify cumulative HDF string capacity and conservation with native PyTables."""
import pandas as pd
import pytest

from autobehaver.postinf.mmgen import hdf_store as hdf


def test_rc9_paired_growth_preserves_all_rows_widths_and_attributes(tmp_path):
    key = hdf.HDF_KEYS["video_lengths_tmp"]
    frames = [pd.DataFrame({"output_inventory": ["i" * inventory],
                           "output_fingerprints": ["f" * fingerprints], "n_frames": [160]}, index=[index])
              for index, (inventory, fingerprints) in enumerate(((256, 55504), (264, 55648),
                                                                 (520, 60000), (530, 120000)))]
    with hdf.open_store(str(tmp_path / "paired.h5"), "w") as store:
        hdf.hdf_safe_append(store, key, frames[0])
        widths = hdf._stored_string_itemsize_by_column(store, key)
        assert widths["output_inventory"] == 264
        assert widths["output_fingerprints"] == 55512
        store.get_storer(key).attrs.processing_identity = {"run": "rc9-integration", "frames": 160}
        for frame in frames[1:]:
            previous = widths
            hdf.hdf_safe_append(store, key, frame)
            widths = hdf._stored_string_itemsize_by_column(store, key)
            assert all(widths[column] >= width for column, width in previous.items())
            assert store.get_storer(key).attrs.processing_identity == {"run": "rc9-integration", "frames": 160}
        pd.testing.assert_frame_equal(store.select(key), pd.concat(frames))
        assert store.keys() == [key]


def test_schema_growth_retains_capacity_and_sizes_the_whole_incoming_batch(tmp_path):
    key = "/records"
    first = pd.DataFrame({"inventory": ["i" * 400], "fingerprints": ["f" * 1000], "value": [1.]}, index=[5])
    later = pd.DataFrame({"inventory": ["small", "tiny"], "fingerprints": ["x", "y"], "value": [2., 3.],
                          "note": ["short", "🦋" * 40]}, index=[6, 7])
    with hdf.open_store(str(tmp_path / "schema.h5"), "w") as store:
        hdf.hdf_safe_append(store, key, first, min_itemsize_map={"inventory": 600, "fingerprints": 3000})
        previous = hdf._stored_string_itemsize_by_column(store, key)
        store.get_storer(key).attrs.processing_identity = "unchanged"
        hdf.hdf_safe_append(store, key, later, min_itemsize_map={"note": 16})
        widths = hdf._stored_string_itemsize_by_column(store, key)
        assert all(widths[column] >= width for column, width in previous.items())
        assert widths["note"] >= len(later.iloc[1]["note"].encode("utf-8"))
        pd.testing.assert_frame_equal(store.select(key), pd.concat([first, later]))
        assert store.get_storer(key).attrs.processing_identity == "unchanged"


def test_unicode_widths_and_explicit_floors_use_encoded_bytes(tmp_path):
    key = "/unicode"
    frames = [pd.DataFrame({"label": ["é" * 4], "other": ["μ"]}),
              pd.DataFrame({"label": ["🦋" * 20], "other": ["μ" * 30]})]
    with hdf.open_store(str(tmp_path / "unicode.h5"), "w") as store:
        hdf.hdf_safe_append(store, key, frames[0], min_itemsize_map={"label": 8, "other": 32})
        hdf.hdf_safe_append(store, key, frames[1], min_itemsize_map={"label": 8, "other": 32})
        widths = hdf._stored_string_itemsize_by_column(store, key)
        assert widths["label"] >= 80 and widths["other"] >= 60
        assert widths["label"] < 200 and widths["other"] < 200
        pd.testing.assert_frame_equal(store.select(key), pd.concat(frames))
        hdf.hdf_safe_append(store, key, frames[0], min_itemsize_map={"label": 256, "other": 32})
        assert hdf._stored_string_itemsize_by_column(store, key)["label"] >= 256


def test_legacy_shared_string_fields_are_mapped_and_canonicalized(tmp_path):
    # Recreate the shared object-string block used by legacy pandas writers.
    # Pandas 3 otherwise infers separate StringDtype storage for each column.
    with pd.option_context('future.infer_string', False):
        key = "/legacy"
        first = pd.DataFrame({"left": ["a" * 80], "right": ["b"], "count": [1]})
        later = pd.DataFrame({"left": ["short"], "right": ["c"], "count": [2]})
        with hdf.open_store(str(tmp_path / "legacy.h5"), "w") as store:
            store.append(key, first, format="table", data_columns=["count"], index=False)
            assert hdf._stored_string_itemsize_by_column(store, key) == {"left": 80, "right": 80}
            store.get_storer(key).attrs.processing_identity = "legacy"
            hdf.hdf_safe_append(store, key, later, data_column_candidates=[], min_itemsize_map={})
            widths = hdf._stored_string_itemsize_by_column(store, key)
            assert widths["left"] >= 80 and widths["right"] >= 80
            assert {axis.cname for axis in store.get_storer(key).values_axes} >= {"left", "right"}
            pd.testing.assert_frame_equal(store.select(key), pd.concat([first, later]))
            pd.testing.assert_frame_equal(store.select(key, where="count == 1"), first)
            assert store.get_storer(key).attrs.processing_identity == "legacy"


def test_exception_recovery_keeps_a_prior_expansion(monkeypatch, tmp_path):
    key = "/retry"
    first = pd.DataFrame({"output_inventory": ["i" * 256], "output_fingerprints": ["f" * 55504]})
    later = pd.DataFrame({"output_inventory": ["i" * 264], "output_fingerprints": ["f" * 55648]})
    with hdf.open_store(str(tmp_path / "retry.h5"), "w") as store:
        hdf.hdf_safe_append(store, key, first, min_itemsize_map={"output_inventory": 264, "output_fingerprints": 111024})
        monkeypatch.setattr(hdf, "_required_string_width_updates", lambda *args: {})
        hdf.hdf_safe_append(store, key, later)
        widths = hdf._stored_string_itemsize_by_column(store, key)
        assert widths["output_fingerprints"] >= 111024
        assert widths["output_inventory"] >= 272
        pd.testing.assert_frame_equal(store.select(key), pd.concat([first, later]))


def test_failed_replacement_restores_original_table(monkeypatch, tmp_path):
    key = "/rollback"
    first = pd.DataFrame({"name": ["a"], "value": [1.]})
    with hdf.open_store(str(tmp_path / "rollback.h5"), "w") as store:
        hdf.hdf_safe_append(store, key, first)
        store.get_storer(key).attrs.processing_identity = "original"
        native_rename = store._handle.rename_node
        def fail_replacement(where, newname, *args, **kwargs):
            if str(where).endswith("__string_expand_tmp"):
                raise OSError("Injected replacement failure")
            return native_rename(where, newname, *args, **kwargs)
        monkeypatch.setattr(store._handle, "rename_node", fail_replacement)
        with pytest.raises(OSError, match="Injected replacement failure"):
            hdf.hdf_safe_append(store, key, pd.DataFrame({"name": ["z" * 100], "value": [2.]}))
        pd.testing.assert_frame_equal(store.select(key), first)
        assert store.get_storer(key).attrs.processing_identity == "original"
        assert store.keys() == [key]
