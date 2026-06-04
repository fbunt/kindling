"""Tests for the host-side dataset module (schema/sample reads only).

Query execution moved entirely into the sandbox worker; its behavior is covered
by tests/test_sandbox_container.py. This file now only exercises get_dataset_info.
"""

from app.query_engine import get_dataset_info


class TestGetDatasetInfo:
    def test_return_structure(self):
        info = get_dataset_info()
        assert "columns" in info
        assert "key_columns" in info
        assert "row_count_approx" in info
        assert "sample_rows" in info

    def test_columns_match_schema(self):
        info = get_dataset_info()
        assert isinstance(info["columns"], dict)
        assert "year" in info["columns"]
        assert "Event_ID" in info["columns"]
        assert "area_m2" in info["columns"]

    def test_key_columns_present(self):
        info = get_dataset_info()
        kc = info["key_columns"]
        assert "year" in kc
        assert "bs" in kc
        assert "nlcd" in kc
        assert "Incid_Type" in kc

    def test_sample_rows(self):
        info = get_dataset_info()
        assert isinstance(info["sample_rows"], list)
        assert len(info["sample_rows"]) == 5
        assert "year" in info["sample_rows"][0]

    def test_row_count(self):
        info = get_dataset_info()
        assert info["row_count_approx"] == "~745 million"
