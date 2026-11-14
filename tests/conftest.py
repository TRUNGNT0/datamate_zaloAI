"""Fixture dùng chung. Dữ liệu giả và backend giả nằm ở evaluation/fakes.py (xem lý do ở đó)."""
import pytest

from app.config import Config
from evaluation.fakes import write_split


@pytest.fixture
def split_dirs(tmp_path):
    write_split(tmp_path / "train")
    write_split(tmp_path / "test", with_answers=False)
    return str(tmp_path / "train"), str(tmp_path / "test"), str(tmp_path / "out")


@pytest.fixture
def cfg(split_dirs):
    train, test, out = split_dirs
    return Config(train_dir=train, test_dir=test, out_dir=out, backend="gemini", query_engine="pandas", max_biz=None,
                  n_samples=3, agree_min=2)
