"""
BaseJob utility tests: path conversions.
"""
from agents.manager.base_job import BaseJob


def test_work_dir_roundtrip():
    cfg = './configs/foo/bar/baz.py'
    work = BaseJob.get_work_dir(cfg)
    assert work == './logs/foo/bar/baz'
    cfg2 = BaseJob.get_config(work)
    assert cfg2 == cfg

