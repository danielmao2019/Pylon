"""
BaseJob utility tests: path conversions.
"""
from agents.manager.default_job import DefaultJob


def test_work_dir_roundtrip():
    cfg = './configs/foo/bar/baz.py'
    work = DefaultJob.get_work_dir(cfg)
    assert work == './logs/foo/bar/baz'
    cfg2 = DefaultJob.get_config_filepath(work)
    assert cfg2 == cfg

