from pathlib import Path

from libxmp import XMPFiles, XMPMeta


def load_xmp_packet(file_path: Path) -> XMPMeta:
    # Input validations
    assert isinstance(file_path, Path), f"file_path must be Path, got {type(file_path)}"

    xmp_file = XMPFiles(file_path=str(file_path))
    xmp = xmp_file.get_xmp()
    xmp_file.close_file()
    assert xmp is not None, f"XMP packet missing in {file_path.name}"
    return xmp
