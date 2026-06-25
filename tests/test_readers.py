import numpy as np
from pathlib import Path
from tpcve.cloud.generate_cloud import _read_pcd_np, _read_ply_np

def test_read_pcd_ascii_crlf():
    """PCD with \\r\\n line endings should parse correctly."""
    content = (
        "VERSION .7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        "WIDTH 3\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        "DATA ascii\n"
        "1.0 2.0 3.0\r\n"
        "4.0 5.0 6.0\r\n"
        "7.0 8.0 9.0\r\n"
    ).encode('ascii')
    tmp = Path("/tmp/test_crlf.pcd")
    tmp.write_bytes(content)
    try:
        pts = _read_pcd_np(str(tmp))
        assert pts.shape == (3, 3), f"Expected (3,3), got {pts.shape}"
        np.testing.assert_array_almost_equal(pts, [[1,2,3],[4,5,6],[7,8,9]])
    finally:
        tmp.unlink()


def test_read_ply_binary_raises_friendly():
    """Binary PLY should raise ValueError, not UnicodeDecodeError."""
    content = (
        b"ply\n"
        b"format binary_little_endian 1.0\n"
        b"element vertex 2\n"
        b"property float x\n"
        b"property float y\n"
        b"property float z\n"
        b"end_header\n"
        + np.array([[1,2,3],[4,5,6]], dtype=np.float32).tobytes()
    )
    tmp = "/tmp/test_binary.ply"
    with open(tmp, "wb") as f:
        f.write(content)
    try:
        _read_ply_np(tmp)
        assert False, "Expected ValueError"
    except UnicodeDecodeError:
        assert False, "Got UnicodeDecodeError instead of ValueError"
    except ValueError as e:
        assert "ASCII" in str(e) or "ascii" in str(e)
    finally:
        import os; os.unlink(tmp)


def test_read_ply_format_with_extra_spaces():
    """PLY with multiple spaces in format line should still work."""
    content = (
        "ply\n"
        "format  ascii  1.0\n"
        "element vertex 2\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
        "1.0 2.0 3.0\n"
        "4.0 5.0 6.0\n"
    ).encode('ascii')
    tmp = "/tmp/test_spaces.ply"
    with open(tmp, "wb") as f:
        f.write(content)
    try:
        pts = _read_ply_np(tmp)
        assert pts.shape == (2, 3)
        np.testing.assert_array_almost_equal(pts, [[1,2,3],[4,5,6]])
    finally:
        import os; os.unlink(tmp)
