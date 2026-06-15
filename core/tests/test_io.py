"""Round-trip IO: read(write(x)) == x."""

from mesh2marker import io


def test_roundtrip_exact(valid_corr, tmp_path):
    path = tmp_path / "correspondence.json"
    io.write(valid_corr, path)
    back = io.read(path)
    assert back == valid_corr
