from areal.tools.check_license_header import fix_file, needs_header


def test_existing_bsd_attribution_is_preserved(tmp_path):
    source = tmp_path / "vendor.py"
    original = "# SPDX-License-Identifier: BSD-3-Clause\nvalue = 1\n"
    source.write_text(original)
    assert not needs_header(source)
    assert source.read_text() == original


def test_missing_header_gets_project_license_after_shebang(tmp_path):
    source = tmp_path / "script.py"
    source.write_text("#!/usr/bin/env python3\nvalue = 1\n")
    assert needs_header(source)
    fix_file(source)
    assert not needs_header(source)
    assert source.read_text().startswith(
        "#!/usr/bin/env python3\n# SPDX-License-Identifier: Apache-2.0\n"
    )
