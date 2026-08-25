"""CLI smoke tests — argparse, --help, error paths, and the AfriNano shim."""
from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from qvac_nmt._afritranslate_cli import AFRITRANSLATE_TAGS
from qvac_nmt._afritranslate_cli import build_parser as build_at_parser
from qvac_nmt._afritranslate_cli import main as at_main
from qvac_nmt.cli_convert import build_parser as build_convert_parser
from qvac_nmt.cli_convert import main as convert_main
from qvac_nmt.cli_translate import build_parser as build_translate_parser
from qvac_nmt.cli_translate import main as translate_main


def _help_text(parser_factory) -> str:
    p = parser_factory()
    buf = io.StringIO()
    p.print_help(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Generic qvac-translate CLI
# ---------------------------------------------------------------------------


def test_translate_help_does_not_hardcode_african_tags() -> None:
    """The generic CLI must not advertise any specific language whitelist."""
    txt = _help_text(build_translate_parser)
    assert "--models" in txt
    assert "--tag" in txt
    # No reason for AfriNano-specific tags to appear in the generic help.
    for at_tag in AFRITRANSLATE_TAGS:
        assert at_tag not in txt or "##" in txt  # ##TAG example is fine


def test_translate_accepts_arbitrary_tag() -> None:
    """The generic CLI must not reject an unknown tag — only the model
    decides what's valid."""
    parser = build_translate_parser()
    ns = parser.parse_args(["--models", "/tmp", "--tag", "de", "Hello"])
    assert ns.tag == "DE"  # validator uppercases


def test_translate_rejects_invalid_tag_format() -> None:
    parser = build_translate_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--models", "/tmp", "--tag", "this-is-too-long", "x"])


def test_translate_main_returns_2_when_models_missing(tmp_path: Path) -> None:
    err = io.StringIO()
    with redirect_stderr(err):
        rc = translate_main(["--models", str(tmp_path / "nonexistent"), "hi"])
    assert rc == 2
    assert "not a directory" in err.getvalue()


# ---------------------------------------------------------------------------
# AfriNano shim CLI (afritranslate)
# ---------------------------------------------------------------------------


def test_afritranslate_shim_help_lists_only_at_tags() -> None:
    txt = _help_text(build_at_parser)
    for tag in AFRITRANSLATE_TAGS:
        assert tag in txt
    assert "TranslatePsy-AfriNano" in txt


def test_afritranslate_shim_rejects_non_at_tag() -> None:
    parser = build_at_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--models", "/tmp", "--tag", "DE", "x"])


def test_afritranslate_shim_accepts_at_tag() -> None:
    parser = build_at_parser()
    ns = parser.parse_args(["--models", "/tmp", "--tag", "SW", "Hello"])
    assert ns.tag == "SW"


def test_afritranslate_shim_returns_2_when_models_missing(tmp_path: Path) -> None:
    err = io.StringIO()
    with redirect_stderr(err):
        rc = at_main(["--models", str(tmp_path / "nope"), "--tag", "SW", "hi"])
    assert rc == 2


# ---------------------------------------------------------------------------
# qvac-convert CLI
# ---------------------------------------------------------------------------


def test_convert_help_lists_dtype_aliases() -> None:
    txt = _help_text(build_convert_parser)
    for alias in ("fp16", "fp32", "int8", "float16"):
        assert alias in txt


def test_convert_main_returns_2_when_src_missing(tmp_path: Path) -> None:
    err = io.StringIO()
    with redirect_stderr(err):
        rc = convert_main(
            [str(tmp_path / "nope.npz"), str(tmp_path / "out.npz"), "--dtype", "fp16"]
        )
    assert rc == 2
    assert "not found" in err.getvalue()


def test_convert_round_trip_via_cli(tmp_path: Path, tiny_cfg) -> None:
    """End-to-end: save a fresh tiny model, convert via CLI, load it back."""
    from qvac_nmt import MarianSSRU, infer_config, save_npz

    src = tmp_path / "src.npz"
    dst = tmp_path / "dst.fp16.npz"
    save_npz(MarianSSRU(tiny_cfg), src, dtype="float32")

    out = io.StringIO()
    with redirect_stdout(out):
        rc = convert_main([str(src), str(dst), "--dtype", "fp16"])
    assert rc == 0
    assert dst.is_file()
    cfg2 = infer_config(dst)
    assert cfg2.d_model == tiny_cfg.d_model
