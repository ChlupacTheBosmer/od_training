import pytest

from src.cli_utils import _convert_value, parse_unknown_args


pytestmark = pytest.mark.unit


def test_convert_value_basic_types():
    assert _convert_value("true") is True
    assert _convert_value("false") is False
    assert _convert_value("none") is None
    assert _convert_value("12") == 12
    assert _convert_value("3.14") == 3.14
    assert _convert_value("hello") == "hello"


def test_convert_value_comma_separated_tuple():
    value = _convert_value("1,2.5,false,None,text")
    assert value == (1, 2.5, False, None, "text")


def test_parse_unknown_args_typed_values_and_flags():
    unknown = [
        "--lr0",
        "0.01",
        "--freeze",
        "10",
        "--imgsz",
        "640,640",
        "--use-nms",
        "false",
        "--verbose",
    ]

    kwargs = parse_unknown_args(unknown)

    assert kwargs == {
        "lr0": 0.01,
        "freeze": 10,
        "imgsz": (640, 640),
        "use_nms": False,
        "verbose": True,
    }


def test_parse_unknown_args_skips_non_flag_tokens():
    kwargs = parse_unknown_args(["orphan", "--batch", "8"])
    assert kwargs == {"batch": 8}
