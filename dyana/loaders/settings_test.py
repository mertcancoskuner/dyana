import pytest

from dyana.loaders.settings import LoaderArgument, LoaderSettings, ParsedArgument


def test_parse_build_args() -> None:
    settings = LoaderSettings(description="test loader", build_args={"model": "MODEL_ARG", "version": "VERSION_ARG"})

    args = ["--model", "gpt2", "--version", "1.0"]
    build_args = settings.parse_build_args(args)

    assert build_args == {"MODEL_ARG": "gpt2", "VERSION_ARG": "1.0"}


def test_parse_build_args_with_equals() -> None:
    settings = LoaderSettings(description="test loader", build_args={"model": "MODEL_ARG"})

    args = ["--model=gpt2"]
    build_args = settings.parse_build_args(args)

    assert build_args == {"MODEL_ARG": "gpt2"}


def test_parse_args() -> None:
    settings = LoaderSettings(
        description="test loader",
        args=[
            LoaderArgument(name="model", description="Model to use"),
            LoaderArgument(name="data", description="Data path", volume=True),
        ],
    )

    args = ["--model", "gpt2", "--data", "/path/to/data"]
    parsed_args = settings.parse_args(args) or []

    assert len(parsed_args) == 2
    assert parsed_args[0] == ParsedArgument(name="model", value="gpt2", volume=False)
    assert parsed_args[1] == ParsedArgument(name="data", value="/path/to/data", volume=True)


def test_parse_args_with_default() -> None:
    settings = LoaderSettings(
        description="test loader", args=[LoaderArgument(name="model", description="Model to use", default="gpt2")]
    )

    args: list[str] = []
    parsed_args = settings.parse_args(args) or []

    assert len(parsed_args) == 1
    assert parsed_args[0] == ParsedArgument(name="model", value="gpt2", volume=False)


def test_parse_args_missing_required() -> None:
    settings = LoaderSettings(
        description="test loader", args=[LoaderArgument(name="model", description="Model to use", required=True)]
    )

    args: list[str] = []
    with pytest.raises(ValueError, match="Argument --model is required"):
        settings.parse_args(args)
