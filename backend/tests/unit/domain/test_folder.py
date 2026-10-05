import pytest

from daikonstudio.domain.shared.errors import ValidationError
from daikonstudio.domain.shared.folder import clean_folder_name


def test_a_name_is_stripped():
    assert clean_folder_name("  Gyrase ") == "Gyrase"


@pytest.mark.parametrize("raw", ["", "   ", "x" * 101])
def test_a_name_must_be_1_to_100_characters(raw):
    with pytest.raises(ValidationError, match="1 to 100 characters"):
        clean_folder_name(raw)


def test_a_100_character_name_is_fine():
    assert clean_folder_name("x" * 100) == "x" * 100
