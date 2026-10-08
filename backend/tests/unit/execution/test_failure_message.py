from daikonstudio.application.execution.failure_message import user_facing_error
from daikonstudio.domain.shared.errors import ValidationError


def test_domain_errors_pass_through_with_their_detail():
    error = ValidationError("conditions invalid", detail="n_estimators above maximum 2000")
    assert user_facing_error(error) == "conditions invalid (n_estimators above maximum 2000)"


def test_a_domain_error_without_detail_is_just_its_message():
    assert user_facing_error(ValidationError("no valid structures")) == "no valid structures"


def test_library_value_errors_keep_their_text():
    message = user_facing_error(ValueError("Input y contains NaN."))
    assert message == "Input y contains NaN."


def test_everything_else_is_reduced_to_its_class():
    message = user_facing_error(FileNotFoundError("/data/blobs/ws/protocols/x/model.joblib"))
    assert "/data/blobs" not in message
    assert "FileNotFoundError" in message


def test_a_degradable_leg_does_not_say_the_run_failed() -> None:
    """Two callers report a failure on a run that *succeeds*: the optimism gap and the
    extra split draws. "The run failed" inside a message that also says two of three
    draws completed contradicts itself, on a card whose purpose is honest reporting."""
    message = user_facing_error(KeyError("target"), subject="This draw")

    assert "The run failed" not in message
    assert message.startswith("This draw failed")
    assert "KeyError" in message


def test_the_subject_defaults_to_the_run() -> None:
    # A genuine run-level failure keeps the wording it has always had.
    assert user_facing_error(KeyError("x")).startswith("The run failed")
