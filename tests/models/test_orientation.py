from bmlab.models import Orientation
from bmlab.session import Session


def test_reflection_is_not_shared_between_orientations():
    first = Orientation()
    first.set_reflection(vertically=True, horizontally=True)

    second = Orientation()

    assert second.reflection == {'vertically': False, 'horizontally': False}
    assert first.reflection == {'vertically': True, 'horizontally': True}


def test_given_reflection_is_copied_and_completed():
    given = {'vertically': True}
    orientation = Orientation(rotation=1, reflection=given)

    orientation.set_reflection(horizontally=True)

    assert given == {'vertically': True}  # the caller's dict is untouched
    assert orientation.reflection == {
        'vertically': True, 'horizontally': True}
    assert orientation.rotation == 1


def test_a_cleared_session_does_not_keep_the_previous_reflection():
    session = Session.get_instance()
    session.clear()
    session.set_reflection(vertically=True)

    session.clear()

    assert session.orientation.reflection == {
        'vertically': False, 'horizontally': False}
