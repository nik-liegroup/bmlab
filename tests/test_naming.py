from unittest.mock import Mock
import datetime

from bmlab.export.naming import repetition_or_timestamp_tag, \
    sanitize_for_filename


def test_sanitize_for_filename():
    assert sanitize_for_filename('Brightfield z overview') == \
        'BrightfieldZOverview'
    assert sanitize_for_filename('Red') == 'Red'


def test_repetition_or_timestamp_tag_with_brillouin_repetition():
    payload = Mock()
    payload.get_brillouin_repetition_index.return_value = 3
    assert repetition_or_timestamp_tag(payload, '0') == '_BMrep3'
    # A real repetition index takes priority - get_date() isn't even
    # consulted.
    payload.get_date.assert_not_called()


def test_repetition_or_timestamp_tag_falls_back_to_timestamp():
    payload = Mock()
    payload.get_brillouin_repetition_index.return_value = None
    payload.get_date.return_value = \
        datetime.datetime(2026, 9, 20, 14, 30, 12)
    assert repetition_or_timestamp_tag(payload, '0') == '_20260920-143012'


def test_repetition_or_timestamp_tag_no_repetition_no_date():
    payload = Mock()
    payload.get_brillouin_repetition_index.return_value = None
    payload.get_date.return_value = None
    assert repetition_or_timestamp_tag(payload, '0') == ''
