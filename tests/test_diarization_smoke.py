"""现有测试：用替身 pipeline 覆盖既有正常路径。跑通说明角色归属的基本流程没坏。"""
import pytest

from backend import speaker_diarizer
from backend.speaker_diarizer import PyannoteSpeakerDiarizer, SpeakerSegment


class _Turn:
    def __init__(self, start, end):
        self.start = float(start)
        self.end = float(end)


class _Diarization:
    def __init__(self, turns):
        self._turns = turns

    def itertracks(self, yield_label=False):
        for turn, speaker in self._turns:
            if isinstance(turn, tuple):
                turn = _Turn(*turn)
            yield turn, None, speaker


def stub_pipeline(turns):
    def _run(audio_path):
        return _Diarization(turns)

    return _run


def make_diarizer(turns):
    diarizer = PyannoteSpeakerDiarizer(auth_token="stub-token")
    diarizer.diarization_pipeline = stub_pipeline(turns)
    return diarizer


def seg(speaker_id, start, end, confidence=0.85):
    return SpeakerSegment(speaker_id, "unknown", start, end, confidence, "")


TWO_SPEAKERS = [((0.0, 10.0), "SPEAKER_00"), ((10.0, 20.0), "SPEAKER_01")]


def test_stub_pipeline_yields_speaker_segments():
    diarizer = make_diarizer(TWO_SPEAKERS)
    segments = diarizer.diarize_audio("fake.wav")
    assert [s.speaker_id for s in segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert [s.speaker_role for s in segments] == ["unknown", "unknown"]
    assert all(s.confidence == pytest.approx(0.85) for s in segments)
    assert (segments[0].start_time, segments[0].end_time) == (0.0, 10.0)
    assert (segments[1].start_time, segments[1].end_time) == (10.0, 20.0)


def test_align_maps_text_onto_speaker():
    diarizer = PyannoteSpeakerDiarizer(auth_token="stub-token")
    transcript = [
        {"start": 0.0, "end": 10.0, "text": "钻杆液压压力正常，本回次进尺三米"},
        {"start": 10.0, "end": 20.0, "text": "这段气泡浓度偏高，二氧化碳要复测"},
    ]
    segments = diarizer._align_transcript_with_speakers(
        [seg("SPEAKER_00", 0.0, 10.0), seg("SPEAKER_01", 10.0, 20.0)], transcript)
    assert [s.speaker_id for s in segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert [s.text for s in segments] == [row["text"] for row in transcript]
    assert [(s.start_time, s.end_time) for s in segments] == [(0.0, 10.0), (10.0, 20.0)]
    assert [s.speaker_role for s in segments] == ["engineer", "climatologist"]
    assert all(0.0 < s.confidence <= 1.0 for s in segments)


def test_unmatched_transcript_falls_back_to_unknown():
    diarizer = PyannoteSpeakerDiarizer(auth_token="stub-token")
    transcript = [{"start": 50.0, "end": 60.0, "text": "钻杆液压压力正常"}]
    segments = diarizer._align_transcript_with_speakers(
        [seg("SPEAKER_00", 0.0, 10.0)], transcript)
    assert len(segments) == 1
    assert segments[0].speaker_id == "UNKNOWN"
    assert segments[0].speaker_role == "unknown"
    assert segments[0].confidence == pytest.approx(0.5)


def test_summary_reports_each_speaker():
    diarizer = PyannoteSpeakerDiarizer(auth_token="stub-token")
    diarizer._update_speaker_profile("SPEAKER_00", "钻杆液压压力正常")
    summary = diarizer.get_speaker_summary()
    assert list(summary) == ["SPEAKER_00"]
    entry = summary["SPEAKER_00"]
    assert {"role", "confidence", "total_segments", "total_words"} <= set(entry)
    assert entry["role"] == "engineer"
    assert entry["total_segments"] == 1


def test_latin_word_count_is_whitespace_based():
    diarizer = PyannoteSpeakerDiarizer(auth_token="stub-token")
    diarizer._update_speaker_profile("SPEAKER_01", "drill pipe pressure ok")
    entry = diarizer.get_speaker_summary()["SPEAKER_01"]
    assert entry["total_words"] == 4
    assert entry["role"] == "unknown"


def test_keyword_classification_and_reset():
    diarizer = PyannoteSpeakerDiarizer(auth_token="stub-token")
    assert diarizer._classify_role_by_keywords("钻杆液压压力正常")[0] == "engineer"
    assert diarizer._classify_role_by_keywords("二氧化碳和甲烷浓度偏高")[0] == "climatologist"
    assert diarizer._classify_role_by_keywords("") == ("unknown", 0.5)
    assert diarizer._classify_role_by_keywords("今天风很大")[0] == "unknown"

    diarizer._update_speaker_profile("SPEAKER_00", "钻杆液压压力正常")
    assert diarizer.get_speaker_summary() != {}
    diarizer.reset_profiles()
    assert diarizer.get_speaker_summary() == {}
