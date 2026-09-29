"""回归用例：一次调用一个角色、对齐置信度与 needs_review、计分开关、
中性/同分裁决、CJK 字数、diarize_and_label 输出键。全部用替身，不碰音频后端。"""
import pytest

from backend import speaker_diarizer
from backend.speaker_diarizer import (
    PyannoteSpeakerDiarizer,
    SpeakerSegment,
    create_diarizer,
)


class _Turn:
    def __init__(self, start, end):
        self.start = float(start)
        self.end = float(end)


class _Diarization:
    def __init__(self, turns):
        self._turns = turns

    def itertracks(self, yield_label=False):
        for (start, end), speaker in self._turns:
            yield _Turn(start, end), None, speaker


def make_diarizer(turns, **kwargs):
    diarizer = PyannoteSpeakerDiarizer(auth_token="stub-token", **kwargs)
    diarizer.diarization_pipeline = lambda audio_path: _Diarization(turns)
    return diarizer


def seg(speaker_id, start, end, confidence=0.85):
    return SpeakerSegment(speaker_id, "unknown", start, end, confidence, "")


def test_single_call_gives_one_final_role_per_speaker():
    diarizer = make_diarizer([((0.0, 10.0), "SPEAKER_00"), ((10.0, 20.0), "SPEAKER_00")])
    transcript = [
        {"start": 0.0, "end": 10.0, "text": "今天风很大，随便聊聊"},
        {"start": 10.0, "end": 20.0, "text": "钻杆液压压力正常，本回次进尺三米"},
    ]
    segments = diarizer.diarize_audio("fake.wav", transcript)
    assert [s.speaker_id for s in segments] == ["SPEAKER_00", "SPEAKER_00"]
    assert [s.speaker_role for s in segments] == ["engineer", "engineer"]


def test_alignment_confidence_flags_low_overlap_segments():
    diarizer = PyannoteSpeakerDiarizer(auth_token="stub-token")
    speakers = [seg("SPEAKER_00", 0.0, 10.0, confidence=0.8)]
    transcript = [
        {"start": 0.0, "end": 10.0, "text": "钻杆"},
        {"start": 0.0, "end": 20.0, "text": "钻杆"},
    ]
    segments = diarizer._align_transcript_with_speakers(speakers, transcript)
    assert segments[0].needs_review is False
    assert segments[1].needs_review is True
    assert segments[0].confidence == pytest.approx(0.8)


def test_review_threshold_is_configurable():
    assert PyannoteSpeakerDiarizer().review_threshold == 0.6
    assert PyannoteSpeakerDiarizer(review_threshold=0.3).review_threshold == 0.3
    assert create_diarizer(review_threshold=0.9).review_threshold == 0.9

    diarizer = PyannoteSpeakerDiarizer(review_threshold=0.3)
    speakers = [seg("SPEAKER_00", 0.0, 10.0, confidence=0.8)]
    transcript = [{"start": 0.0, "end": 20.0, "text": "钻杆"}]
    segments = diarizer._align_transcript_with_speakers(speakers, transcript)
    assert segments[0].needs_review is False


def test_low_confidence_segments_do_not_score_roles():
    diarizer = PyannoteSpeakerDiarizer(auth_token="stub-token")
    speakers = [seg("SPEAKER_00", 0.0, 10.0, confidence=0.85)]
    transcript = [
        {"start": 0.0, "end": 20.0, "text": "钻杆液压压力正常"},
        {"start": 0.0, "end": 10.0, "text": "二氧化碳甲烷浓度偏高"},
    ]
    segments = diarizer._align_transcript_with_speakers(speakers, transcript)
    assert [s.needs_review for s in segments] == [True, False]
    assert all(s.speaker_role == "climatologist" for s in segments)

    entry = diarizer.get_speaker_summary()["SPEAKER_00"]
    assert entry["role"] == "climatologist"
    assert entry["total_segments"] == 2
    assert entry["needs_review_segments"] == 1


def test_neutral_segments_do_not_dilute_role():
    diarizer = PyannoteSpeakerDiarizer()
    diarizer._update_speaker_profile("SPEAKER_00", "今天风很大，随便聊聊")
    diarizer._update_speaker_profile("SPEAKER_00", "钻杆液压压力正常")
    role, confidence = diarizer._get_speaker_role("SPEAKER_00")
    assert role == "engineer"
    assert confidence == pytest.approx(1.0)


def test_tie_between_roles_returns_unknown_explicitly():
    diarizer = PyannoteSpeakerDiarizer()
    diarizer._update_speaker_profile("SPEAKER_00", "钻杆")
    diarizer._update_speaker_profile("SPEAKER_00", "气候")
    assert diarizer._get_speaker_role("SPEAKER_00") == ("unknown", 0.5)


def test_cjk_and_mixed_word_count():
    diarizer = PyannoteSpeakerDiarizer()
    diarizer._update_speaker_profile("SPEAKER_00", "本回次进尺 3.2 米")
    assert diarizer.get_speaker_summary()["SPEAKER_00"]["total_words"] == 7

    diarizer.reset_profiles()
    diarizer._update_speaker_profile("SPEAKER_00", "CO2 浓度偏高")
    assert diarizer.get_speaker_summary()["SPEAKER_00"]["total_words"] == 5


def test_unmatched_transcript_marked_needs_review():
    diarizer = PyannoteSpeakerDiarizer(auth_token="stub-token")
    transcript = [{"start": 50.0, "end": 60.0, "text": "钻杆液压压力正常"}]
    segments = diarizer._align_transcript_with_speakers(
        [seg("SPEAKER_00", 0.0, 10.0)], transcript)
    assert segments[0].speaker_id == "UNKNOWN"
    assert segments[0].speaker_role == "unknown"
    assert segments[0].confidence == pytest.approx(0.5)
    assert segments[0].needs_review is True


def test_diarize_and_label_keeps_six_keys_and_adds_needs_review(monkeypatch):
    class _StubDiarizer:
        def diarize_audio(self, audio_path, transcript_segments=None):
            return [SpeakerSegment("SPEAKER_00", "engineer", 0.0, 1.0, 0.9,
                                   "钻杆", needs_review=True)]

    monkeypatch.setattr(speaker_diarizer, "create_diarizer",
                        lambda *args, **kwargs: _StubDiarizer())
    rows = speaker_diarizer.diarize_and_label(
        "fake.wav", [{"start": 0.0, "end": 1.0, "text": "钻杆"}])
    assert len(rows) == 1
    row = rows[0]
    assert {"speaker_id", "speaker_role", "start_time", "end_time",
            "confidence", "text", "needs_review"} <= set(row)
    assert row["speaker_id"] == "SPEAKER_00"
    assert row["speaker_role"] == "engineer"
    assert row["needs_review"] is True
