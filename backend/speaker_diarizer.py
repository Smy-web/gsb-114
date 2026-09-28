import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class SpeakerSegment:
    speaker_id: str
    speaker_role: str
    start_time: float
    end_time: float
    confidence: float
    text: str = ""


class PyannoteSpeakerDiarizer:
    def __init__(self, auth_token: str = None, use_gpu: bool = False):
        self.auth_token = auth_token or os.getenv('PYANNOTE_AUTH_TOKEN', '')
        self.use_gpu = use_gpu
        self.diarization_pipeline = None
        self.embedding_model = None
        self._role_keywords = {
            'engineer': [
                '钻探', '钻头', '钻机', '钻速', '钻杆', '液压', '马达', '压力',
                '机械', '设备', '维护', '故障', '修理', '安装', '操作', '深度',
                '进尺', '回次', '取芯', '冰芯质量', '卡钻', '偏斜', '方位',
            ],
            'climatologist': [
                '气候', '冰期', '间冰期', 'CO2', '二氧化碳', '甲烷', 'CH4',
                '氧同位素', 'δ18O', '粉尘', '气溶胶', '气泡', '浓度', '年代',
                '古气候', '温室气体', '海平面', '温度', '降水', '环流',
                '冰川', '冰盖', '冰架', '消融', '积累', '雪冰',
            ]
        }
        self._speaker_profiles = {}

    def load_pipeline(self):
        if self.diarization_pipeline is None and self.auth_token:
            try:
                from pyannote.audio import Pipeline
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.auth_token
                )
                if self.use_gpu:
                    self.diarization_pipeline = self.diarization_pipeline.to(torch.device('cuda'))
            except Exception as e:
                print(f"警告: 无法加载pyannote模型，将使用模拟模式: {e}")
        return self.diarization_pipeline

    def _classify_role_by_keywords(self, text: str) -> Tuple[str, float]:
        if not text:
            return 'unknown', 0.5
        
        text_lower = text.lower()
        engineer_score = 0
        climatologist_score = 0
        
        for keyword in self._role_keywords['engineer']:
            if keyword.lower() in text_lower:
                engineer_score += 1
        
        for keyword in self._role_keywords['climatologist']:
            if keyword.lower() in text_lower:
                climatologist_score += 1
        
        total = engineer_score + climatologist_score
        if total == 0:
            return 'unknown', 0.5
        
        if engineer_score > climatologist_score:
            confidence = engineer_score / total
            return 'engineer', confidence
        elif climatologist_score > engineer_score:
            confidence = climatologist_score / total
            return 'climatologist', confidence
        else:
            return 'unknown', 0.5

    def _update_speaker_profile(self, speaker_id: str, text: str):
        if speaker_id not in self._speaker_profiles:
            self._speaker_profiles[speaker_id] = {
                'texts': [],
                'role_votes': {'engineer': 0, 'climatologist': 0, 'unknown': 0}
            }
        
        self._speaker_profiles[speaker_id]['texts'].append(text)
        role, confidence = self._classify_role_by_keywords(text)
        self._speaker_profiles[speaker_id]['role_votes'][role] += confidence

    def _get_speaker_role(self, speaker_id: str) -> Tuple[str, float]:
        if speaker_id not in self._speaker_profiles:
            return 'unknown', 0.5
        
        votes = self._speaker_profiles[speaker_id]['role_votes']
        total_votes = sum(votes.values())
        
        if total_votes == 0:
            return 'unknown', 0.5
        
        best_role = max(votes, key=votes.get)
        confidence = votes[best_role] / total_votes
        
        return best_role, confidence

    def _simulate_diarization(self, audio_path: str, 
                              segments_info: List[Dict[str, Any]] = None) -> List[SpeakerSegment]:
        import librosa
        
        y, sr = librosa.load(audio_path, sr=16000)
        duration = len(y) / sr
        
        if segments_info and len(segments_info) > 0:
            speaker_segments = []
            for i, seg_info in enumerate(segments_info):
                speaker_id = f"SPEAKER_{i % 2:02d}"
                start = seg_info.get('start', i * 5.0)
                end = seg_info.get('end', min(start + 5.0, duration))
                text = seg_info.get('text', '')
                
                self._update_speaker_profile(speaker_id, text)
                role, role_confidence = self._get_speaker_role(speaker_id)
                
                speaker_segments.append(SpeakerSegment(
                    speaker_id=speaker_id,
                    speaker_role=role,
                    start_time=start,
                    end_time=end,
                    confidence=0.7 + 0.2 * role_confidence,
                    text=text
                ))
            return speaker_segments
        
        num_speakers = 2
        segment_duration = 8.0
        speaker_segments = []
        
        current_time = 0.0
        segment_idx = 0
        
        while current_time < duration:
            speaker_id = f"SPEAKER_{segment_idx % num_speakers:02d}"
            seg_end = min(current_time + segment_duration, duration)
            
            speaker_segments.append(SpeakerSegment(
                speaker_id=speaker_id,
                speaker_role='unknown',
                start_time=current_time,
                end_time=seg_end,
                confidence=0.65
            ))
            
            current_time = seg_end
            segment_idx += 1
        
        return speaker_segments

    def diarize_audio(self, audio_path: str, 
                      transcript_segments: List[Dict[str, Any]] = None) -> List[SpeakerSegment]:
        pipeline = self.load_pipeline()
        
        if pipeline is not None:
            try:
                diarization = pipeline(audio_path)
                speaker_segments = []
                
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    speaker_segments.append(SpeakerSegment(
                        speaker_id=speaker,
                        speaker_role='unknown',
                        start_time=turn.start,
                        end_time=turn.end,
                        confidence=0.85
                    ))
            except Exception as e:
                print(f"Pyannote diarization failed, using simulation: {e}")
                speaker_segments = self._simulate_diarization(audio_path, transcript_segments)
        else:
            speaker_segments = self._simulate_diarization(audio_path, transcript_segments)
        
        if transcript_segments:
            speaker_segments = self._align_transcript_with_speakers(
                speaker_segments, transcript_segments
            )
        
        return speaker_segments

    def _align_transcript_with_speakers(self, 
                                       speaker_segments: List[SpeakerSegment],
                                       transcript_segments: List[Dict[str, Any]]) -> List[SpeakerSegment]:
        aligned_segments = []
        
        for transcript_seg in transcript_segments:
            trans_start = transcript_seg.get('start', 0)
            trans_end = transcript_seg.get('end', 0)
            trans_text = transcript_seg.get('text', '')
            
            best_speaker = None
            max_overlap = 0
            
            for speaker_seg in speaker_segments:
                overlap_start = max(trans_start, speaker_seg.start_time)
                overlap_end = min(trans_end, speaker_seg.end_time)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_speaker = speaker_seg
            
            if best_speaker:
                self._update_speaker_profile(best_speaker.speaker_id, trans_text)
                role, role_confidence = self._get_speaker_role(best_speaker.speaker_id)
                
                aligned_segments.append(SpeakerSegment(
                    speaker_id=best_speaker.speaker_id,
                    speaker_role=role,
                    start_time=trans_start,
                    end_time=trans_end,
                    confidence=best_speaker.confidence * (0.6 + 0.4 * role_confidence),
                    text=trans_text
                ))
            else:
                aligned_segments.append(SpeakerSegment(
                    speaker_id='UNKNOWN',
                    speaker_role='unknown',
                    start_time=trans_start,
                    end_time=trans_end,
                    confidence=0.5,
                    text=trans_text
                ))
        
        return aligned_segments

    def get_speaker_summary(self) -> Dict[str, Any]:
        summary = {}
        for speaker_id, profile in self._speaker_profiles.items():
            role, confidence = self._get_speaker_role(speaker_id)
            summary[speaker_id] = {
                'role': role,
                'confidence': confidence,
                'total_segments': len(profile['texts']),
                'total_words': sum(len(text.split()) for text in profile['texts'])
            }
        return summary

    def reset_profiles(self):
        self._speaker_profiles = {}


def create_diarizer(auth_token: str = None) -> PyannoteSpeakerDiarizer:
    return PyannoteSpeakerDiarizer(auth_token=auth_token)


def diarize_and_label(audio_path: str, 
                      transcript_segments: List[Dict[str, Any]] = None,
                      auth_token: str = None) -> List[Dict[str, Any]]:
    diarizer = create_diarizer(auth_token)
    segments = diarizer.diarize_audio(audio_path, transcript_segments)
    
    return [{
        'speaker_id': seg.speaker_id,
        'speaker_role': seg.speaker_role,
        'start_time': seg.start_time,
        'end_time': seg.end_time,
        'confidence': seg.confidence,
        'text': seg.text
    } for seg in segments]
