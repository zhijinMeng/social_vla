"""Tests for Qwen-Omni turn-taking and PCM preroll."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from social_vla.dialogue.qwen_omni_adapter import QwenOmniAdapter, QwenOmniConfig, _PcmPreroll
from social_vla.session.trigger import TriggerRouter
from social_vla.session.manager import SessionManager
from social_vla.types import PerPersonScores, SessionState


def test_preroll_flush_since():
    buf = _PcmPreroll(max_samples=16000)
    buf.push(1.0, np.array([1, 2], dtype=np.int16))
    buf.push(2.0, np.array([3, 4], dtype=np.int16))
    buf.push(3.0, np.array([5, 6], dtype=np.int16))
    flushed = buf.flush_since(2.0)
    assert len(flushed) == 2
    assert flushed[0].tolist() == [3, 4]


def test_begin_user_turn_replays_preroll():
    cfg = QwenOmniConfig(mock=True, echo_guard_ms=0.0, pcm_preroll_s=1.0, preroll_lead_s=0.3)
    adapter = QwenOmniAdapter(cfg)
    adapter._connected = True

    pcm = np.ones(160, dtype=np.int16)
    t0 = time.time()
    adapter.push_user_pcm(pcm, timestamp=t0 - 0.2)
    adapter.note_vad(True)
    adapter.push_user_pcm(pcm, timestamp=t0 + 0.05)

    adapter.begin_user_turn()

    assert adapter._in_turn is True
    assert not adapter._audio_q.empty()


def test_arm_listening_ignores_stale_echo_vad():
    """Stale VAD from speaker echo must not auto-start a user turn."""
    cfg = QwenOmniConfig(mock=True, echo_guard_ms=0.0)
    adapter = QwenOmniAdapter(cfg)
    adapter._connected = True
    adapter._listening = False
    adapter._vad_active = True

    adapter._arm_listening_after_guard()
    time.sleep(0.05)

    assert adapter._listening is True
    assert adapter._in_turn is False
    assert adapter._vad_active is False


def test_gate_vad_blocks_during_playback():
    cfg = QwenOmniConfig(mock=True)
    adapter = QwenOmniAdapter(cfg)
    adapter._player._playing.set()
    assert adapter.gate_vad(True) is False
    adapter._player._playing.clear()
    assert adapter.gate_vad(True) is True


def test_user_initiated_on_vad_falling_edge():
    router = TriggerRouter()
    session = SessionManager()
    scores = [
        PerPersonScores(track_id=1, talknet_prob=0.8, speech_directed=0.8, engagement=0.7)
    ]

    d1 = router.evaluate(scores, session, vad_active=True, utterance_duration_s=0.9, doa_track_id=1, now=0.0)
    assert d1.path is None

    d2 = router.evaluate(scores, session, vad_active=False, utterance_duration_s=0.9, doa_track_id=1, now=0.1)
    assert d2.path == "user_initiated"
    assert d2.target_id == 1


def test_turn_only_on_falling_edge_in_dialogue():
    router = TriggerRouter()
    session = SessionManager()
    session.session.state = SessionState.IN_DIALOGUE
    session.session.target_id = 1

    d1 = router.evaluate([], session, vad_active=True, utterance_duration_s=0.9, doa_track_id=1, now=0.0)
    assert d1.path is None

    d2 = router.evaluate([], session, vad_active=False, utterance_duration_s=0.9, doa_track_id=1, now=0.1)
    assert d2.path == "turn"
