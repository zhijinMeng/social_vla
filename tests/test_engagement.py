"""Unit tests for engagement fusion."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from social_vla.scoring.engagement import EngagementScorer
from social_vla.types import BBox, PersonTrack


def test_silent_prefers_lam():
    scorer = EngagementScorer()
    person = PersonTrack(track_id=1, bbox=BBox(100, 50, 200, 300), timestamp=0.0, velocity=10, dwell_s=4)
    silent = scorer.fuse(person, lam_prob=0.9, talknet_prob=0.1, vad_active=False, frame_width=640, ready_lam=True, ready_talknet=True)
    audible = scorer.fuse(person, lam_prob=0.9, talknet_prob=0.1, vad_active=True, frame_width=640, ready_lam=True, ready_talknet=True)
    assert silent.engagement > audible.engagement


def test_audible_prefers_talknet():
    scorer = EngagementScorer()
    person = PersonTrack(track_id=1, bbox=BBox(280, 50, 380, 300), timestamp=0.0, velocity=5, dwell_s=2)
    low_talk = scorer.fuse(person, lam_prob=0.2, talknet_prob=0.2, vad_active=True, frame_width=640, ready_lam=True, ready_talknet=True)
    high_talk = scorer.fuse(person, lam_prob=0.2, talknet_prob=0.9, vad_active=True, frame_width=640, ready_lam=True, ready_talknet=True)
    assert high_talk.engagement > low_talk.engagement
