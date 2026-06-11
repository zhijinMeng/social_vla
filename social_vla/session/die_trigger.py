"""Dyadic Interaction Encoder (DIE) — core MEN algorithm module.

Replaces the hand-crafted TriggerRouter with a learned model that jointly
models human signals and robot action history to infer interaction intent.

Architecture:
  Per-person: linear projection → Transformer (temporal) → cross-attention
              (human signals ↔ robot actions) → belief + stage + readiness heads
  Multi-party: priority scoring with social constraints → TriggerDecision

In untrained state (no checkpoint), falls back to TriggerRouter heuristics
so the rest of the system keeps working during development.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque

import torch
import torch.nn as nn

from social_vla.session.manager import SessionManager
from social_vla.session.trigger import TriggerDecision, TriggerRouter
from social_vla.types import PerPersonScores, SessionState

# ── Constants ────────────────────────────────────────────────────────────────

HUMAN_DIM = 6   # lam, talknet, approach_vel, body_orient, speech_vad, dwell
ROBOT_DIM = 5   # one-hot: wait / head_turn / orient / speak / disengage
INPUT_DIM = HUMAN_DIM + ROBOT_DIM

ROBOT_ACTION_WAIT       = 0
ROBOT_ACTION_HEAD_TURN  = 1
ROBOT_ACTION_ORIENT     = 2
ROBOT_ACTION_SPEAK      = 3
ROBOT_ACTION_DISENGAGE  = 4

# Interaction stage names (indices 0-4)
STAGE_NAMES = ["detection", "awareness", "approach", "opening", "dialogue"]

# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class DIEConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 4
    history_len: int = 30          # frames of history per person (~3s at 10Hz)
    dropout: float = 0.1
    checkpoint: str | None = None  # path to trained weights
    # Optimal stopping threshold: initiate when P(wants) >= cost_ratio
    # cost_ratio = C_awk / (R_r + C_awk); default ~0.4 for exhibition
    cost_ratio: float = 0.4
    # Multi-party priority weights (learned from data; these are init values)
    priority_alpha: float = 0.5    # belief weight
    priority_beta: float = 0.3     # readiness weight
    priority_gamma: float = 0.2    # waiting-time fairness weight
    fallback_to_heuristic: bool = True  # use TriggerRouter if untrained


@dataclass
class DyadicBelief:
    """Per-person belief state maintained across frames."""
    track_id: int
    # Ring buffer of (human_signal, robot_action) pairs
    history: Deque[tuple[list[float], int]] = field(
        default_factory=lambda: deque(maxlen=30)
    )
    # Latest model outputs (updated each frame)
    p_wants: float = 0.0           # P(θ = wants_interaction)
    p_passing: float = 0.0         # P(θ = passing_by)
    p_occupied: float = 0.0        # P(θ = occupied_elsewhere)
    stage: int = 0                 # current interaction stage [0-4]
    readiness: float = 0.0         # readiness for next-stage action
    # How long this person has been visible without initiation (for fairness)
    waiting_frames: int = 0
    # Last robot action taken toward this person
    last_robot_action: int = ROBOT_ACTION_WAIT


# ── Neural network ───────────────────────────────────────────────────────────

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class DyadicInteractionEncoder(nn.Module):
    """
    Input:  (B, T, INPUT_DIM)  — T timesteps of [human_signal | robot_action_onehot]
    Output: p_type  (B, 3)     — P(wants / passing / occupied)
            stage   (B, 5)     — P(stage 0..4)
            readiness (B, 1)   — scalar readiness for next-stage escalation
    """

    def __init__(self, cfg: DIEConfig):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_DIM, cfg.d_model)
        self.pos_enc = _PositionalEncoding(cfg.d_model, max_len=cfg.history_len + 4)

        # Separate encoders for human and robot streams (for cross-attention)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4, dropout=cfg.dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)

        # Cross-attention: robot queries, human keys/values
        # "How did my actions affect their signals?"
        self.human_proj  = nn.Linear(HUMAN_DIM, cfg.d_model)
        self.robot_proj  = nn.Linear(ROBOT_DIM,  cfg.d_model)
        self.cross_attn  = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )

        # Output heads (operate on mean-pooled representation)
        self.head_type      = nn.Linear(cfg.d_model * 2, 3)   # wants/passing/occupied
        self.head_stage     = nn.Linear(cfg.d_model * 2, 5)   # stage 0-4
        self.head_readiness = nn.Linear(cfg.d_model * 2, 1)   # scalar

    def forward(
        self,
        x: torch.Tensor,              # (B, T, INPUT_DIM)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape

        # Joint temporal encoding
        h = self.pos_enc(self.input_proj(x))           # (B, T, d)
        h = self.temporal_encoder(h)                    # (B, T, d)
        temporal_rep = h.mean(dim=1)                    # (B, d)

        # Cross-stream attention: robot → human
        human_emb = self.pos_enc(self.human_proj(x[..., :HUMAN_DIM]))   # (B,T,d)
        robot_emb = self.pos_enc(self.robot_proj(x[..., HUMAN_DIM:]))   # (B,T,d)
        cross_rep, _ = self.cross_attn(robot_emb, human_emb, human_emb) # (B,T,d)
        cross_rep = cross_rep.mean(dim=1)               # (B, d)

        rep = torch.cat([temporal_rep, cross_rep], dim=-1)  # (B, 2d)

        p_type    = torch.softmax(self.head_type(rep),      dim=-1)   # (B, 3)
        stage     = torch.softmax(self.head_stage(rep),     dim=-1)   # (B, 5)
        readiness = torch.sigmoid(self.head_readiness(rep)).squeeze(-1)  # (B,)

        return p_type, stage, readiness


# ── Per-person belief manager ────────────────────────────────────────────────

class DIETrigger:
    """
    Drop-in replacement for TriggerRouter.

    Call evaluate() each frame — same signature and return type.
    Maintains per-track belief state internally.
    Falls back to TriggerRouter if the model is untrained (no checkpoint).
    """

    def __init__(self, cfg: DIEConfig | None = None, device: str = "cpu"):
        self.cfg = cfg or DIEConfig()
        self.device = device
        self._beliefs: dict[int, DyadicBelief] = {}
        self._model: DyadicInteractionEncoder | None = None
        self._trained = False
        self._fallback = TriggerRouter()  # heuristic fallback
        self._last_vad_active = False

        self._load_model()

    def _load_model(self) -> None:
        self._model = DyadicInteractionEncoder(self.cfg).to(self.device)
        self._model.eval()

        if self.cfg.checkpoint and Path(self.cfg.checkpoint).is_file():
            state = torch.load(self.cfg.checkpoint, map_location=self.device)
            self._model.load_state_dict(state)
            self._trained = True

    def reset(self) -> None:
        self._beliefs.clear()
        self._fallback.reset()
        self._last_vad_active = False

    # ── Main entry point ──────────────────────────────────────────────────

    def evaluate(
        self,
        scores: list[PerPersonScores],
        session: SessionManager,
        vad_active: bool,
        utterance_duration_s: float,
        doa_track_id: int | None,
        now: float,
        last_robot_action: dict[int, int] | None = None,
    ) -> TriggerDecision:
        """Evaluate one frame and return a trigger decision.

        Args:
            scores:               Per-person perception scores from Layer A.
            session:              Current session state machine.
            vad_active:           Whether VAD is currently active.
            utterance_duration_s: Duration of current utterance.
            doa_track_id:         Direction-of-arrival track ID (if available).
            now:                  Current timestamp.
            last_robot_action:    Dict[track_id → robot action enum] for this frame.
                                  Used to update belief histories.
        """
        sm = session.session

        # In-dialogue turn detection (aligned with TriggerRouter)
        if sm.state == SessionState.IN_DIALOGUE:
            vad_ended = self._last_vad_active and not vad_active
            if vad_ended and utterance_duration_s >= 0.5:
                self._last_vad_active = vad_active
                return TriggerDecision("turn", sm.target_id, "user_utterance_end")
            self._last_vad_active = vad_active
            return TriggerDecision(None, sm.target_id, "in_dialogue_wait")

        if sm.state != SessionState.IDLE:
            return TriggerDecision(None, None, f"state_{sm.state.value}")

        # User-initiated on utterance end (aligned with TriggerRouter)
        vad_ended = self._last_vad_active and not vad_active
        if vad_ended and utterance_duration_s >= 0.5:
            candidate = self._best_speech_directed(scores, doa_track_id)
            self._last_vad_active = vad_active
            if candidate is not None:
                return TriggerDecision("user_initiated", candidate, "vad_end+speech_directed")
            return TriggerDecision(None, None, "vad_end_no_speaker")

        self._last_vad_active = vad_active

        if not scores:
            return TriggerDecision(None, None, "no_person")

        # Update belief state for each visible person
        active_ids = {s.track_id for s in scores}
        self._prune_stale(active_ids)

        for s in scores:
            self._update_belief(
                s,
                robot_action=(last_robot_action or {}).get(s.track_id, ROBOT_ACTION_WAIT),
            )

        # Untrained model → fall back to heuristic (sync VAD edge state)
        if not self._trained and self.cfg.fallback_to_heuristic:
            self._fallback._last_vad_active = self._last_vad_active
            return self._fallback.evaluate(
                scores, session, vad_active, utterance_duration_s, doa_track_id, now
            )

        # Run inference and update beliefs
        self._run_inference(scores)

        # Multi-party arbitration → pick best candidate
        best_id, best_belief = self._arbitrate(scores, session, now)
        if best_id is None or best_belief is None:
            return TriggerDecision(None, None, "no_candidate")

        # Optimal stopping: initiate when P(wants) ≥ cost_ratio threshold
        if best_belief.p_wants >= self.cfg.cost_ratio:
            return TriggerDecision(
                "robot_initiated",
                best_id,
                f"DIE:p_wants={best_belief.p_wants:.2f} stage={best_belief.stage}",
                score=best_belief.p_wants,
            )

        return TriggerDecision(
            None, best_id, "below_threshold", score=best_belief.p_wants
        )

    # ── Belief management ─────────────────────────────────────────────────

    def _update_belief(self, scores: PerPersonScores, robot_action: int) -> None:
        tid = scores.track_id
        if tid not in self._beliefs:
            self._beliefs[tid] = DyadicBelief(track_id=tid)

        b = self._beliefs[tid]
        b.waiting_frames += 1
        b.last_robot_action = robot_action

        human_signal = [
            scores.lam_prob,
            scores.talknet_prob,
            scores.approach_speed,
            scores.body_orient,
            float(scores.speech_directed > 0.3),
            scores.dwell_time,
        ]
        b.history.append((human_signal, robot_action))

    def _run_inference(self, scores: list[PerPersonScores]) -> None:
        """Batch inference for all visible people."""
        assert self._model is not None
        tids = [s.track_id for s in scores if s.track_id in self._beliefs]
        if not tids:
            return

        tensors = []
        for tid in tids:
            b = self._beliefs[tid]
            tensors.append(self._belief_to_tensor(b))

        x = torch.stack(tensors).to(self.device)  # (N, T, INPUT_DIM)

        with torch.no_grad():
            p_type, stage_logits, readiness = self._model(x)

        for i, tid in enumerate(tids):
            b = self._beliefs[tid]
            pt = p_type[i].cpu().tolist()
            b.p_wants, b.p_passing, b.p_occupied = pt[0], pt[1], pt[2]
            b.stage = int(stage_logits[i].argmax().item())
            b.readiness = float(readiness[i].item())

    def _belief_to_tensor(self, b: DyadicBelief) -> torch.Tensor:
        """Convert DyadicBelief history to (T, INPUT_DIM) tensor, zero-padded."""
        T = self.cfg.history_len
        out = torch.zeros(T, INPUT_DIM)
        hist = list(b.history)
        for j, (human_sig, robot_act) in enumerate(hist[-T:]):
            out[j, :HUMAN_DIM] = torch.tensor(human_sig, dtype=torch.float32)
            out[j, HUMAN_DIM + robot_act] = 1.0   # one-hot robot action
        return out

    def _prune_stale(self, active_ids: set[int]) -> None:
        for tid in list(self._beliefs):
            if tid not in active_ids:
                del self._beliefs[tid]

    # ── Multi-party arbitration ───────────────────────────────────────────

    def _arbitrate(
        self, scores: list[PerPersonScores], session: SessionManager, now: float
    ) -> tuple[int | None, DyadicBelief | None]:
        """Select the best initiation candidate with social constraints."""
        candidates = []
        for s in scores:
            if not session.can_engage(s.track_id, now):
                continue
            b = self._beliefs.get(s.track_id)
            if b is None:
                continue
            # Hard constraint: skip if person appears to be in mutual conversation
            # (high talknet + another person nearby also has high talknet → social pair)
            if self._is_in_social_pair(s, scores):
                continue
            priority = (
                self.cfg.priority_alpha * b.p_wants
                + self.cfg.priority_beta  * b.readiness
                + self.cfg.priority_gamma * min(b.waiting_frames / 100.0, 1.0)
            )
            candidates.append((priority, s.track_id, b))

        if not candidates:
            return None, None

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, best_id, best_belief = candidates[0]
        return best_id, best_belief

    @staticmethod
    def _is_in_social_pair(
        target: PerPersonScores, all_scores: list[PerPersonScores]
    ) -> bool:
        """True if target appears to be in mutual conversation with another person."""
        if target.talknet_prob < 0.5:
            return False
        # Another person also has high talknet (they're talking to each other, not robot)
        others_talking = [
            s for s in all_scores
            if s.track_id != target.track_id and s.talknet_prob >= 0.5
        ]
        return len(others_talking) > 0

    # ── Speech-directed (unchanged from TriggerRouter) ────────────────────

    def _best_speech_directed(
        self, scores: list[PerPersonScores], doa_track_id: int | None
    ) -> int | None:
        if doa_track_id is not None:
            return doa_track_id
        eligible = [
            s for s in scores
            if s.talknet_prob >= 0.6 or s.speech_directed >= 0.6
        ]
        if not eligible:
            return None
        return max(eligible, key=lambda s: (s.talknet_prob, s.speech_directed)).track_id

    # ── Inspection helpers ────────────────────────────────────────────────

    def beliefs(self) -> dict[int, DyadicBelief]:
        return dict(self._beliefs)

    def is_trained(self) -> bool:
        return self._trained
