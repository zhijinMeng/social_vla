"""Session state machine for SocialVLA-Engage."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from social_vla.types import SessionState


@dataclass
class TurnRecord:
    role: str
    text: str
    t: float


@dataclass
class Session:
    session_id: str
    state: SessionState = SessionState.IDLE
    target_id: int | None = None
    initiator: str | None = None
    turn_history: list[TurnRecord] = field(default_factory=list)
    cooldown_until: float = 0.0
    last_trigger_t: float = 0.0


class SessionManager:
    def __init__(self, cooldown_s: float = 5.0):
        self.cooldown_s = cooldown_s
        self.session = Session(session_id=str(uuid.uuid4()))

    def tick(self, now: float) -> None:
        if self.session.state == SessionState.COOLDOWN and now >= self.session.cooldown_until:
            self.session.state = SessionState.IDLE
            self.session.target_id = None
            self.session.initiator = None

    def can_engage(self, track_id: int, now: float) -> bool:
        if self.session.state != SessionState.IDLE:
            return False
        if now < self.session.cooldown_until:
            return False
        return True

    def start_robot_initiated(self, track_id: int, now: float) -> None:
        self.session.state = SessionState.ENGAGING
        self.session.target_id = track_id
        self.session.initiator = "robot"
        self.session.last_trigger_t = now

    def enter_dialogue(self, opening: str, now: float) -> None:
        self.session.state = SessionState.IN_DIALOGUE
        self.session.turn_history.append(TurnRecord(role="robot", text=opening, t=now))

    def start_user_initiated(self, track_id: int, user_text: str, reply: str, now: float) -> None:
        self.session.state = SessionState.IN_DIALOGUE
        self.session.target_id = track_id
        self.session.initiator = "user"
        self.session.last_trigger_t = now
        self.session.turn_history.append(TurnRecord(role="user", text=user_text, t=now))
        self.session.turn_history.append(TurnRecord(role="robot", text=reply, t=now))

    def end_dialogue(self, now: float) -> None:
        self.session.state = SessionState.COOLDOWN
        self.session.cooldown_until = now + self.cooldown_s
