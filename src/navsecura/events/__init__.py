"""Visit event lifecycle, state machine, and spatio-temporal stitching package."""

from __future__ import annotations

from navsecura.events.models import CloseReason, Visit, VisitState
from navsecura.events.state_machine import (
    TrackVisitSession,
    VisitStateMachine,
)
from navsecura.events.stitch import TrackletStitcher

__all__ = [
    "Visit",
    "VisitState",
    "CloseReason",
    "TrackVisitSession",
    "VisitStateMachine",
    "TrackletStitcher",
]
