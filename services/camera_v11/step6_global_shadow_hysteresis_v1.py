from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from .step5_global_shadow_v1 import (
    CONFIRMED_SHADOW,
    EXPIRED_SHADOW,
    GLOBAL_SHADOW_CONFIRM,
    GLOBAL_SHADOW_CONFLICT,
    GLOBAL_SHADOW_EXPIRE,
    GLOBAL_SHADOW_OBSERVE,
    GlobalShadowEventV1,
)


VERIFY_PENDING = "VERIFY_PENDING"
VERIFIED_SHADOW = "VERIFIED_SHADOW"
CONFLICT_HOLD_SHADOW = "CONFLICT_HOLD_SHADOW"
EXPIRED_VERIFY_SHADOW = "EXPIRED_VERIFY_SHADOW"

GLOBAL_VERIFY_PENDING = "GLOBAL_VERIFY_PENDING"
GLOBAL_VERIFY_PASS = "GLOBAL_VERIFY_PASS"
GLOBAL_VERIFY_HOLD = "GLOBAL_VERIFY_HOLD"
GLOBAL_VERIFY_RECOVER = "GLOBAL_VERIFY_RECOVER"
GLOBAL_VERIFY_CONFLICT_PERSISTENT = "GLOBAL_VERIFY_CONFLICT_PERSISTENT"
GLOBAL_VERIFY_EXPIRE = "GLOBAL_VERIFY_EXPIRE"


TrackKeyV1 = tuple[str, str]
PairKeyV1 = tuple[TrackKeyV1, TrackKeyV1]


def _pair_from_event(event: GlobalShadowEventV1) -> PairKeyV1:
    first = (str(event.camera_a), str(event.track_a))
    second = (str(event.camera_b), str(event.track_b))
    return tuple(sorted((first, second)))  # type: ignore[return-value]


def _pct(values: deque[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(round((len(ordered) - 1) * float(quantile)))),
    )
    return float(ordered[index])


@dataclass
class ShadowVerificationRecordV1:
    shadow_global_id: str
    room: str
    pair_key: PairKeyV1
    state: str
    created_at_ns: int
    updated_at_ns: int
    clean_observations: int
    total_observations: int
    conflict_events: int
    conflict_streak: int
    last_conflict_pair: PairKeyV1 | None
    last_score: float | None


@dataclass(frozen=True)
class ShadowVerificationEventV1:
    timestamp_ns: int
    event: str
    shadow_global_id: str
    room: str
    camera_a: str
    track_a: str
    camera_b: str
    track_b: str
    state: str
    clean_observations: int
    total_observations: int
    conflict_events: int
    conflict_streak: int
    robust_score: float | None
    status: str


class GlobalShadowHysteresisV1:
    """Step6-only temporal hysteresis over Step5 shadow identities.

    The verifier consumes Step5 events only.  It never changes Step5 ownership,
    tracker IDs, Room IDs, UI IDs, or production Global IDs.  A confirmed Step5
    identity must survive additional clean observations before becoming
    VERIFIED_SHADOW.  Conflicts immediately move impacted identities into a hold
    state; recovery also requires repeated clean observations.  Persistent
    conflicts are surfaced but intentionally not resolved in this step.

    Geometry is deliberately absent here: calibrated common-world coordinates are
    required before a geometric gate can be trustworthy.  This module therefore
    implements the safe Step6 temporal/conflict layer without inventing camera
    geometry from raw pixels.
    """

    def __init__(
        self,
        *,
        verify_clean_observations: int = 3,
        recover_clean_observations: int = 3,
        persistent_conflict_observations: int = 3,
        max_records: int = 512,
    ) -> None:
        self.verify_clean_observations = max(1, int(verify_clean_observations))
        self.recover_clean_observations = max(1, int(recover_clean_observations))
        self.persistent_conflict_observations = max(
            1, int(persistent_conflict_observations)
        )
        self.max_records = max(8, int(max_records))
        self._records: dict[str, ShadowVerificationRecordV1] = {}
        self.verify_ms: deque[float] = deque(maxlen=4096)

        self.records_created = 0
        self.verified_total = 0
        self.hold_events = 0
        self.recovered_total = 0
        self.persistent_conflicts = 0
        self.expired_total = 0
        self.events_total = 0

    @property
    def records(self) -> tuple[ShadowVerificationRecordV1, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    @staticmethod
    def _event(
        source: GlobalShadowEventV1,
        record: ShadowVerificationRecordV1,
        event: str,
        status: str,
    ) -> ShadowVerificationEventV1:
        (camera_a, track_a), (camera_b, track_b) = record.pair_key
        return ShadowVerificationEventV1(
            timestamp_ns=int(source.timestamp_ns),
            event=event,
            shadow_global_id=record.shadow_global_id,
            room=record.room,
            camera_a=camera_a,
            track_a=track_a,
            camera_b=camera_b,
            track_b=track_b,
            state=record.state,
            clean_observations=int(record.clean_observations),
            total_observations=int(record.total_observations),
            conflict_events=int(record.conflict_events),
            conflict_streak=int(record.conflict_streak),
            robust_score=source.robust_score,
            status=status,
        )

    def _ensure_capacity(self) -> None:
        if len(self._records) < self.max_records:
            return
        expired = sorted(
            (
                record.updated_at_ns,
                shadow_id,
            )
            for shadow_id, record in self._records.items()
            if record.state == EXPIRED_VERIFY_SHADOW
        )
        for _updated_at_ns, shadow_id in expired:
            if len(self._records) < self.max_records:
                break
            self._records.pop(shadow_id, None)
        if len(self._records) >= self.max_records:
            raise RuntimeError("Step6 verification record capacity exhausted")

    def observe_step5_event(
        self, source: GlobalShadowEventV1
    ) -> tuple[ShadowVerificationEventV1, ...]:
        started_ns = time.perf_counter_ns()
        try:
            if source.event == GLOBAL_SHADOW_CONFIRM:
                if source.state != CONFIRMED_SHADOW or not source.shadow_global_id:
                    return ()
                record = self._records.get(source.shadow_global_id)
                if record is None:
                    self._ensure_capacity()
                    record = ShadowVerificationRecordV1(
                        shadow_global_id=source.shadow_global_id,
                        room=str(source.room),
                        pair_key=_pair_from_event(source),
                        state=VERIFY_PENDING,
                        created_at_ns=int(source.timestamp_ns),
                        updated_at_ns=int(source.timestamp_ns),
                        clean_observations=0,
                        total_observations=0,
                        conflict_events=0,
                        conflict_streak=0,
                        last_conflict_pair=None,
                        last_score=source.robust_score,
                    )
                    self._records[record.shadow_global_id] = record
                    self.records_created += 1
                record.updated_at_ns = int(source.timestamp_ns)
                record.last_score = source.robust_score
                event = self._event(source, record, GLOBAL_VERIFY_PENDING, VERIFY_PENDING)
                self.events_total += 1
                return (event,)

            if source.event == GLOBAL_SHADOW_OBSERVE:
                if source.state != CONFIRMED_SHADOW or not source.shadow_global_id:
                    return ()
                record = self._records.get(source.shadow_global_id)
                if record is None or record.state == EXPIRED_VERIFY_SHADOW:
                    return ()
                record.updated_at_ns = int(source.timestamp_ns)
                record.total_observations += 1
                record.clean_observations += 1
                record.last_score = source.robust_score
                record.conflict_streak = 0
                record.last_conflict_pair = None
                output: list[ShadowVerificationEventV1] = []
                if (
                    record.state == VERIFY_PENDING
                    and record.clean_observations >= self.verify_clean_observations
                ):
                    record.state = VERIFIED_SHADOW
                    self.verified_total += 1
                    output.append(
                        self._event(source, record, GLOBAL_VERIFY_PASS, VERIFIED_SHADOW)
                    )
                elif (
                    record.state == CONFLICT_HOLD_SHADOW
                    and record.clean_observations >= self.recover_clean_observations
                ):
                    record.state = VERIFIED_SHADOW
                    self.recovered_total += 1
                    output.append(
                        self._event(
                            source, record, GLOBAL_VERIFY_RECOVER, VERIFIED_SHADOW
                        )
                    )
                self.events_total += len(output)
                return tuple(output)

            if source.event == GLOBAL_SHADOW_CONFLICT:
                conflict_pair = _pair_from_event(source)
                impacted = [
                    record
                    for record in self._records.values()
                    if record.state != EXPIRED_VERIFY_SHADOW
                    and any(member in record.pair_key for member in conflict_pair)
                ]
                output: list[ShadowVerificationEventV1] = []
                for record in sorted(impacted, key=lambda item: item.shadow_global_id):
                    same_conflict = record.last_conflict_pair == conflict_pair
                    record.conflict_streak = record.conflict_streak + 1 if same_conflict else 1
                    record.last_conflict_pair = conflict_pair
                    record.conflict_events += 1
                    record.clean_observations = 0
                    record.updated_at_ns = int(source.timestamp_ns)
                    record.state = CONFLICT_HOLD_SHADOW
                    self.hold_events += 1
                    output.append(
                        self._event(
                            source,
                            record,
                            GLOBAL_VERIFY_HOLD,
                            CONFLICT_HOLD_SHADOW,
                        )
                    )
                    if record.conflict_streak == self.persistent_conflict_observations:
                        self.persistent_conflicts += 1
                        output.append(
                            self._event(
                                source,
                                record,
                                GLOBAL_VERIFY_CONFLICT_PERSISTENT,
                                CONFLICT_HOLD_SHADOW,
                            )
                        )
                self.events_total += len(output)
                return tuple(output)

            if source.event == GLOBAL_SHADOW_EXPIRE:
                if not source.shadow_global_id:
                    return ()
                record = self._records.get(source.shadow_global_id)
                if record is None:
                    return ()
                record.state = EXPIRED_VERIFY_SHADOW
                record.updated_at_ns = int(source.timestamp_ns)
                self.expired_total += 1
                event = self._event(
                    source, record, GLOBAL_VERIFY_EXPIRE, EXPIRED_VERIFY_SHADOW
                )
                self.events_total += 1
                return (event,)

            return ()
        finally:
            self.verify_ms.append((time.perf_counter_ns() - started_ns) / 1_000_000.0)

    def snapshot(self) -> dict[str, int | float]:
        pending = sum(1 for item in self._records.values() if item.state == VERIFY_PENDING)
        verified = sum(
            1 for item in self._records.values() if item.state == VERIFIED_SHADOW
        )
        hold = sum(
            1 for item in self._records.values() if item.state == CONFLICT_HOLD_SHADOW
        )
        expired = sum(
            1 for item in self._records.values() if item.state == EXPIRED_VERIFY_SHADOW
        )
        return {
            "verify_records_created": self.records_created,
            "verify_pending": pending,
            "verify_verified": verified,
            "verify_hold": hold,
            "verify_expired": expired,
            "verify_verified_total": self.verified_total,
            "verify_hold_events": self.hold_events,
            "verify_recovered_total": self.recovered_total,
            "verify_persistent_conflicts": self.persistent_conflicts,
            "verify_events_total": self.events_total,
            "verify_p50_ms": _pct(self.verify_ms, 0.50),
            "verify_p95_ms": _pct(self.verify_ms, 0.95),
        }
