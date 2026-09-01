from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass


PROVISIONAL = "PROVISIONAL"
CONFIRMED_SHADOW = "CONFIRMED_SHADOW"
EXPIRED_SHADOW = "EXPIRED_SHADOW"
CONFLICT_PENDING = "CONFLICT_PENDING"

GLOBAL_SHADOW_CREATE = "GLOBAL_SHADOW_CREATE"
GLOBAL_SHADOW_OBSERVE = "GLOBAL_SHADOW_OBSERVE"
GLOBAL_SHADOW_CONFIRM = "GLOBAL_SHADOW_CONFIRM"
GLOBAL_SHADOW_CONFLICT = "GLOBAL_SHADOW_CONFLICT"
GLOBAL_SHADOW_EXPIRE = "GLOBAL_SHADOW_EXPIRE"


TrackKeyV1 = tuple[str, str]
PairKeyV1 = tuple[TrackKeyV1, TrackKeyV1]


def canonical_pair_v1(
    camera_a: str, track_a: str, camera_b: str, track_b: str
) -> PairKeyV1:
    first = (str(camera_a), str(track_a))
    second = (str(camera_b), str(track_b))
    if first == second:
        raise ValueError("shadow global pair endpoints must be distinct")
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
class ShadowGlobalIdentityV1:
    shadow_global_id: str
    room: str
    pair_key: PairKeyV1
    created_at_ns: int
    updated_at_ns: int
    state: str
    proposal_count: int
    consecutive_count: int
    missed_cycles: int
    first_score: float | None
    last_score: float | None
    best_score: float | None
    last_observed_cycle: int

    @property
    def members(self) -> PairKeyV1:
        return self.pair_key


@dataclass(frozen=True)
class GlobalShadowEventV1:
    timestamp_ns: int
    event: str
    shadow_global_id: str
    room: str
    camera_a: str
    track_a: str
    camera_b: str
    track_b: str
    proposal_count: int
    consecutive_count: int
    state: str
    robust_score: float | None
    status: str


class GlobalShadowStateMachineV1:
    """Conservative Step5 shadow lifecycle over Step4 match proposals.

    A shadow identity represents a physical-person hypothesis, not an immutable
    pair of local tracker IDs. Step4/4.5 may fragment a person's local track, so
    Step5 lets a new local member replace the current member for that camera when
    the other endpoint already anchors the proposal to one existing shadow ID.

    Safety stays strict: two different owners conflict; a same-cycle unknown
    same-camera replacement conflicts; historical aliases cannot immediately
    steal current ownership back; Step5 never merges two global identities.
    """

    def __init__(
        self,
        *,
        confirm_observations: int = 3,
        confirm_consecutive: int = 3,
        expire_provisional_after_missed_cycles: int = 6,
        max_records: int = 512,
    ) -> None:
        self.confirm_observations = max(1, int(confirm_observations))
        self.confirm_consecutive = max(1, int(confirm_consecutive))
        self.expire_provisional_after_missed_cycles = max(
            1, int(expire_provisional_after_missed_cycles)
        )
        self.max_records = max(8, int(max_records))
        self._next_id = 1
        self._records: dict[str, ShadowGlobalIdentityV1] = {}
        self._member_to_active_id: dict[TrackKeyV1, str] = {}
        self._alias_to_active_id: dict[TrackKeyV1, str] = {}
        self._aliases_by_active_id: dict[str, set[TrackKeyV1]] = {}
        self._member_last_seen_cycle: dict[TrackKeyV1, int] = {}
        self._cycle = 0
        self._seen_shadow_ids_this_cycle: set[str] = set()
        self._seen_pairs_this_cycle: set[PairKeyV1] = set()
        self.global_shadow_created = 0
        self.global_shadow_observations = 0
        self.global_shadow_confirmed_total = 0
        self.global_shadow_conflicts = 0
        self.global_shadow_expired = 0
        self.state_ms: deque[float] = deque(maxlen=4096)

    @property
    def records(self) -> tuple[ShadowGlobalIdentityV1, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def get(self, shadow_global_id: str) -> ShadowGlobalIdentityV1 | None:
        return self._records.get(str(shadow_global_id))

    def begin_cycle(self, cycle: int) -> None:
        cycle = int(cycle)
        if cycle <= 0:
            raise ValueError("cycle must be positive")
        if self._cycle and cycle <= self._cycle:
            raise ValueError("cycles must be strictly increasing")
        self._cycle = cycle
        self._seen_shadow_ids_this_cycle = set()
        self._seen_pairs_this_cycle = set()

    def _allocate_id(self) -> str:
        self._prune_expired_history()
        if len(self._records) >= self.max_records:
            raise RuntimeError("shadow global record capacity exhausted")
        value = f"GSH-{self._next_id:06d}"
        self._next_id += 1
        return value

    def _prune_expired_history(self) -> None:
        if len(self._records) < self.max_records:
            return
        expired = sorted(
            (record.updated_at_ns, shadow_id)
            for shadow_id, record in self._records.items()
            if record.state == EXPIRED_SHADOW
        )
        for _updated_at_ns, shadow_id in expired:
            if len(self._records) < self.max_records:
                break
            self._records.pop(shadow_id, None)

    @staticmethod
    def _score(value: float | None) -> float | None:
        if value is None:
            return None
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("robust_score must be finite or None")
        return numeric

    @staticmethod
    def _event(
        timestamp_ns: int,
        event: str,
        record: ShadowGlobalIdentityV1 | None,
        pair_key: PairKeyV1,
        room: str,
        robust_score: float | None,
        status: str,
    ) -> GlobalShadowEventV1:
        (camera_a, track_a), (camera_b, track_b) = pair_key
        return GlobalShadowEventV1(
            timestamp_ns=int(timestamp_ns),
            event=event,
            shadow_global_id="" if record is None else record.shadow_global_id,
            room=str(room),
            camera_a=camera_a,
            track_a=track_a,
            camera_b=camera_b,
            track_b=track_b,
            proposal_count=0 if record is None else int(record.proposal_count),
            consecutive_count=0 if record is None else int(record.consecutive_count),
            state=CONFLICT_PENDING if record is None else record.state,
            robust_score=robust_score,
            status=status,
        )

    @staticmethod
    def _current_member_for_camera(
        record: ShadowGlobalIdentityV1, camera: str
    ) -> TrackKeyV1 | None:
        camera = str(camera)
        for member in record.pair_key:
            if member[0] == camera:
                return member
        return None

    def _remember_alias(self, shadow_id: str, member: TrackKeyV1) -> None:
        owner = self._alias_to_active_id.get(member)
        if owner is not None and owner != shadow_id:
            raise RuntimeError("local track alias already owned by another shadow identity")
        self._alias_to_active_id[member] = shadow_id
        self._aliases_by_active_id.setdefault(shadow_id, set()).add(member)

    def _replace_current_member(
        self, record: ShadowGlobalIdentityV1, member: TrackKeyV1
    ) -> None:
        shadow_id = record.shadow_global_id
        camera = member[0]
        current = self._current_member_for_camera(record, camera)
        if current is None:
            raise RuntimeError("Step5 V1 cannot expand a two-camera identity to a new camera")
        if current == member:
            self._remember_alias(shadow_id, member)
            self._member_to_active_id[member] = shadow_id
            return
        if self._member_to_active_id.get(current) == shadow_id:
            self._member_to_active_id.pop(current, None)
        updated = [member if item[0] == camera else item for item in record.pair_key]
        record.pair_key = tuple(sorted(updated))  # type: ignore[assignment]
        self._remember_alias(shadow_id, member)
        self._member_to_active_id[member] = shadow_id

    def _create_record(
        self,
        *,
        cycle: int,
        timestamp_ns: int,
        room: str,
        pair_key: PairKeyV1,
        score: float | None,
    ) -> ShadowGlobalIdentityV1:
        shadow_id = self._allocate_id()
        record = ShadowGlobalIdentityV1(
            shadow_global_id=shadow_id,
            room=room,
            pair_key=pair_key,
            created_at_ns=timestamp_ns,
            updated_at_ns=timestamp_ns,
            state=PROVISIONAL,
            proposal_count=1,
            consecutive_count=1,
            missed_cycles=0,
            first_score=score,
            last_score=score,
            best_score=score,
            last_observed_cycle=int(cycle),
        )
        self._records[shadow_id] = record
        for member in pair_key:
            self._remember_alias(shadow_id, member)
            self._member_to_active_id[member] = shadow_id
            self._member_last_seen_cycle[member] = int(cycle)
        self._seen_shadow_ids_this_cycle.add(shadow_id)
        self._seen_pairs_this_cycle.add(pair_key)
        self.global_shadow_created += 1
        self.global_shadow_observations += 1
        return record

    def _expire_record(self, record: ShadowGlobalIdentityV1) -> None:
        shadow_id = record.shadow_global_id
        for member in record.pair_key:
            if self._member_to_active_id.get(member) == shadow_id:
                self._member_to_active_id.pop(member, None)
        for alias in self._aliases_by_active_id.pop(shadow_id, set()):
            if self._alias_to_active_id.get(alias) == shadow_id:
                self._alias_to_active_id.pop(alias, None)
            self._member_last_seen_cycle.pop(alias, None)

    def _conflict(
        self,
        *,
        timestamp_ns: int,
        pair_key: PairKeyV1,
        room: str,
        score: float | None,
    ) -> tuple[GlobalShadowEventV1, ...]:
        self.global_shadow_conflicts += 1
        return (
            self._event(
                timestamp_ns,
                GLOBAL_SHADOW_CONFLICT,
                None,
                pair_key,
                room,
                score,
                CONFLICT_PENDING,
            ),
        )

    def observe_proposal(
        self,
        *,
        cycle: int,
        timestamp_ns: int,
        room: str,
        camera_a: str,
        track_a: str,
        camera_b: str,
        track_b: str,
        robust_score: float | None,
        reciprocal: bool,
        assigned: bool,
        status: str,
    ) -> tuple[GlobalShadowEventV1, ...]:
        started_ns = time.perf_counter_ns()
        try:
            if int(cycle) != self._cycle:
                raise ValueError("proposal cycle does not match active cycle")
            if status != "MATCH_PROPOSED" or not reciprocal or not assigned:
                return ()
            score = self._score(robust_score)
            raw_pair = canonical_pair_v1(camera_a, track_a, camera_b, track_b)
            timestamp_ns = int(timestamp_ns)
            room = str(room)
            owners = {
                self._alias_to_active_id[member]
                for member in raw_pair
                if member in self._alias_to_active_id
            }
            if len(owners) > 1:
                return self._conflict(
                    timestamp_ns=timestamp_ns, pair_key=raw_pair, room=room, score=score
                )
            if not owners:
                if raw_pair in self._seen_pairs_this_cycle:
                    return ()
                record = self._create_record(
                    cycle=int(cycle),
                    timestamp_ns=timestamp_ns,
                    room=room,
                    pair_key=raw_pair,
                    score=score,
                )
                return (
                    self._event(
                        timestamp_ns,
                        GLOBAL_SHADOW_CREATE,
                        record,
                        record.pair_key,
                        room,
                        score,
                        PROVISIONAL,
                    ),
                )

            shadow_id = next(iter(owners))
            record = self._records.get(shadow_id)
            if record is None or record.state == EXPIRED_SHADOW:
                raise RuntimeError("active local-track alias points to missing/expired shadow")
            if record.room != room:
                return self._conflict(
                    timestamp_ns=timestamp_ns, pair_key=raw_pair, room=room, score=score
                )

            replacements: list[TrackKeyV1] = []
            for member in raw_pair:
                camera = member[0]
                current = self._current_member_for_camera(record, camera)
                if current is None:
                    return self._conflict(
                        timestamp_ns=timestamp_ns, pair_key=raw_pair, room=room, score=score
                    )
                if member == current:
                    continue
                member_owner = self._alias_to_active_id.get(member)
                if member_owner == shadow_id:
                    current_last_seen = self._member_last_seen_cycle.get(current, -1)
                    if current_last_seen >= int(cycle) - 1:
                        return ()
                    replacements.append(member)
                    continue
                if member_owner is not None:
                    return self._conflict(
                        timestamp_ns=timestamp_ns, pair_key=raw_pair, room=room, score=score
                    )
                if self._member_last_seen_cycle.get(current) == int(cycle):
                    return self._conflict(
                        timestamp_ns=timestamp_ns, pair_key=raw_pair, room=room, score=score
                    )
                replacements.append(member)

            for member in replacements:
                self._replace_current_member(record, member)
            effective_pair = record.pair_key
            if effective_pair in self._seen_pairs_this_cycle:
                return ()
            self._seen_pairs_this_cycle.add(effective_pair)
            self._seen_shadow_ids_this_cycle.add(shadow_id)
            for member in effective_pair:
                self._member_last_seen_cycle[member] = int(cycle)

            record.proposal_count += 1
            record.consecutive_count = (
                record.consecutive_count + 1
                if record.last_observed_cycle == int(cycle) - 1
                else 1
            )
            record.last_observed_cycle = int(cycle)
            record.updated_at_ns = timestamp_ns
            record.missed_cycles = 0
            record.last_score = score
            if score is not None and (
                record.best_score is None or score > record.best_score
            ):
                record.best_score = score
            self.global_shadow_observations += 1

            events = [
                self._event(
                    timestamp_ns,
                    GLOBAL_SHADOW_OBSERVE,
                    record,
                    effective_pair,
                    room,
                    score,
                    record.state,
                )
            ]
            if (
                record.state == PROVISIONAL
                and record.proposal_count >= self.confirm_observations
                and record.consecutive_count >= self.confirm_consecutive
            ):
                record.state = CONFIRMED_SHADOW
                record.updated_at_ns = timestamp_ns
                self.global_shadow_confirmed_total += 1
                events.append(
                    self._event(
                        timestamp_ns,
                        GLOBAL_SHADOW_CONFIRM,
                        record,
                        effective_pair,
                        room,
                        score,
                        CONFIRMED_SHADOW,
                    )
                )
            return tuple(events)
        finally:
            self.state_ms.append((time.perf_counter_ns() - started_ns) / 1_000_000.0)

    def end_cycle(
        self, *, cycle: int, timestamp_ns: int
    ) -> tuple[GlobalShadowEventV1, ...]:
        started_ns = time.perf_counter_ns()
        try:
            if int(cycle) != self._cycle:
                raise ValueError("end cycle does not match active cycle")
            timestamp_ns = int(timestamp_ns)
            events: list[GlobalShadowEventV1] = []
            for shadow_id in sorted(tuple(self._records)):
                record = self._records[shadow_id]
                if record.state == EXPIRED_SHADOW:
                    continue
                if shadow_id in self._seen_shadow_ids_this_cycle:
                    continue
                record.missed_cycles += 1
                record.consecutive_count = 0
                record.updated_at_ns = timestamp_ns
                if (
                    record.state == PROVISIONAL
                    and record.missed_cycles
                    >= self.expire_provisional_after_missed_cycles
                ):
                    record.state = EXPIRED_SHADOW
                    self._expire_record(record)
                    self.global_shadow_expired += 1
                    events.append(
                        self._event(
                            timestamp_ns,
                            GLOBAL_SHADOW_EXPIRE,
                            record,
                            record.pair_key,
                            record.room,
                            record.last_score,
                            EXPIRED_SHADOW,
                        )
                    )
            return tuple(events)
        finally:
            self.state_ms.append((time.perf_counter_ns() - started_ns) / 1_000_000.0)

    def snapshot(self) -> dict[str, int | float]:
        provisional = sum(
            1 for record in self._records.values() if record.state == PROVISIONAL
        )
        confirmed = sum(
            1
            for record in self._records.values()
            if record.state == CONFIRMED_SHADOW
        )
        return {
            "global_shadow_created": self.global_shadow_created,
            "global_shadow_provisional": provisional,
            "global_shadow_confirmed": confirmed,
            "global_shadow_observations": self.global_shadow_observations,
            "global_shadow_conflicts": self.global_shadow_conflicts,
            "global_shadow_expired": self.global_shadow_expired,
            "global_shadow_active": provisional + confirmed,
            "global_shadow_member_tracks": len(self._member_to_active_id),
            "state_p50_ms": _pct(self.state_ms, 0.50),
            "state_p95_ms": _pct(self.state_ms, 0.95),
        }
