from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from threaded_earth.models import Event, Household, Resource


FOOD_TYPES = ("grain", "fish")
MATERIAL_TYPES = ("reeds", "tools")
SCARCITY_FOOD_THRESHOLD = 18.0


def household_food(household: Household) -> float:
    return round(sum(float(household.stored_resources.get(resource, 0.0)) for resource in FOOD_TYPES), 2)


def household_materials(household: Household) -> float:
    return round(sum(float(household.stored_resources.get(resource, 0.0)) for resource in MATERIAL_TYPES), 2)


def add_household_resource(
    session: Session,
    run_id: str,
    household_id: str,
    resource_type: str,
    amount: float,
) -> float:
    resource = _resource_row(session, run_id, household_id, resource_type)
    before = resource.quantity
    resource.quantity = max(0.0, round(resource.quantity + amount, 2))
    _sync_household_resource(session, household_id, resource_type, resource.quantity)
    return round(resource.quantity - before, 2)


def consume_household_food(
    session: Session,
    run_id: str,
    household_id: str,
    requested_quantity: float,
) -> dict[str, Any]:
    remaining = round(max(0.0, requested_quantity), 2)
    consumed_by_type: dict[str, float] = {}
    total_consumed = 0.0
    for resource_type in FOOD_TYPES:
        if remaining <= 0:
            break
        resource = _resource_row(session, run_id, household_id, resource_type)
        consumed = round(min(resource.quantity, remaining), 2)
        resource.quantity = round(resource.quantity - consumed, 2)
        _sync_household_resource(session, household_id, resource_type, resource.quantity)
        consumed_by_type[resource_type] = consumed
        total_consumed = round(total_consumed + consumed, 2)
        remaining = round(remaining - consumed, 2)
    return {
        "requested_food": round(requested_quantity, 2),
        "consumed_food": total_consumed,
        "shortage_amount": round(max(0.0, requested_quantity - total_consumed), 2),
        "consumed_by_type": consumed_by_type,
        "status": "shortage" if total_consumed < requested_quantity else "met",
    }


def transfer_household_resource(
    session: Session,
    run_id: str,
    source_household_id: str,
    target_household_id: str | None,
    resource_type: str,
    requested_quantity: float,
) -> dict[str, Any]:
    if target_household_id is None:
        return {
            "resource_type": resource_type,
            "requested_quantity": requested_quantity,
            "transferred_quantity": 0.0,
            "source_household_id": source_household_id,
            "target_household_id": None,
            "status": "failed",
            "reason": "no target household",
        }
    source = _resource_row(session, run_id, source_household_id, resource_type)
    target = _resource_row(session, run_id, target_household_id, resource_type)
    transferred = round(min(max(0.0, requested_quantity), source.quantity), 2)
    status = "success" if transferred == requested_quantity else ("limited" if transferred > 0 else "failed")
    source.quantity = round(source.quantity - transferred, 2)
    target.quantity = round(target.quantity + transferred, 2)
    _sync_household_resource(session, source_household_id, resource_type, source.quantity)
    _sync_household_resource(session, target_household_id, resource_type, target.quantity)
    return {
        "resource_type": resource_type,
        "requested_quantity": round(requested_quantity, 2),
        "transferred_quantity": transferred,
        "source_household_id": source_household_id,
        "target_household_id": target_household_id,
        "status": status,
        "reason": "available transfer" if status == "success" else "insufficient source resource",
    }


def household_resource_summary(session: Session, run_id: str) -> dict[str, Any]:
    households = session.query(Household).filter(Household.run_id == run_id).order_by(Household.household_id).all()
    total_food = sum(household_food(household) for household in households)
    total_materials = sum(household_materials(household) for household in households)
    scarce = [household.household_id for household in households if household_food(household) < SCARCITY_FOOD_THRESHOLD]
    return {
        "total_food": round(total_food, 2),
        "total_materials": round(total_materials, 2),
        "average_food_per_household": round(total_food / len(households), 2) if households else 0.0,
        "households_below_scarcity_threshold": len(scarce),
        "scarce_household_ids": scarce,
        "households": [
            {
                "household_id": household.household_id,
                "household_name": household.household_name,
                "member_count": len(household.members),
                "food": household_food(household),
                "materials": household_materials(household),
                "food_per_member": round(household_food(household) / len(household.members), 2) if household.members else 0.0,
                "stored_resources": dict(household.stored_resources),
            }
            for household in households
        ],
    }


def transfers_for_tick(session: Session, run_id: str, tick: int) -> list[dict[str, Any]]:
    events = (
        session.query(Event)
        .filter(Event.run_id == run_id, Event.tick == tick, Event.event_type.in_(("resource_exchange", "cooperation", "conflict")))
        .order_by(Event.event_id)
        .all()
    )
    transfers = []
    for event in events:
        transfer = event.payload.get("resource_transfer")
        if transfer:
            transfers.append({"event_id": event.event_id, "event_type": event.event_type, **transfer})
    return transfers


def upkeep_stats_for_tick(session: Session, run_id: str, tick: int) -> dict[str, Any]:
    events = (
        session.query(Event)
        .filter(Event.run_id == run_id, Event.tick == tick, Event.event_type.in_(("household_upkeep", "household_shortage")))
        .order_by(Event.event_id)
        .all()
    )
    consumed = 0.0
    shortage = 0.0
    shortage_households: set[str] = set()
    for event in events:
        consumed += float(event.payload.get("food_consumed", 0.0))
        shortage_amount = float(event.payload.get("shortage_amount", 0.0))
        shortage += shortage_amount
        if shortage_amount > 0 and event.target:
            shortage_households.add(event.target)
    return {
        "food_consumed_this_tick": round(consumed, 2),
        "households_with_shortage": len(shortage_households),
        "shortage_household_ids": sorted(shortage_households),
        "total_shortage_amount": round(shortage, 2),
    }


def recent_transfer_events(session: Session, run_id: str, limit: int = 10) -> list[Event]:
    return (
        session.query(Event)
        .filter(Event.run_id == run_id)
        .order_by(Event.tick.desc(), Event.event_id.desc())
        .all()
    )[:limit]


def _resource_row(session: Session, run_id: str, household_id: str, resource_type: str) -> Resource:
    resource = (
        session.query(Resource)
        .filter(
            Resource.run_id == run_id,
            Resource.owner_scope == "household",
            Resource.owner_id == household_id,
            Resource.resource_type == resource_type,
        )
        .one_or_none()
    )
    if resource is None:
        resource = Resource(
            run_id=run_id,
            resource_type=resource_type,
            owner_scope="household",
            owner_id=household_id,
            quantity=0.0,
        )
        session.add(resource)
        session.flush()
    return resource


def _sync_household_resource(session: Session, household_id: str, resource_type: str, quantity: float) -> None:
    household = session.get(Household, household_id)
    if household is None:
        return
    stored = dict(household.stored_resources)
    stored[resource_type] = round(quantity, 2)
    household.stored_resources = stored
