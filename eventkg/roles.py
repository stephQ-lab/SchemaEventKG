# -*- coding: utf-8 -*-
"""Optional participant roles. Missing roles stay missing; nothing is invented."""
from eventkg.ontology import POLICY_MODALITY, ROLE_SCHEMAS, MAINTENANCE_STATUS, role_binding

KIND_ALIASES = {"装置检修": "MAINTENANCE", "政策调整": "POLICY_CHANGE", "物流中断": "LOGISTICS_DISRUPTION"}
FIELD_MAP = {
    "MAINTENANCE": {"agent": "subject", "product": "commodity", "location": "scope", "start_time": "event_time"},
    "POLICY_CHANGE": {"publisher": "subject", "affected_region": "scope", "effective_time": "event_time"},
    "LOGISTICS_DISRUPTION": {"start_time": "event_time"},
    "PRODUCTION_CHANGE": {"product": "commodity", "location": "scope", "start_time": "event_time"},
}
CLOSED = {"status": MAINTENANCE_STATUS, "modality": POLICY_MODALITY}


def event_kind(event):
    given = event.get("event_kind") or ""
    if given in ROLE_SCHEMAS:
        return given
    if given in KIND_ALIASES:
        return KIND_ALIASES[given]
    blob = " ".join(str(event.get(key) or "") for key in ("name", "indicator", "subject", "evidence", "var"))
    if any(word in blob for word in ("政策", "关税")) or ("政府" in blob and "限产" in blob):
        return "POLICY_CHANGE"
    if any(word in blob for word in ("检修", "停车", "停机")):
        return "MAINTENANCE"
    if any(word in blob for word in ("停产", "限产")):
        return "PRODUCTION_CHANGE"
    if any(word in blob for word in ("封航", "停运", "中断", "受阻")):
        return "LOGISTICS_DISRUPTION"
    if any(word in blob for word in ("发运", "到港", "疏港")):
        return "LOGISTICS_CHANGE"
    if any(word in blob for word in ("产量", "开工", "产能")):
        return "PRODUCTION_CHANGE"
    return ""


def assign_roles(event):
    """Copy a role only when the event already has that value. Do not call a model.

    A commodity is not an affected company, and a rise in arrivals is not a disruption.
    """
    kind = event_kind(event)
    schema = ROLE_SCHEMAS.get(kind) or ()
    given = dict(event.get("roles") or {})
    roles = {}
    for key in schema:
        value = given.get(key)
        if value in (None, ""):
            value = event.get(FIELD_MAP.get(kind, {}).get(key, ""))
        if value in (None, ""):
            continue
        allowed = CLOSED.get(key)
        if allowed and value not in allowed:
            continue
        roles[key] = value
    event["event_kind"] = kind
    event["roles"] = roles
    event["missing_roles"] = [key for key in schema if key not in roles]
    event["role_bindings"] = [role_binding(value, key) for key, value in roles.items()]
    return event
