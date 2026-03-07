"""
SCRIPT NAME: rbac.py
====================================
Execution Location: market-hawk-mvp/config/
Purpose: Role-Based Access Control for Market Hawk dashboard and trading.
Creation Date: 2026-03-07

Roles:
    VIEWER  — Dashboard read-only (charts, portfolio view, logs)
    TRADER  — Can execute paper/live trades + everything VIEWER can do
    ADMIN   — Full access: config changes, key rotation, system management

The active role is read from MH_USER_ROLE env var (default: VIEWER).

Usage:
    from config.rbac import Permission, check_permission, requires_permission

    if check_permission(Permission.TRADE):
        execute_trade(...)

    @requires_permission(Permission.TRADE)
    def place_order(...):
        ...
"""

import functools
import logging
import os
from enum import Enum, unique
from typing import Callable, TypeVar, Any

logger = logging.getLogger("market_hawk.rbac")

F = TypeVar("F", bound=Callable[..., Any])


# ============================================================
# ENUMS
# ============================================================

@unique
class UserRole(Enum):
    """User access levels, ordered by privilege."""
    VIEWER = "VIEWER"
    TRADER = "TRADER"
    ADMIN = "ADMIN"


@unique
class Permission(Enum):
    """Granular permissions mapped to roles."""
    VIEW_DASHBOARD = "VIEW_DASHBOARD"
    VIEW_PORTFOLIO = "VIEW_PORTFOLIO"
    VIEW_LOGS = "VIEW_LOGS"
    RUN_SCAN = "RUN_SCAN"
    TRADE = "TRADE"
    MANAGE_CONFIG = "MANAGE_CONFIG"
    ROTATE_KEYS = "ROTATE_KEYS"
    MANAGE_USERS = "MANAGE_USERS"


# Mapare roluri -> permisiuni
_ROLE_PERMISSIONS: dict[UserRole, set[Permission]] = {
    UserRole.VIEWER: {
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_PORTFOLIO,
        Permission.VIEW_LOGS,
    },
    UserRole.TRADER: {
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_PORTFOLIO,
        Permission.VIEW_LOGS,
        Permission.RUN_SCAN,
        Permission.TRADE,
    },
    UserRole.ADMIN: set(Permission),  # Toate permisiunile
}


# ============================================================
# ROLE RESOLUTION
# ============================================================

def get_current_role() -> UserRole:
    """Read the active user role from MH_USER_ROLE env var.

    Returns:
        UserRole enum value. Defaults to VIEWER if not set or invalid.
    """
    raw = os.environ.get("MH_USER_ROLE", "VIEWER").upper().strip()
    try:
        return UserRole(raw)
    except ValueError:
        logger.warning("Invalid MH_USER_ROLE=%r, defaulting to VIEWER", raw)
        return UserRole.VIEWER


def get_permissions(role: UserRole) -> set[Permission]:
    """Return the set of permissions granted to a role."""
    return _ROLE_PERMISSIONS.get(role, set())


# ============================================================
# PERMISSION CHECKS
# ============================================================

def check_permission(required: Permission,
                     role: UserRole | None = None) -> bool:
    """Check if a role has a specific permission.

    Args:
        required: The permission to check.
        role: User role to check. If None, reads from env var.

    Returns:
        True if the role has the required permission.
    """
    if role is None:
        role = get_current_role()
    return required in get_permissions(role)


def requires_permission(required: Permission) -> Callable[[F], F]:
    """Decorator that gates a function behind a permission check.

    Raises PermissionError if the current role lacks the required permission.

    Usage:
        @requires_permission(Permission.TRADE)
        def place_order(symbol, qty):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            role = get_current_role()
            if not check_permission(required, role):
                raise PermissionError(
                    f"Role {role.value} lacks permission {required.value} "
                    f"for {func.__qualname__}"
                )
            return func(*args, **kwargs)
        return wrapper  # type: ignore[return-value]
    return decorator


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s")

    for role in UserRole:
        perms = get_permissions(role)
        perm_names = sorted(p.value for p in perms)
        print(f"{role.value:8s} -> {', '.join(perm_names)}")

    print(f"\nCurrent role: {get_current_role().value}")
    print(f"Can trade: {check_permission(Permission.TRADE)}")
    print(f"Can view dashboard: {check_permission(Permission.VIEW_DASHBOARD)}")
    print(f"Can manage config: {check_permission(Permission.MANAGE_CONFIG)}")
