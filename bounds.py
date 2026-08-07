"""Single source of truth for config bounds.

Hard ceilings enforced by load_settings. Schema sliders may be tighter for UX;
code never accepts values above these ceilings.
"""

from __future__ import annotations

# Keep in sync with _conf_schema.json slider max for user-facing fields.
MAX_MIN_LENGTH = 200
MAX_SEGMENTS = 20
MAX_TEMPERATURE = 2.0
MAX_TOKENS = 4096
MAX_TIMEOUT_SECONDS = 60.0
MAX_DELAY_BASE = 5.0
MAX_DELAY_MAX = 10.0
MAX_DELAY_PER_CHAR = 0.2
MAX_STREAMING_MIN_CHARS = 80
MAX_STREAMING_MAX_CHARS = 300

PENDING_FOLLOW_UP_TTL_SECONDS = 60.0

ALLOWED_STYLES = frozenset({"natural", "conservative", "active"})
