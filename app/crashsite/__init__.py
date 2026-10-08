"""Crash site: workflow execution snapshots and breakpoint recovery (M1).

See bigplan ``.codebuddy/memory/2026-10-07-crashsite-bigplan.md`` and
docs/crashsite.md.  The provider captures node outputs incrementally as they
enter the host output cache; resuming a crashed/failed/finished prompt just
re-queues it and the host skips every node whose cache key (input signature
SHA256) is found in the snapshot database.
"""
