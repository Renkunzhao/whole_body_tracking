# Trampoline Structure

This file describes the overall implementation structure for trampoline-related work in `whole_body_tracking`.

## How To Use

- Read this file first when you need the architecture-level picture.
- Read [TODOs.md](TODOs.md) for active backlog, recent progress, and incremental work log.
- Read `task-*.md` files for one concrete implementation pass, phase, or focused design thread.

## Intended Contents

### Overall architecture
- Which trampoline approaches exist in the repo
- How they relate to training, play, and standalone smoke scripts

### Core modules and entrypoints
- Main runtime scripts
- Main task configs
- Shared utilities and action terms

### Interfaces and invariants
- Shared config surfaces
- Expected data flow between task config, env logic, and utilities
- Important assumptions that should remain stable across implementations

### Current canonical approach
- Which path is currently preferred for deformable trampoline
- Which path is currently preferred for custom contact model
- Which paths are experimental or legacy

### Validation map
- Which scripts or tasks are used for smoke testing
- Which tasks represent end-to-end validation

## Notes

- Keep this file stable and architecture-oriented.
- Move detailed one-off implementation decisions into `task-*.md`.
- Move active next steps and dated progress into [TODOs.md](TODOs.md).
