---
name: trampoline
description: Use for all trampoline-related work in whole_body_tracking. Read Structure.md for the overall implementation shape, TODOs.md for current incremental status, and then only the most relevant task document in the same directory for concrete trampoline work such as deformable integration, ball-drop validation, custom contact models, and play or train integration.
---

# Trampoline

## Use This Skill When

- The request is about any trampoline-related task in `whole_body_tracking`.
- The request mentions deformable trampoline integration, domain randomization, custom contact models, spring-damper terrain dynamics, ball-drop validation, or play/train integration.
- The user wants an architecture summary, current status, implementation plan, code change, or a consistency check between docs and code.

## Startup Workflow

1. Read [Structure.md](Structure.md) first to understand the current overall implementation shape.
2. Read [TODOs.md](TODOs.md) when the request depends on current status, recent changes, open items, or next steps.
3. Read only the most relevant `task*.md` file(s) from the routing table below for concrete implementation details. Default to 1-2 files, not the whole directory.
4. If the selected doc mentions concrete implementation files, inspect those code files next. Do not stop at the document layer.
5. If the request spans multiple phases, read multiple task docs, but keep the read set minimal and explicit.

## Document Routing

- Read [Structure.md](Structure.md) for the overall architecture, major modules, interfaces, and how the trampoline work fits into the repo.
- Read [TODOs.md](TODOs.md) for active backlog, recent progress, open questions, and incremental work log.
- Read [task1-DeformableObject.md](task1-DeformableObject.md) for deformable trampoline integration, G1 tracking trampoline, `DeformableObject`, and trampoline material or mass randomization.
- Read [task4-custom-contact-model.md](task4-custom-contact-model.md) for custom contact model design, spring-damper terrain dynamics replacement, and PhysX contact replacement tradeoffs.
- Read [task4-phase1-ball-drop.md](task4-phase1-ball-drop.md) for the phase-1 ball-drop prototype and `scripts/trampoline_spring.py`.
- Read [task4-phase2-interface.md](task4-phase2-interface.md) for play-path integration, `foot_contact`, lower-leg collision removal, and URDF-based collision changes.
- Treat each `task-*.md` file as one concrete implementation pass or focused design artifact, not the global source of truth.
- For cross-phase work, read multiple docs as needed, but do not load every markdown file in the directory by default.

## Repo Validation Rules

- Treat the markdown docs in this directory as the primary source of project intent and planned design.
- Prefer `Structure.md` for stable architecture intent and `TODOs.md` for active state.
- Treat the repository code as the source of current implementation state.
- If the docs and code disagree, inspect the relevant code paths and explicitly call out the drift.
- In answers or plans, state which markdown files you read and which code files you checked.
- When a task doc references implementation files, follow those references instead of answering from the doc alone.

## Output Conventions

- Lead with the conclusion for the current phase or subtask, not a full documentation recap.
- Return to structure-level summary only when the user explicitly asks for overall architecture or prioritization.
- Do not paste long excerpts from the docs. Convert them into concrete implementation steps, checks, or differences.
- When drift exists, summarize it in the form: doc intent, current code state, and recommended next step.
