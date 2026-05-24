# Project instructions

Project-specific Claude notes also live in `.claude/CLAUDE.md`.

- For trampoline calibration, ball-drop sweeps, robot-drop tests, or IsaacLab/MuJoCo comparison work, read `.agents/instruction.md` and `.agents/logs.md` first.
- Treat `.agents/instruction.md` as the source of truth for long-term plans, metrics, scripts, parameter mappings, and next steps.
- Treat `.agents/logs.md` as the source of truth for experiment results, output paths, failure modes, and modification history.
- When adding new sweep results or commands, update `.agents/logs.md` instead of putting the canonical record under `logs/`.
- Always respond in Chinese.

## Code modification tasks

- Do not guess uncertain details or intent. Ask for clarification before changing code when requirements are ambiguous.
- Before editing code, provide a short plan that states the purpose, the file list, and why each file will be modified.
- Keep implementations minimal and directly executable. Do not add compatibility layers, aliases, fallbacks, or defensive scaffolding unless explicitly required. When a planned change makes related old code obsolete, clean up that obsolete task-related code; do not touch unrelated unused code outside the plan scope.
- Do not modify anything unrelated to the stated plan purpose. If editing files outside the planned file list becomes necessary, ask for confirmation first. 
- Avoid incidental formatting, whitespace, or style-only changes.
