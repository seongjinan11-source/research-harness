# Research harness

Read `CLAUDE.md` for this repository's data, citation, build, and commit workflow.

When incorporating experiment reports, keep the original in `findings/sources/`
and update structured `findings/*.yaml`. Distinguish failed experiments,
preserved effects, lessons, and untested proposals. Record confirmed blockers
separately from hypotheses or unknown causes, and include reopening conditions,
strong comparison baselines, and a minimal recheck.

When adding a verified paper, run `revisit --paper ID` and rebuild the outputs.
Matches are retrieval leads; they never establish that a failure is solved.
Do not restart training merely because a tag or keyword matched.

Run `validate`, the relevant tests, and `build` before committing. Generated
files in `out/` come from the YAML and builder; edit their sources instead.
