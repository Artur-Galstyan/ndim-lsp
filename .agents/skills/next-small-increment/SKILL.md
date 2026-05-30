---
name: next-small-increment
description: Provide a batch of 3 incremental new changes that are distinct from each other and can be implemented separately
disable-model-invocation: false
---

You operate inside a code repository. Your job is to propose 3 small, incremental changes and write each one as a standalone prompt I can hand to a separate agent working in parallel.
Priorities, in order:

Each change is genuinely useful and moves the project forward. No filler.
Each change can be implemented and reviewed on its own.
When all 3 are merged together, there are as few merge conflicts as possible.
Do not skip a useful abstraction just to avoid conflicts. If it helps, include it and give it a clear owner.

Steps:

Read the repo. Open the main files and note how the modules fit together. Do not guess from names.
Pick 3 changes. Prefer ones that touch different files, or different regions of the same file. Keep each one small enough to finish in a single focused session. Avoid over-engineering.
Map the overlap. For each change, list the exact files it will create or edit. Compare the lists across all 3. If two changes edit the same lines of the same file, redesign one of them.
Watch registration points. Files like module lists, route tables, dependency manifests, and central entry points are where conflicts usually happen, because every change wants to add itself there. Where two changes must both touch such a file, say so plainly and keep each one's edit as small and separated as possible.
Handle shared abstractions explicitly. If two or more changes need the same new helper, type, or interface, do one of these:

Put the abstraction inside exactly one of the tasks, and have the others assume it exists.
Or make it a separate "task 0" the others build on, and label it as a prerequisite.
Never let two agents independently build the same abstraction.



Output format:
First, an overview: the 3 changes in one line each, plus a small table showing which files each change touches, so the overlap is visible at a glance.
Then, for each change, a self-contained prompt containing:

Title
What to build and why (2 to 3 sentences)
Exact files to create or edit, and any files to avoid touching
Any abstraction it owns or depends on
Definition of done

Rules for the prompts:

Each prompt must stand alone. The receiving agent will not see this skill or the other two prompts.
State assumptions instead of asking questions, since these go to agents that cannot reply.
