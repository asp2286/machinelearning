# Agent Guidelines for dotnet/machinelearning

Welcome! This file summarizes expectations for contributors and AI agents working in this repository.

## Repository orientation
- The solution file `Microsoft.ML.sln` aggregates the primary ML.NET libraries under `src/` and their accompanying tests in `test/`.
- Shared build logic lives under `build/` and `eng/`. Review these directories before changing build infrastructure.
- Documentation updates generally belong in `docs/`, with project-specific contributor guidance in `docs/project-docs/`.

## Coding conventions
- Follow the existing style in any file you modify. Align with the .NET runtime [coding style guidelines](https://github.com/dotnet/runtime/blob/main/docs/coding-guidelines/coding-style.md) when writing new code, but do **not** submit style-only changes.
- Prefer minimal, focused pull requests. Avoid mixing unrelated product, test, formatting, or infrastructure edits.
- When touching public APIs, ensure XML documentation stays in sync and update any relevant manifest files (for example the entry-point catalogs) if build/test output calls for it.

## Build and test expectations
- Always run a repo-level build before focusing on individual projects: `./build.sh -configuration Debug` (or `build.cmd` on Windows). Use `-configuration Release` or `/p:TargetArchitecture=<x64|x86>` when appropriate.
- Execute `./build.sh -test` (or `build.cmd -test`) to compile the product and run the full test suite. For targeted changes, you may additionally run `dotnet test` inside the affected project under `test/` after the initial repo build succeeds.
- If you need to regenerate entry point catalogs (`core_manifest.json` or `core_ep-list.tsv`), temporarily enable the `RegenerateEntryPointCatalog` test, run it, validate the output, and re-disable the skip attribute before committing.

## Workflow tips
- Initialize submodules with `git submodule update --init` before building if they are not already present.
- Use `build -help` to discover additional build/test flags supported by the repo scripts.
- Expect PRs to be squash-merged; keep commits reviewable and avoid merge commits.

## Communication
- Reference relevant issues and tag appropriate reviewers in PR descriptions.
- Adhere to the project's Code of Conduct and contribution guidelines when interacting with other contributors.
