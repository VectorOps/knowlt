# Parser Performance Optimization Plan

## Goal

Improve parser throughput and reduce Python-side overhead by:

1. Replacing parser-facing `pydantic.BaseModel` parse result objects with lightweight dataclasses.
2. Moving validation into parser implementations and conversion boundaries.
3. Introducing a native parser path built with Cython and vendored Tree-sitter C sources.
4. Preserving the current storage model and public behavior while migrating language parsers one by one.

## Current Project Analysis

### Parser protocol and hot-path data objects

The current parser contract lives in `src/knowlt/parsers.py`.

Key observations:

- `ParsedImportEdge`, `ParsedPackage`, `ParsedNode`, and `ParsedFile` are currently `pydantic.BaseModel` classes.
- These objects are created heavily during parsing, including recursive `ParsedNode.children` trees.
- `AbstractCodeParser.parse()` returns `ParsedFile` and language parsers build large numbers of nested parser objects through `_make_node()` and `_process_node()`.
- This makes Pydantic object construction part of the parser hot path.

Expected performance impact:

- Pydantic validation and model machinery likely adds measurable overhead for every parsed node, import edge, package, and file.
- Recursive symbol trees amplify this overhead on large files and repositories.

Conclusion:

- The parser protocol is a strong candidate for replacing parser-side `BaseModel` objects with plain dataclasses or other minimal Python containers.

### Database and storage model separation already exists

The persistent models live in `src/knowlt/models.py`.

Key observations:

- `Repo`, `Package`, `File`, `Node`, and `ImportEdge` remain Pydantic models.
- This is a useful boundary: parser output can become lightweight transient dataclasses, then be converted into storage-facing models later.

Conclusion:

- We do not need to remove Pydantic everywhere.
- The lowest-risk change is to remove it only from transient parser result structures first.

### Current language parser landscape

The repository currently has dedicated parser implementations for:

- Python: `src/knowlt/lang/python.py`
- Go: `src/knowlt/lang/golang.py`
- JavaScript: `src/knowlt/lang/javascript.py`
- TypeScript: `src/knowlt/lang/typescript.py`
- C: `src/knowlt/lang/c.py`
- C++: `src/knowlt/lang/cpp.py`
- Terraform: `src/knowlt/lang/terraform.py`

Additional notes:

- Parsers use Python `tree_sitter` bindings and language-specific Python packages such as `tree_sitter_python`, `tree_sitter_go`, and similar.
- C++ extends the C parser, so the migration order should account for that dependency.
- There is already strong parser test coverage by language under `tests/lang/...`.

Conclusion:

- The codebase is in a good position for incremental migration because parser implementations are already separated by language and have targeted tests.

### Testing readiness

Relevant existing test coverage includes:

- `tests/lang/python/*`
- `tests/lang/golang/*`
- `tests/lang/javascript/*`
- `tests/lang/typescript/*`
- `tests/lang/c/*`
- `tests/lang/cpp/*`
- `tests/lang/terraform/*`

Conclusion:

- Migration should be validated parser-by-parser using existing language-specific test suites.
- Benchmarking scripts should be added to the test suite so parser performance changes are measured continuously during the migration.

## Recommended Target Architecture

### 1. Split parser-domain records from storage-domain models

Introduce parser result dataclasses as the only objects created in the parsing hot path.

Recommended parser-domain records:

- `ParsedImportEdge`
- `ParsedPackage`
- `ParsedNode`
- `ParsedFile`

Recommended characteristics:

- Use `@dataclass(slots=True)` where practical.
- Keep fields close to the current protocol to minimize migration cost.
- Keep recursive `children` support for `ParsedNode`.
- Keep helper serialization only if still needed by callers; otherwise prefer explicit conversion functions.

Why this helps:

- Lower object construction overhead.
- Lower attribute storage overhead with slots.
- No automatic validation cost on every node creation.

### 2. Move validation into explicit parser logic

Validation should happen where syntax and semantics are already being interpreted, not inside generic object construction.

Recommended validation approach:

- Validate required parser invariants before constructing dataclasses.
- Keep helper constructors such as `_make_node()` responsible for enforcing minimum valid state.
- Add focused assertions or lightweight checks for invariants like:
  - `start_byte <= end_byte`
  - `start_line <= end_line`
  - `kind` is present for symbol-bearing nodes
  - import edges have `virtual_path` and `raw`

Why this helps:

- Validation cost becomes intentional and localized.
- Parser behavior stays explicit and easier to tune.

### 3. Reuse the existing scanner conversion boundary

An explicit parser-to-storage boundary already exists in `src/knowlt/scanner.py`, which consumes `ParsedFile` and persists `Package`, `File`, `Node`, and `ImportEdge` models.

Recommended approach:

- Reuse `scanner.py` as the existing conversion boundary.
- Replace parser-side `BaseModel` objects with dataclasses that preserve the same field names, nesting, and defaults expected by scanner.
- Avoid introducing a new conversion layer unless a later native implementation proves the existing scanner boundary is insufficient.

Parity requirements for the dataclass protocol:

- `ParsedFile`, `ParsedPackage`, `ParsedNode`, and `ParsedImportEdge` should keep the same shape as the current parser objects.
- Attribute access patterns should remain unchanged, including nested access such as `parsed_file.package`, `parsed_file.nodes`, `node.children`, and `parsed_file.imports`.
- Optional and default values should remain compatible with current scanner behavior.
- Any existing parser-result serialization helper still in use, such as `to_dict()`, should remain until all call sites are removed.

Why this helps:

- Keeps the migration smaller and lower risk.
- Preserves the current parser-to-storage flow.
- Still separates transient parser data from persisted models without adding another abstraction layer.

### 4. Introduce a native parsing subsystem behind the same parser contract

Add a new native module path for supported languages while preserving the Python parser contract.

Recommended high-level package layout:

```text
src/knowlt/
  native/
    __init__.py
    api.py
    _native.pyx
    _native.cpp
    _tree_sitter.pxd
    _records.pxd
    vendor/
      tree-sitter/
      grammars/
        python/
        javascript/
        typescript/
        go/
        c/
        cpp/
```

Recommended native boundary:

- Cython handles Tree-sitter parser lifecycle and AST traversal.
- Parsing and tree traversal run `nogil` where possible.
- Native code accumulates compact intermediate records using C/C++ data structures.
- Python objects are created only at the end of extraction, ideally once per final record.
- Final return value should be parser dataclasses, not storage models.

Why this architecture is a good fit:

- It preserves current Python-facing semantics.
- It minimizes Python allocation in the hot path.
- It allows gradual native adoption by language.

### 5. Vendor generated Tree-sitter C sources instead of relying on Python grammar packages at runtime

Recommended build model:

- Vendor one pinned Tree-sitter runtime.
- Vendor generated grammar C sources per supported language.
- Track vendored runtime and grammars in a lock file, for example `grammars.lock.yml`, as the source of truth for upstream repos, pinned refs, vendored destinations, required files, and whether generation is required.
- Build them into one extension module initially.
- Commit generated Cython output needed for source builds.
- Do not run `tree-sitter generate` during end-user install.

Recommended lock file rules:

- Use immutable refs only, preferably exact commit SHAs.
- Do not use floating refs such as `master` or `main`.
- Distinguish between grammars that can copy already-generated files and grammars that require a generation step during vendoring.
- Record enough metadata to reproduce updates reliably, including runtime version, grammar commit, and generation mode.

Recommended vendoring workflow:

- A utility script should read `grammars.lock.yml`.
- For each runtime or grammar entry, clone the upstream repository into a temporary location and check out the pinned ref.
- If the entry requires generation, run a pinned `tree-sitter generate` step during the vendoring workflow, not during package installation.
- Copy only the declared files into the vendored destination.
- Write revision metadata alongside the vendored sources so updates are auditable and reproducible.

Benefits:

- Reproducible builds.
- No Node.js or Tree-sitter CLI required for users.
- Full control over runtime and grammar versions.
- Enables direct C API usage from Cython without Python binding overhead.

### 6. Keep one extension first, then split only if needed

Recommended first target:

- Single extension module, for example `knowlt.native._native`.

Reasoning:

- Lower migration complexity.
- Easier packaging and CI setup.
- Better fit while the native API is still evolving.

Only consider splitting into per-language extensions if:

- build times become problematic,
- wheel size becomes problematic, or
- optional language installation becomes a real product requirement.

## Proposed Migration Principles

### Preserve existing public behavior

The migration should not change:

- parser registry behavior,
- language-to-extension mapping,
- summary helper behavior,
- storage model semantics,
- test-visible parser output structure unless intentionally documented.

### Convert parsers one by one

Each language parser should be migrated independently behind a stable protocol.

Reasons:

- Limits blast radius.
- Keeps debugging manageable.
- Allows performance benchmarking by language.
- Uses existing test suites as acceptance gates.

### Keep Python implementations as reference behavior during transition

Before a language has a native implementation:

- it should use the new dataclass-based protocol in Python,
- remain the correctness reference,
- and continue to pass all existing tests.

After native implementation exists:

- keep the Python parser as a temporary reference during rollout,
- compare native output against Python output in targeted tests where feasible.

## Detailed Phased Plan

## Implementation Status Snapshot

- [x] Phase 0: Baseline and design freeze
- [x] Phase 1: Introduce lightweight parser dataclasses
- [x] Phase 2: Migrate Python parser to the new protocol first
- [ ] Phase 3: Migrate remaining pure-Python parser implementations one by one (in progress)
- [ ] Phase 4: Add native build foundations
- [ ] Phase 5: Implement the first native parser end to end
- [ ] Phase 6: Native migration of remaining languages one by one
- [ ] Phase 7: Rollout, fallback policy, and cleanup

Current completed implementation items:

- [x] Parser-domain `BaseModel` result objects replaced with dataclasses in `src/knowlt/parsers.py`
- [x] Existing scanner boundary in `src/knowlt/scanner.py` preserved without introducing a new conversion module
- [x] Lightweight parser-side validation added in parser dataclasses
- [x] Parser protocol regression tests added in `tests/test_parsers.py`
- [x] Parser-only benchmark added in `tests/benchmarks/test_python_parser_benchmark.py`
- [x] Parse-plus-upsert benchmark added in `tests/benchmarks/test_python_upsert_benchmark.py`
- [x] Python parser low-risk text-decoding cache optimization completed
- [x] Go parser low-risk text-decoding cache optimization completed
- [x] JavaScript parser low-risk text-decoding cache optimization completed
- [x] TypeScript parser low-risk text-decoding cache optimization completed
- [x] Repository-wide validation passes via `uv` with `75 passed, 2 skipped`

### Phase 0: Baseline and design freeze

Status: [x] Completed

Objective:

- Establish the current parser contract and baseline performance before changing behavior.

Work items:

- [x] Inventory all parser-domain `BaseModel` usage and creation sites.
- [x] Identify all places where parser-domain objects are converted into storage models or dictionaries.
- [x] Add benchmarking scripts to the test suite so baseline and post-migration performance can be compared consistently.
- [x] Record initial file-level benchmark timings for stable Python fixture workloads.
- [x] Freeze the parser protocol fields that must remain semantically compatible.
- [ ] Benchmark representative parser workloads per language.
- [ ] Record repo-level parse timings for a stable multi-language fixture set.

Deliverables:

- Baseline benchmark notes.
- Benchmarking scripts integrated into the test suite.
- Agreed parser dataclass schema.
- Agreed migration acceptance criteria.

Acceptance criteria:

- Current behavior is documented well enough to detect regressions.
- Performance measurement is repeatable through the test suite.

Initial inventory findings for Phase 0, Step 1:

- Parser-domain `BaseModel` definitions currently live in `src/knowlt/parsers.py`:
  - `ParsedImportEdge`
  - `ParsedPackage`
  - `ParsedNode`
  - `ParsedFile`
- These are the primary parser hot-path objects and are constructed recursively during parse tree traversal.
- Common parser-side construction sites currently include:
  - `AbstractCodeParser._make_node()` in `src/knowlt/parsers.py`
  - `AbstractCodeParser._create_package()` in `src/knowlt/parsers.py`
  - `AbstractCodeParser._create_file()` in `src/knowlt/parsers.py`
  - language-specific import helpers that construct `ParsedImportEdge`, such as C include resolution and JavaScript/TypeScript import handlers
- The main parser-to-storage consumption boundary is `upsert_parsed_file()` in `src/knowlt/scanner.py`.
- Existing scanner integration tests already exercise this boundary, especially `tests/test_scanner.py`.
- Existing language parser tests that validate parser-domain output include:
  - `tests/lang/python/test_python_parser.py`
  - `tests/lang/golang/test_golang_parser.py`
  - `tests/lang/javascript/test_javascript_parser.py`
  - `tests/lang/typescript/test_typescript_parser.py`
  - `tests/lang/c/test_c_parser.py`
  - `tests/lang/cpp/test_cpp_parser.py`
  - `tests/lang/terraform/test_terraform_parser.py`
- An initial opt-in benchmark harness now exists at `tests/benchmarks/test_python_parser_benchmark.py` and uses the existing Python parser sample fixture as the first baseline workload.
- Immediate migration constraint: dataclass replacements must preserve the current field shape and nested attribute access expected by `scanner.py` and the parser tests.

Initial inventory findings for Phase 0, Step 2:

- The primary parser-to-storage conversion path is `upsert_parsed_file()` in `src/knowlt/scanner.py`.
- Current conversion from parser-domain objects to persisted models happens through existing `to_dict()` helpers on parser objects:
  - `ParsedPackage.to_dict()` -> `Package(...)` create/update payload
  - `ParsedFile.to_dict()` -> `File(...)` create/update payload
  - `ParsedImportEdge.to_dict()` -> `ImportEdge(...)` create/update payload
  - `ParsedNode.to_dict()` -> `Node(...)` create payload inside recursive symbol insertion
- `upsert_parsed_file()` relies on plain attribute access in addition to `to_dict()`, including:
  - `parsed_file.package`
  - `parsed_file.path`
  - `parsed_file.imports`
  - `parsed_file.nodes`
  - `parsed_file.package.virtual_path`
  - `imp.virtual_path`, `imp.alias`, `imp.dot`, `imp.external`
  - `psym.body`, `psym.docstring`, `psym.children`
- Recursive symbol persistence depends on `ParsedNode.children` preserving the current nested tree structure.
- No parser conversion path currently depends on Pydantic validation APIs such as `model_dump()`; the active compatibility surface is field shape, attribute access, and the existing `to_dict()` methods.
- `tests/test_scanner.py` is the main integration test covering this conversion boundary and should be treated as a migration gate.
- Immediate migration implication: parser dataclasses can replace parser-side `BaseModel` objects without a new conversion layer as long as they preserve the current attributes, defaults, recursion structure, and `to_dict()` behavior expected by `scanner.py`.

Initial compatibility contract for Phase 0, Step 3:

- The dataclass migration target should preserve the current parser protocol field set exactly unless a later migration step explicitly changes it.
- Required parser-domain object shapes are:

  - `ParsedImportEdge`
    - fields: `physical_path`, `virtual_path`, `alias`, `dot`, `external`, `raw`
    - behavior: `to_dict()` must keep returning `to_package_physical_path`, `to_package_virtual_path`, `alias`, `dot`, `external`, and `raw`

  - `ParsedPackage`
    - fields: `language`, `physical_path`, `virtual_path`, `imports`
    - behavior: `to_dict()` must keep returning `name`, `language`, `virtual_path`, and `physical_path`

  - `ParsedNode`
    - fields: `name`, `body`, `kind`, `subtype`, `start_line`, `end_line`, `start_byte`, `end_byte`, `header`, `visibility`, `docstring`, `comment`, `children`
    - behavior: `to_dict()` must keep returning all scalar persistence fields except `children`

  - `ParsedFile`
    - fields: `package`, `path`, `docstring`, `file_hash`, `last_updated`, `nodes`, `imports`
    - behavior: `to_dict()` must keep returning `path`, `file_hash`, and `last_updated`

- Semantic compatibility requirements:
  - `children`, `nodes`, and `imports` must remain mutable lists.
  - `package` must remain optional.
  - `name`, `subtype`, `header`, `visibility`, `docstring`, and `comment` must remain optional where they are optional today.
  - `kind` and `language` must continue to use the existing enum types.
  - line numbers remain 1-based as produced by current parser helpers.
  - byte offsets remain copied directly from Tree-sitter node byte ranges.

- Construction compatibility requirements:
  - `AbstractCodeParser._create_package()` and `_create_file()` must still be able to instantiate the parser-domain objects with the same keyword arguments they use today.
  - `AbstractCodeParser._make_node()` must still be able to construct parser nodes with the current keyword arguments and default behavior.

- Behavioral compatibility requirements:
  - `AbstractCodeParser.parse()` must keep synchronizing `package.imports` from `parsed_file.imports`.
  - `scanner.py` must be able to recurse through `ParsedNode.children` without any adapter layer.
  - Existing parser tests should not need semantic updates just because the backing type changed from `BaseModel` to dataclass.

Initial Phase 1 status:

- A search for Pydantic-specific parser-result usage outside the parser protocol definitions did not find active dependencies on parser-object APIs such as `model_dump()`, schema generation, or Pydantic copy helpers.
- The active parser-result compatibility surface is still `to_dict()`, direct attribute access, and recursive list traversal in `scanner.py`.
- `ParsedImportEdge`, `ParsedPackage`, `ParsedNode`, and `ParsedFile` have now been converted from `pydantic.BaseModel` to shape-compatible dataclasses in `src/knowlt/parsers.py`.
- The migration currently preserves the existing field names, optionality, mutable list behavior, and `to_dict()` methods used by scanner persistence.
- Initial targeted validation after the dataclass conversion passed for:
  - `tests/lang/python/test_python_parser.py`
  - `tests/lang/text/test_text_parser.py`
  - `tests/test_scanner.py`
- Broader parser and summary validation also passed for the remaining current language implementations:
  - `tests/lang/c/test_c_parser.py`
  - `tests/lang/c/test_c_summary.py`
  - `tests/lang/cpp/test_cpp_parser.py`
  - `tests/lang/cpp/test_cpp_summary.py`
  - `tests/lang/golang/test_golang_parser.py`
  - `tests/lang/golang/test_golang_summary.py`
  - `tests/lang/javascript/test_javascript_parser.py`
  - `tests/lang/javascript/test_javascript_summary.py`
  - `tests/lang/typescript/test_typescript_parser.py`
  - `tests/lang/typescript/test_typescript_summary.py`
  - `tests/lang/terraform/test_terraform_parser.py`
  - `tests/lang/terraform/test_terraform_summary.py`
  - `tests/lang/markdown/test_markdown_parser.py`
- Project-level end-to-end validation also passed for the currently covered language flows:
  - `tests/lang/python/test_python_project.py`
  - `tests/lang/golang/test_golang_project.py`
  - `tests/lang/c/test_c_project.py`
  - `tests/lang/cpp/test_cpp_project.py`
- Initial benchmark baseline captured via `tests/benchmarks/test_python_parser_benchmark.py`:
  - workload: `tests/lang/python/samples/simple.py`
  - iterations: `200`
  - total time: `0.077408s`
  - average per parse: `0.387 ms`
  - minimum per parse: `0.295 ms`
  - maximum per parse: `1.601 ms`
- Lightweight parser-side validation is now implemented directly in the parser dataclasses via `__post_init__` checks for core invariants such as:
  - non-empty `ParsedFile.path`
  - non-empty `ParsedPackage.physical_path` and `ParsedPackage.virtual_path`
  - non-empty `ParsedImportEdge.virtual_path` and `ParsedImportEdge.raw`
  - ordered and non-negative `ParsedNode` line and byte ranges
- Focused post-validation regression coverage passed for:
  - `tests/lang/python/test_python_parser.py`
  - `tests/lang/text/test_text_parser.py`
  - `tests/test_scanner.py`
  - `tests/lang/javascript/test_javascript_parser.py`
  - `tests/lang/typescript/test_typescript_parser.py`
  - `tests/lang/golang/test_golang_parser.py`
  - `tests/lang/c/test_c_parser.py`
  - `tests/lang/terraform/test_terraform_parser.py`
- Repository-wide validation result after the parser dataclass migration:
  - after subsequent fixes and validation via `uv`, the repository test suite passed with `75 passed, 2 skipped`
- Dedicated parser protocol regression coverage now exists in `tests/test_parsers.py` for:
  - parser dataclass `to_dict()` shape compatibility with `scanner.py`
  - recursive `ParsedNode.children` shape preservation
  - direct validation of parser dataclass invariants for file, package, import-edge, and node construction
- Benchmark coverage has been expanded beyond parse-only measurement:
  - `tests/benchmarks/test_python_parser_benchmark.py` measures parser-only baseline cost
  - `tests/benchmarks/test_python_upsert_benchmark.py` measures end-to-end Python parse plus `upsert_parsed_file()` cost through the current scanner persistence path
- Initial end-to-end parse-plus-upsert benchmark baseline captured via `tests/benchmarks/test_python_upsert_benchmark.py`:
  - workload: synthetic Python module written to `mod.py`
  - iterations: `50`
  - parse total time: `0.006941s`
  - parse average per iteration: `0.139 ms`
  - upsert total time: `0.372592s`
  - upsert average per iteration: `7.452 ms`
- First Python-specific optimization pass completed in `src/knowlt/lang/python.py`:
  - added a per-parse node-text cache keyed by byte range to avoid repeated UTF-8 decoding of the same Tree-sitter nodes during Python parser traversal
  - updated decorator gathering, import handling, function/class header building, docstring extraction, comment extraction, async detection, and debug logging to reuse cached node text
- Targeted validation after the Python parser optimization passed via `uv` for:
  - `tests/lang/python/test_python_parser.py`
  - `tests/lang/python/test_python_project.py`
  - `tests/test_scanner.py`
  - `tests/test_parsers.py`
- Updated benchmark results after the Python parser text-cache optimization:
  - parser-only benchmark (`tests/benchmarks/test_python_parser_benchmark.py`)
    - iterations: `200`
    - total time: `0.067838s`
    - average per parse: `0.339 ms`
    - minimum per parse: `0.261 ms`
    - maximum per parse: `1.796 ms`
  - parse-plus-upsert benchmark (`tests/benchmarks/test_python_upsert_benchmark.py`)
    - iterations: `50`
    - parse total time: `0.006537s`
    - parse average per iteration: `0.131 ms`
    - upsert total time: `0.375703s`
    - upsert average per iteration: `7.514 ms`
- First Go-specific optimization pass completed in `src/knowlt/lang/golang.py`:
  - added a per-parse node-text cache keyed by byte range to reduce repeated UTF-8 decoding during Go parser traversal
  - updated package-name extraction, import handling, header building, type handling, comment extraction, member extraction, and debug logging to reuse cached node text
- Targeted validation after the Go parser optimization passed via `uv` for:
  - `tests/lang/golang/test_golang_parser.py`
  - `tests/lang/golang/test_golang_summary.py`
  - `tests/lang/golang/test_golang_project.py`
  - `tests/test_scanner.py`
  - `tests/test_parsers.py`
- First JavaScript-specific optimization pass completed in `src/knowlt/lang/javascript.py`:
  - added a per-parse node-text cache keyed by byte range to reduce repeated UTF-8 decoding during JavaScript parser traversal
  - updated import/export handling, header building, holder-name resolution, comment extraction, CommonJS export detection, require-call collection, lexical handling, and debug logging to reuse cached node text
- Targeted validation after the JavaScript parser optimization passed via `uv` for:
  - `tests/lang/javascript/test_javascript_parser.py`
  - `tests/lang/javascript/test_javascript_summary.py`
  - `tests/test_scanner.py`
  - `tests/test_parsers.py`
- First TypeScript-specific optimization pass completed in `src/knowlt/lang/typescript.py`:
  - added a per-parse node-text cache keyed by byte range to reduce repeated UTF-8 decoding during TypeScript parser traversal
  - updated import/export handling, header building, type parameter extraction, class/interface/enum/namespace handling, lexical handling, CommonJS export detection, require-call collection, comment extraction, and debug logging to reuse cached node text
- Targeted validation after the TypeScript parser optimization passed via `uv` for:
  - `tests/lang/typescript/test_typescript_parser.py`
  - `tests/lang/typescript/test_typescript_summary.py`
  - `tests/test_scanner.py`
  - `tests/test_parsers.py`

### Phase 1: Introduce lightweight parser dataclasses

Status: [x] Completed

Objective:

- Remove Pydantic from the parser hot path without changing storage models.

Work items:

- [x] Replace parser-side `BaseModel` classes with dataclasses.
- [x] Use `slots=True` for parser-domain dataclasses.
- [x] Update `_make_node()` and related helpers to construct dataclasses.
- [x] Keep explicit `to_dict()` where scanner persistence still needs it.
- [x] Preserve shape parity with the current parser objects so `scanner.py` can keep consuming them as the parser-to-storage boundary.
- [x] Verify that `scanner.py` does not require a new conversion layer.
- [x] Verify that no active parser-result call sites depend on Pydantic-only APIs.

Deliverables:

- Dataclass-based parser protocol with shape parity to current parser objects.

Acceptance criteria:

- All existing parser tests pass.
- No parser-facing Pydantic objects remain in the hot path.
- `scanner.py` continues to act as the parser-to-storage boundary without requiring a new conversion module.

### Phase 2: Migrate Python parser to the new protocol first

Status: [x] Completed

Objective:

- Use Python as the first end-to-end migration and reference implementation.

Why Python first:

- It is isolated and mature.
- It has dedicated parser, project, and summary tests.
- It is likely a common workload and a good benchmark anchor.

Work items:

- [x] Migrate `src/knowlt/lang/python.py` to the dataclass protocol.
- [x] Move validation into parser-side dataclass checks and parser helpers.
- [x] Verify that summary generation still works through existing storage and summary flows.
- [x] Run Python parser and project tests.
- [x] Capture benchmark before and after the first Python-specific optimization pass.
- [x] Apply a first low-risk Python parser optimization pass.

Acceptance criteria:

- Python parser behavior remains stable.
- Parsing performance improves or at minimum does not regress.

### Phase 3: Migrate remaining pure-Python parser implementations one by one

Status: [ ] In progress

Objective:

- Standardize every language parser on the lightweight protocol before adding native implementations.

Recommended order:

1. Go
2. JavaScript
3. TypeScript
4. Terraform
5. C
6. C++

Rationale for this order:

- Go, JavaScript, TypeScript, and Terraform are relatively self-contained.
- C should be migrated before C++ because C++ builds on C parser behavior.

Execution checklist:

- [x] Shared dataclass protocol now applies across all existing language parsers.
- [x] Broader parser and summary validation has been run across C, C++, Go, JavaScript, TypeScript, Terraform, and Markdown.
- [x] Project-level validation has been run for Python, Go, C, and C++.
- [x] Go parser low-risk optimization pass completed.
- [x] JavaScript parser low-risk optimization pass completed.
- [x] TypeScript parser low-risk optimization pass completed.
- [ ] Terraform parser low-risk optimization pass.
- [ ] C parser low-risk optimization pass.
- [ ] C++ parser low-risk optimization pass.
- [ ] Language-specific benchmark capture beyond Python.

Acceptance criteria:

- Every existing language parser uses the same lightweight parser protocol.
- Language-specific tests pass after each migration.

### Phase 4: Add native build foundations

Status: [ ] Not started

Objective:

- Prepare the repository for Cython and vendored Tree-sitter sources without changing parser behavior yet.

Work items:

- [ ] Add native package layout under `src/knowlt/native/`.
- [ ] Add Cython build configuration in `pyproject.toml` and, if needed, `setup.py`.
- [ ] Add vendored Tree-sitter runtime.
- [ ] Add vendored generated grammar sources for the first target language.
- [ ] Add `grammars.lock.yml` to define pinned runtime and grammar sources, vendored destinations, required files, and generation behavior.
- [ ] Add a utility script for grammar/runtime updates that reads the lock file, clones pinned upstream repos, optionally runs generation, and copies only the required vendored artifacts.
- [ ] Add source distribution inclusion rules for `.pyx`, `.pxd`, generated `.cpp`, headers, and vendored C/C++ files.
- [ ] Add CI wheel build strategy, ideally with cibuildwheel.

Deliverables:

- Buildable native extension skeleton.
- Reproducible vendoring/update workflow.
- Lockfile-driven grammar/runtime vendoring process.

Acceptance criteria:

- The project can build the native extension in CI.
- No runtime parser generation is required during install.

### Phase 5: Implement the first native parser end to end

Status: [ ] Not started

Objective:

- Prove the native architecture with one language before scaling out.

Recommended first native language:

- Python

Reasons:

- Strong test coverage.
- High likelihood of broad usage.
- Good benchmark signal.

Work items:

- [ ] Define Cython declarations for Tree-sitter C APIs.
- [ ] Define native intermediate record structures.
- [ ] Implement parse and traversal logic under `nogil` where possible.
- [ ] Convert native records into Python parser dataclasses at the boundary.
- [ ] Keep output semantically identical to the Python parser.
- [ ] Add tests comparing native and Python parser output for representative fixtures.
- [ ] Benchmark native vs Python implementation.

Acceptance criteria:

- Native Python parser matches current behavior.
- Meaningful speedup is demonstrated on representative workloads.

### Phase 6: Native migration of remaining languages one by one

Status: [ ] Not started

Objective:

- Incrementally replace Python Tree-sitter binding usage with native extractors.

Recommended native migration order:

1. Go
2. JavaScript
3. TypeScript
4. Terraform
5. C
6. C++

Per-language workflow:

- [ ] Vendor grammar C sources.
- [ ] Implement language entrypoint and traversal in Cython/native code.
- [ ] Serialize final results into parser dataclasses.
- [ ] Run existing language tests.
- [ ] Add native-vs-reference comparison tests for tricky constructs.
- [ ] Benchmark and record improvement.

Special note for C/C++:

- Migrate C first.
- Reuse shared native helpers for declarators, includes, aggregates, and field traversal.
- Then layer C++-specific namespaces, templates, and class semantics on top.

Acceptance criteria:

- Languages are promoted individually only after behavior parity and acceptable performance gains.

### Phase 7: Rollout, fallback policy, and cleanup

Status: [ ] Not started

Objective:

- Make the native path the default when stable and remove transitional complexity carefully.

Work items:

- [ ] Decide whether Python implementations remain as test-only references or production fallbacks.
- [ ] Gate native parser selection behind feature flags during rollout if needed.
- [ ] Remove obsolete parser-side compatibility shims.
- [ ] Tighten benchmarks and regression checks in CI.
- [ ] Once a Cython implementation is fully in place and validated for a language, remove the corresponding pure-Python production parser implementation rather than keeping dual runtime paths indefinitely.
- [ ] Document supported build environments and wheel strategy.

Acceptance criteria:

- Native path is stable in CI and packaging.
- Pure-Python production parser implementations are removed once their Cython replacements are stable and adopted.
- Rollout behavior is documented and predictable.

## Language-by-Language Conversion Matrix

The following sequence should be followed strictly so existing parsers are converted one by one.

| Order | Language | Dataclass migration | Native migration | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | Python | First | First | [x] Protocol migration done; [x] first optimization pass done | Best reference and benchmark anchor |
| 2 | Go | Second | Second | [x] Protocol migration done; [x] first optimization pass done | Self-contained parser |
| 3 | JavaScript | Third | Third | [x] Protocol migration done; [x] first optimization pass done | Good candidate before TS |
| 4 | TypeScript | Fourth | Fourth | [x] Protocol migration done; [x] first optimization pass done | Shares patterns with JS but adds more syntax |
| 5 | Terraform | Fifth | Fifth | [x] Protocol migration done; [ ] optimization pass pending | Self-contained and simpler than C family |
| 6 | C | Sixth | Sixth | [x] Protocol migration done; [ ] optimization pass pending | Base for C++ migration |
| 7 | C++ | Seventh | Seventh | [x] Protocol migration done; [ ] optimization pass pending | Depends on C parser concepts |

Rule:

- Do not begin native migration for a language until its Python implementation already uses the dataclass protocol and passes tests.

## Proposed Repository Additions

Recommended new files and directories:

```text
plan.md
setup.py                       # if programmatic extension config is needed
MANIFEST.in
grammars.lock.yml

src/knowlt/
  parser_records.py            # dataclasses for ParsedFile/ParsedNode/etc.
  native/
    __init__.py
    api.py
    _native.pyx
    _native.cpp
    _native.pyi
    _tree_sitter.pxd
    _records.pxd
    vendor/
      tree-sitter/
      grammars/
        python/
        go/
        javascript/
        typescript/
        terraform/
        c/
        cpp/

scripts/
  update_tree_sitter_runtime.sh
  update_grammar.py
  regenerate_cython.sh
  vendor_grammars.py
```

Note:

- The exact package naming can be adjusted, but parser dataclasses and native code should be isolated from persistence models.
- A dedicated parser converter module is not required initially because `src/knowlt/scanner.py` already serves as the parser-to-storage boundary.

## Risks and Mitigations

### Risk: behavior drift during migration

Mitigation:

- Migrate one parser at a time.
- Use existing language-specific tests as gates.
- Add native-vs-reference comparisons for representative fixtures.

### Risk: packaging complexity

Mitigation:

- Start with one extension module.
- Vendor generated sources.
- Build wheels in CI.
- Avoid install-time parser generation.

### Risk: native and Python outputs diverge subtly

Mitigation:

- Define a strict parser dataclass contract first.
- Compare normalized parser outputs in tests.
- Keep Python implementation as reference until parity is proven.

### Risk: premature over-optimization

Mitigation:

- Benchmark after Phase 1 before committing to deeper rewrites.
- Confirm that Pydantic removal and Python binding overhead are real bottlenecks.

## Recommended Immediate Next Steps

- [x] Implement Phase 0 baseline measurements.
- [x] Replace parser-side Pydantic models with dataclasses.
- [x] Migrate the Python parser fully to the new protocol.
- [x] Validate with `tests/lang/python/`.
- [x] Add benchmarking scripts to the test suite as part of the migration baseline.
- [ ] Continue language-by-language low-risk optimization passes in the order defined above, starting with Terraform.
- [ ] Add benchmark coverage for non-Python language parsers.
- [ ] Introduce the native extension foundation.
- [ ] Remove pure-Python production parser implementations after each Cython replacement is proven stable.

## Definition of Done for the Overall Initiative

This initiative is complete when:

- all parser-domain hot-path objects are lightweight dataclasses,
- all existing language parsers have been converted one by one,
- benchmarking scripts are part of the test suite and used to track parser performance,
- native parsing infrastructure is reproducible and wheel-buildable,
- native implementations exist for targeted languages,
- pure-Python production parser implementations have been removed after successful Cython migration,
- parser outputs remain semantically compatible,
- and benchmarked performance is materially better than the current baseline.