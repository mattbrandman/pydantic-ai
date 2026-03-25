# PLAN: Local Native Toolsets — ShellToolset, TextEditorToolset, ApplyPatchToolset

> **Issue refs:** #3365 (Anthropic/OpenAI Skills), #3963 (Shell/Bash builtin), #3794 (Text Editor tool)
> **Stack:** `continuation-support` -> `skill-support-v2` -> `local-tools` (this)
> **Depends on:** `skill-support-v2` (remote code tools, NativeToolDefinition infrastructure, model adapter native tool handling)

---

## Scope

This change adds the **local native toolset implementations**, their dedicated tests, VCR cassettes, and documentation. The core infrastructure (NativeToolDefinition types, model adapter native tool handling, profile flags) is in the parent change (`skill-support-v2`).

---

## 1. Toolset Implementations

### 1.1 `ShellToolset` (`toolsets/shell.py`)

Client-executed shell using provider-native format when supported (Anthropic `bash_20250124`, OpenAI `shell` with `local` env), falling back to function tool otherwise.

- `ShellToolset.local(cwd, env)` — convenience constructor with subprocess-based executor
- `ShellExecutor` protocol — pluggable execution backend
- `_LocalShellExecutor` — persistent shell session (state persists across calls)
- `ShellOutput` — `output: str`, `exit_code: int`
- `sequential=True` on `ToolDefinition` (subprocess is single-threaded)
- Timeout via `anyio.fail_after` -> `ModelRetry`
- Output truncation at `max_output_chars`

### 1.2 `TextEditorToolset` (`toolsets/text_editor.py`)

Client-executed text editor using Anthropic's `text_editor_20250728` native format.

- `TextEditorCommand` discriminated union: `view`, `str_replace`, `create`, `insert`
- `TextEditorOutput` dataclass
- `TextEditorExecuteFunc` callback — user provides the file operation implementation
- `max_characters` config passed through to native definition

### 1.3 `ApplyPatchToolset` (`toolsets/apply_patch.py`)

Client-executed patch application using OpenAI's `apply_patch` native format (V4A diffs).

- `ApplyPatchOperation`: `create_file`, `update_file`, `delete_file`
- `ApplyPatchOutput` dataclass
- `ApplyPatchExecuteFunc` callback — user provides the patch application implementation

---

## 2. Test Strategy

### 2.1 Unit Tests (`tests/test_shell_toolset.py`)

- `ShellToolset.get_tools()` returns correct `ToolDefinition` with `native_definition`
- `TextEditorToolset.get_tools()` returns correct definitions
- `ApplyPatchToolset.get_tools()` returns correct definitions
- Executor protocol compliance
- Timeout and truncation behavior
- Command parsing for each toolset

### 2.2 VCR Integration Tests

| Test | Provider | Streaming |
|------|----------|-----------|
| `test_anthropic_local_shell_toolset` | Anthropic | No |
| `test_anthropic_local_shell_toolset_stream` | Anthropic | Yes |
| `test_anthropic_text_editor_toolset` | Anthropic | No |
| `test_anthropic_text_editor_toolset_stream` | Anthropic | Yes |
| `test_openai_responses_local_shell_toolset` | OpenAI | No |
| `test_openai_responses_local_shell_toolset_stream` | OpenAI | Yes |
| `test_openai_responses_apply_patch_toolset` | OpenAI | No |
| `test_openai_responses_apply_patch_toolset_stream` | OpenAI | Yes |

### 2.3 Fallback Tests

- `ShellToolset` with unsupported provider emits function tool + `warnings.warn()`
- `TextEditorToolset` with unsupported provider emits function tool + warning
- `ApplyPatchToolset` with unsupported provider emits function tool + warning

---

## 3. Documentation

- `docs/native-tools.md` — user-facing guide with decision tree, security warnings, examples
- `docs/builtin-tools.md` — ShellTool builtin docs (remote)
- `docs/api/toolsets.md` — API reference for toolset classes
- `mkdocs.yml` — nav entry for native-tools page

---

## 4. Security

All shell-related docs include prominent warnings about running LLM-generated commands locally. Built-in safety mechanisms:

- Output truncation (`max_output_chars`)
- Timeout (`ShellToolset.timeout`)
- `ApprovalRequiredToolset` composition for human-in-the-loop
- `FilteredToolset` composition for command restrictions
- `ShellExecutor` protocol gives users full control over execution

---

## 5. Future Work (Out of Scope)

- **Capability wrapper**: `Shell`, `TextEditor`, `CodingEnvironment` capabilities that auto-route between remote and local
- **Execution Environment integration** (post-PR #4393): `CodingEnvironment` backed by `ExecutionEnvironment` ABC
- **`MemoryTool` refactor**: Replace with capability-driven pattern
