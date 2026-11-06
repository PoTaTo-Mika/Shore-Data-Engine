# Code Style Prompt (reusable)

A repository-agnostic style spec distilled from a mature ML codebase (fish-speech). Paste the
**Prompt** block into an agent's system/project instructions when you want clean, professional,
maintainable code in *any* repo. North star: **simplicity** — write less, not more.

## Prompt

```text
Write clean, professional, maintainable code. Match the surrounding codebase's conventions
where they exist; where they don't, follow these rules.

The overriding goal is simplicity: fewer, smaller files and fewer functions. Don't introduce
a helper, wrapper, abstraction layer, or private `_xxx` function unless it is used at least
twice right now AND genuinely clarifies the call site. Speculative generality is noise.

### SIMPLICITY & NON-DUPLICATION
- Don't reinvent what stdlib or an already-imported dep provides (`itertools`, `functools`,
  `collections`, `pathlib`, `dataclasses`, `pydantic`, `torch.nn.functional`). A 3-line
  wrapper around `lru_cache` is noise.
- Don't pre-extract `_helper`/`_impl` for "cleaner main flow" unless reused or too long to
  read in one screen. A 60-line top-to-bottom function beats three 20-line `_step_one/_two/
  _three` called once each.
- Don't add interfaces/base classes/registries/plugin systems before a second concrete
  implementation exists. Two cases first, then abstract.
- Don't stack config dataclass + yaml loader + validator + CLI override when argparse + dict
  would do. Match the repo's existing mechanism; if none, pick the lightest that fits.
- One concern per file ≠ one function per file. Related functions cohabit fine.

### LANGUAGE & COMMENTS
- English only: comments, docstrings, log strings, CLI help. (User-facing translatable
  strings go through i18n — not this rule's concern.) If the repo is non-English, match it.
- Comments sparse, one-line, only for the genuinely non-obvious. Restating code is noise.
- NO process markers: no ★, `# P1`, `# [FIX]`, `# TODO` without a ticket, `# === SECTION ===`
  banners, manual log prefixes like `[Info]`/`[step N]` (the level field already says it).
- NO provenance leakage: never name other repos/upstream/"the closed source" in source.
- NO verbose rationale in comments — the "why" lives in design docs / commits / PRs.
- Docstrings: Google-style, 1-3 lines. Args/Returns only when non-obvious. One short module
  docstring per file.
- `# noqa: <code>` allowed with a reason; bare `# noqa` is not.

### NAMING
- Classes PascalCase; functions/variables snake_case; constants UPPER_SNAKE; internal
  members leading underscore. `__double` only when name mangling is needed (rare).
- Names say what, not how: `load_checkpoint` not `do_thing`; `num_codebooks` not `n`.
  Single letters only for loop indices (`i`,`j`) or math (`x`,`y`,`t`).

### TYPING & SIGNATURES
- Full type hints, modern syntax: `X | Y`, `list[...]`, `dict[str, Any]`, `X | None`.
  Annotate return types.
- ONE typing style per repo. Don't mix `Optional[X]`/`Union[X,Y]` with `X | Y`. Modern
  preferred; if the repo uses `Optional`, match it.
- Either `requires-python >= 3.10` (native `X | Y`) or `from __future__ import annotations`.
  Don't use `Optional` to work around forward refs — use string quotes (`"ModelConfig"`) or
  `TYPE_CHECKING`.
- Closed value sets → `Literal[...]` or `Enum`, not bare strings in comparisons.
- No `any`/`Any` annotation. No bare `dict`/`list`/`tuple`. No `tokenizer: any`.
- No mutable defaults: `def f(x=[])`/`= {}` are bugs. Use `None` sentinel or
  `field(default_factory=list)`.
- No "_function" like functions, please use "function" for better maintainence.

### DATA STRUCTURES
- Internal value objects (config, messages, state): `@dataclass`, prefer `kw_only=True`;
  mutable defaults via `field(default_factory=...)`.
- Cross-boundary schemas (API req/resp, untrusted input, needs validation): `pydantic.BaseModel`.
  Don't hand-roll validators pydantic gives free.
- Don't mix the two for one concept. Pick per role, not per mood.

### STRUCTURE
- File > ~400 lines → extract a module. But a 30-line file holding one `_helper` called once
  is also wrong — merge it.
- No `sys.path.insert` hacks. Use package imports, PYTHONPATH, or
  `pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)`. One per repo.
- No function-local imports except documented circular-dependency breaks. Prefer
  `if TYPE_CHECKING:` for type-only cycle-breaking imports.
- One entry-point mechanism per repo (hydra/argparse/fire/`[project.scripts]`). `if __name__
  == "__main__"` only in entry-point modules, never library modules.
- Module-level side effects (env mutation, global state, `setup_root`, resolver registration)
  only in entry-point modules. Library imports must be side-effect-free.
- `__init__.py`: explicit `__all__`, no `from .x import *`, minimal re-exports.
- No dead code: unreachable branches, unused params, commented-out blocks, stale TODOs,
  "kept for later" stubs.

### ERROR HANDLING & MATURITY
- `assert` for internal invariants (shapes, device, dtype, preconditions you control) — fail
  fast, loud, consistently across sibling modules.
- `raise ValueError/RuntimeError/FileNotFoundError` for user input / environment errors.
- No bare `except:`. No `except Exception as e: pass`. Catch specific exceptions; if broad,
  log with a reason or re-raise.
- Maturity progression — this matters:
  - Prototyping: try/except around the call you don't yet trust is fine.
  - Mature: remove it. Replace `if x is not None and isinstance(x, ...)` chains with an
    assert or a direct call. Defensive scaffolding (dummy forwards to align AllReduce,
    logger enable/disable gymnastics, type-coercion loops, broad try/except swallowing
    errors) is DEBT, not safety. Trust your invariants; let failures surface as stack traces.
  - Review: every try/except and defensive `if` must answer "what breaks if I delete this?"
    If the answer is "nothing, it was for debugging" — delete it.
- A long-term workaround gets a one-line comment naming the constraint, not a ten-line block.

### LOGGING
- ONE logging library per repo. Don't mix stdlib `logging` and `loguru`. Configure once.
- One logger per module: `logging.getLogger(__name__)` or `from loguru import logger`.
- Terse English, no manual prefixes (level field says INFO/WARNING).
- `print` only for CLI output the end user reads (`--help`, visualization, `__main__` demo).
  Runtime info — even debug — goes through the logger.
- Distributed: guard rank-zero via logger config or a `RankedLogger`-style adapter, not
  scattered `if is_main:`. (Training loops keep theirs; don't spread the pattern.)
- f-string vs `%-style`: pick one per repo. loguru → f-strings idiomatic; stdlib hot path →
  `logger.info("Loading: %s", path)`.

### CONFIG & MAGIC VALUES
- No hardcoded URLs, absolute paths, hostnames, or unexplained magic numbers in source.
  Config file, env var, or named constant at module top.
- Empirical constants named + provenance comment: `DECODE_TOKS_PER_SEC = 6.7  # measured on H20`.
- One config mechanism per repo (Hydra / argparse+yaml / dataclass / env). Don't mix. Don't
  ship a yaml that's "documentation only" and never loaded.

### IMPORTS
- Group: stdlib / third-party / local, blank-line separated, alphabetical within group.
- No sys.path hacks (see STRUCTURE). No function-local imports (see STRUCTURE).
- Type-only cycle-breaking imports under `if TYPE_CHECKING:` with string quotes at use site.
- Do not use __init__.py for imports, we import every file as package from root path like from xxx.yy.zzz import kkk.

### TOOLING
- Prefer a configured linter/formatter (ruff/black/isort) + type checker (pyright/mypy). If
  none configured, at minimum run `python -m py_compile` / `ruff check` before claiming done.
  A configured type checker is the higher bar — aim for it.
- New logic gets a test. A repo with zero tests is a smell; don't make it worse.

### WHEN REVIEWING YOUR OWN OUTPUT
- Single-call `_helper`/`_impl`? Inline it. <30-line single-helper file? Merge it.
- Interface/base class/registry with one implementation? Delete it until a second exists.
- try/except or `if isinstance / if x is not None` chain protecting no real failure mode?
  Delete it.
- Non-English text, ★ / `# P1` / `# [FIX]` marker, banner, manual log prefix? Flag it.
- File > ~400 lines that could split? Flag it.
- Hardcoded paths/URLs, sys.path hacks, function-local imports, bare excepts, untyped
  signatures, mixed `Optional`+`X|Y`, mutable defaults, comments restating code? Flag it.
- Every comment: would a domain reader still need it? No → delete.
- Every private function: called more than once? No → inline.

### 训练

训练代码应该分为train和trainer两个脚本，将训练有关的函数等内容封装到trainer里，train只负责调用trainer进行循环。

### 注释

不要在文件开头写注释，类定义后不要写超长注释解释各类元素，用简单的注释说明就行。
```

## Examples (do / don't)

### Comments — don't
```python
# ★ Disable loralib auto-merge, switch to rsLoRA scaling = alpha/sqrt(r)
# merge_weights=False -> train()/eval() won't auto-merge; scaling=0 reliably emulates base model
# ==================== Reference Policy LoRA Snapshot (trust region, mirrors upstream) ====================
# ref = policy snapshot from N steps ago (not disable_lora=base): KL(policy||recent) trust region,
# does not cap improvements that need to move far from base...
```

### Comments — do
```python
# Disable loralib auto-merge; use rsLoRA scaling = alpha/sqrt(r).
# --- reference policy LoRA snapshot (trust region) ---
# ref = policy snapshot from N steps ago; KL(policy||recent) does not cap base-divergent gains.
```

### Simplicity — don't 随意加私有方案，我看到函数前面有下划线就会躯体化发作
```python
def _load_json(path: Path) -> dict:        # called once
    with open(path, encoding="utf-8") as f:
        return json.load(f)
def _parse_config(raw: dict) -> Config:    # called once
    return Config(**raw)
def _validate(c: Config) -> None:          # called once and have underline preifx (I don't like this prefix)
    assert c.lr > 0
def main():
    c = _validate(_parse_config(_load_json(cfg_path)))
```

### Simplicity — do
```python
def main():
    with open(cfg_path, encoding="utf-8") as f:
        c = Config(**json.load(f))
    assert c.lr > 0
```

### Maturity — don't (mature code still carrying debugging scaffolding)
```python
def encode(tokens, model):
    try:
        out = model(tokens)
    except Exception as e:
        logger.warning(f"model failed: {e}")   # silent zero on failure
        out = torch.zeros_like(tokens, dtype=torch.float32)
    if out is None:                              # model never returns None
        out = torch.zeros_like(tokens, dtype=torch.float32)
    if not isinstance(out, torch.Tensor):        # model returns Tensor
        out = torch.tensor(out)
    return out
```

### Maturity — do
```python
def encode(tokens: torch.Tensor, model: nn.Module) -> torch.Tensor:
    return model(tokens)
```

### Logging — don't
```python
logger.info(f"[Info] loading model: {path} (~43s)")
logger.warning("[Warn] judge unavailable! reward will be all 0, no training signal.")
```

### Logging — do
```python
logger.info(f"Loading model: {path} (~43s)")
logger.warning("Judge unavailable; reward will be 0, no training signal.")
```

### Typing — don't
```python
def compute_log_probs(model, tokens, labels, mask, config=None):
def encode(self, tokenizer: any, ignore_loss_tokens: list[str] = []):
```

### Typing — do
```python
def compute_log_probs(
    model: nn.Module,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    config: "ModelConfig | None" = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

def encode(
    self,
    tokenizer: PreTrainedTokenizerFast,
    ignore_loss_tokens: list[str] | None = None,   # None sentinel, not []
) -> EncodedMessage:
    if ignore_loss_tokens is None:
        ignore_loss_tokens = []
```

### Config — don't
```python
url = config.get("url", "http://10.0.0.1:8990/evaluate")   # hardcoded URL
tmp_dir = "/mnt/shared/rollout_tmp"                          # hardcoded absolute path
est = max_tokens / 6.7                                       # unexplained magic number
```

### Config — do
```python
url = config["judge_url"]                       # from yaml/env, not a source default
tmp_dir = config["tmp_dir"]
DECODE_TOKS_PER_SEC = 6.7  # measured on H20; used to estimate rollout time
est = max_tokens / DECODE_TOKS_PER_SEC
```

### Imports — don't
```python
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from helper import do_thing           # script-style import
def main():
    import logging                   # function-local import
```

### Imports — do
```python
import logging
from mypkg.workers.helper import do_thing
# large project: pyrootutils.setup_root(__file__, indicator=".project-root",
# pythonpath=True) at the top of the entry-point module only, then normal package imports.
```

## How to use

- **As agent instructions**: paste the entire `## Prompt` fenced block into the system prompt
  or project instructions file.
- **As a repo convention**: commit to repo root, reference in PR templates / review checklists.
- **As a review rubric**: the "when reviewing your own output" section is a checklist for any
  diff.

Domain-agnostic (web, ML, CLI, library all apply). Examples use ML idioms but the rules don't
assume ML.
