#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Grammar filter for accepted speculative blocks (structured output + MTP)

Structured-output requests (JSON / tool-calling with the qwen3 reasoning
parser) intermittently died with

    ERROR [backend_xgrammar.py] Failed to advance FSM ... for tokens 271
    ERROR [scheduler.py] Unexpected: grammar rejected tokens [248069, 271, 90]

(248069 = the  end marker, 271 = "\n\n", 90 = "{"). This is the open
upstream bug vllm#53181 (follow-up to #46118 / #44006 / #52767): an accepted
speculative block can cross the reasoning boundary or the grammar completion
and carry a trailing token the FSM rejects, because the masks that sampled it
saw the matcher state *before* the transition. The scheduler then kills the
request (FINISHED_ERROR -> HTTP 500).

The fix is the core of upstream PR vllm#52452, adapted to this tree after the
#48200 refactor: this tree has no should_advance/trim_reasoning_for_advance,
but it has `_get_constraint_start` and a non-advancing `validate_tokens`, so
the filter reuses them. Before the scheduler commits a block,
`filter_speculative_grammar_tokens` validates it with `validate_tokens` (the
matcher is rolled back, the state is untouched), commits only the longest
grammar-valid prefix, and keeps the rejected suffix schedulable
(`num_computed_tokens` and `num_output_placeholders` are rolled back), so the
suffix is resampled on the next step with the mask now engaged. The request
no longer dies on a grammar rejection.

Deliberately untouched (safety):
  * a block when the grammar has already terminated - the commit path
    (`accept_tokens` stop-at-termination, vllm#52805, already in this tree)
    handles the tail, and trimming could drop the stop token;
  * a block ending with a stop token - the stop token is not part of the
    grammar and the request stops on it.

With `speculative_config` unset or an unstructured request the filter returns
immediately, and a single-token block is never trimmed, so non-speculative
traffic is untouched. Measured on 2x CMP 170HX: 10/10 structured-output
acceptance requests JSON-OK, 0 FSM rejection errors in the engine log (was 7
in the equivalent pre-patch window).

usage: python 17_xgrammar_spec_filter.py install|check|remove [<vllm-tree-root>]
       install is idempotent and writes <file>.bak-17_xgrammar_spec_filter
       next to every file it changes.
"""

import shutil
import sys
from pathlib import Path

TAG = "17_xgrammar_spec_filter"

# (relative path inside the vLLM tree, text before, text after)
EDITS = (
    (
        "vllm/v1/structured_output/__init__.py",
        # ---- before (StructuredOutputManager, end of validate_tokens)
        """        prefix = spec_tokens[:constraint_start]
        validated = grammar.validate_tokens(spec_tokens[constraint_start:])
        return prefix + validated

    def grammar_bitmask(""",
        # ---- after (new method inserted before grammar_bitmask)
        """        prefix = spec_tokens[:constraint_start]
        validated = grammar.validate_tokens(spec_tokens[constraint_start:])
        return prefix + validated

    def filter_speculative_grammar_tokens(
        self, request: "Request", new_token_ids: list[int]
    ) -> tuple[list[int], int]:
        # PATCH #18 (xgrammar_spec_filter): validate an accepted speculative
        # block before the scheduler commits it (backport of upstream
        # vllm#52452, fixes vllm#53181 / #46118). An accepted block can cross
        # the reasoning boundary or the grammar completion and carry a
        # trailing token the grammar rejects, because the masks that sampled
        # it saw the FSM state before the transition. validate_tokens is
        # non-advancing (it rolls the matcher back), so it reports the
        # longest grammar-valid prefix without disturbing the state that the
        # scheduler's normal commit path advances. Returns (safe_to_commit,
        # num_rejected); the rejected suffix stays schedulable and is
        # resampled next step with the mask now engaged.
        if self.vllm_config.speculative_config is None:
            return new_token_ids, 0
        if len(new_token_ids) < 2 or not request.use_structured_output:
            return new_token_ids, 0
        structured_request = request.structured_output_request
        if structured_request is None:
            return new_token_ids, 0
        grammar = structured_request.grammar
        if not isinstance(grammar, StructuredOutputGrammar):
            return new_token_ids, 0
        # A terminated grammar is handled by the commit path (accept_tokens
        # stops at termination); filtering here could drop the trailing stop
        # token.
        if grammar.is_terminated():
            return new_token_ids, 0
        # Never trim a block that ends with a stop token: the stop token is
        # not part of the grammar and the request stops on it.
        stop_ids = set()
        if request.sampling_params is not None:
            stop_ids = set(request.sampling_params.all_stop_token_ids or ())
        if stop_ids and new_token_ids[-1] in stop_ids:
            return new_token_ids, 0
        stripped = strip_speculative_padding(new_token_ids)
        if len(stripped) < 2:
            return new_token_ids, 0
        valid = self.validate_tokens(request, stripped)
        rejected = len(stripped) - len(valid)
        if rejected <= 0:
            return new_token_ids, 0
        return valid, rejected

    def grammar_bitmask(""",
    ),
    (
        "vllm/v1/core/sched/scheduler.py",
        # ---- before (update_from_output, commit path)
        """            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids, is_stale=output_is_stale
                )""",
        # ---- after (pre-commit grammar filter)
        """            # Check for stop and update request status.
            if new_token_ids:
                # PATCH #18 (xgrammar_spec_filter): validate an accepted
                # speculative block before it is committed. An accepted block
                # can cross the reasoning boundary or the grammar completion
                # and carry a trailing token the grammar rejects; the masks
                # that sampled it saw the FSM state before the transition.
                # Commit only the grammar-valid prefix and keep the rejected
                # suffix schedulable (it is resampled next step with the mask
                # now engaged) instead of terminating the request (backport of
                # upstream vllm#52452, fixes vllm#53181 / #46118).
                if not output_is_stale:
                    new_token_ids, num_grammar_rejected = (
                        self.structured_output_manager.filter_speculative_grammar_tokens(
                            request, new_token_ids
                        )
                    )
                    if num_grammar_rejected > 0:
                        if request.num_computed_tokens > 0:
                            request.num_computed_tokens -= num_grammar_rejected
                        if request.num_output_placeholders > 0:
                            request.num_output_placeholders -= num_grammar_rejected
                if new_token_ids:
                    new_token_ids, stopped = self._update_request_with_output(
                        request, new_token_ids, is_stale=output_is_stale
                    )""",
    ),
)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cmd = args[0] if args and args[0] in ("install", "check", "remove") else "check"
    rest = args[1:] if args and args[0] in ("install", "check", "remove") else args
    root = Path(rest[0]).resolve() if rest else Path(__file__).resolve().parent.parent
    if not (root / "vllm").is_dir():
        print(f"not a vLLM tree: {root} (pass its root as an argument)", file=sys.stderr)
        return 2

    bad = []
    for rel, old, new in EDITS:
        path = root / rel
        if not path.is_file():
            print(f"  MISSING  {rel}")
            bad.append(rel)
            continue
        text = path.read_text(encoding="utf-8")
        if new in text:
            print(f"  already  {rel}")
            continue
        if old not in text:
            print(f"  ANCHOR?  {rel} - the before text is not there")
            bad.append(rel)
            continue
        if cmd == "check":
            print(f"  would    {rel}")
            continue
        bak = path.with_name(path.name + ".bak-" + TAG)
        if not bak.is_file():
            shutil.copy2(path, bak)
        path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="")
        print(f"  patched  {rel}")

    if cmd == "remove":
        done = set()
        for rel, _old, _new in EDITS:
            if rel in done:
                continue
            done.add(rel)
            path = root / rel
            bak = path.with_name(path.name + ".bak-" + TAG)
            if bak.is_file():
                shutil.copy2(bak, path)
                print(f"  restored {rel} <- {bak.name}")
            else:
                print(f"  no backup for {rel} - was this ever installed here?")
                bad.append(rel)
        return 1 if bad else 0

    if bad and cmd == "install":
        print(f"WARNING: {len(bad)} edit(s) did not apply", file=sys.stderr)
        return 1
    if cmd == "check":
        print("check only, nothing written" if not bad
              else f"check: {len(bad)} edit(s) do not match this patch")
        return 1 if bad else 0
    print(f"{TAG}: {len(EDITS)} edit(s) in place")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
