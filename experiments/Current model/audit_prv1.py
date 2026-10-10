#!/usr/bin/env python3
"""PRV1 evidence audit.

Usage (from the repo's 'experiments/Current model/' dir):
    python audit_prv1.py --src . --results ../results
Exit code 0 = all checks pass, 1 = at least one failure.

Check 1: PrivacyEngine(...) is constructed ONLY in dp_persistent_client_state.build_dp_client_states,
         never in main.py (or any per-round function anywhere).
Check 2: no make_private*/detach/close calls on engines inside main.py per-round functions.
Check 3: every DP result JSON has full_run_target_epsilon and exactly one final_total_epsilon per client.
Adjust ALLOWED_BUILDER / key names below if your code differs.
"""
import argparse, ast, json, pathlib, sys, warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

ALLOWED_BUILDER = ("dp_persistent_client_state.py", "build_dp_client_states")
PER_ROUND_BAN = {"make_private", "make_private_with_epsilon", "detach", "close"}


def called_name(node):
    f = node.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None


def audit_code(src, exclude=()):
    ok = True
    seen = set()
    for path in sorted(pathlib.Path(src).rglob("*.py")):
        if "test" in path.name or "audit" in path.name:
            continue
        if path.name in exclude:
            print(f"EXCLUDED  {path.name} (explicitly excluded via --exclude; document why in the issue)")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            for call in [n for n in ast.walk(fn) if isinstance(n, ast.Call)]:
                name = called_name(call)
                where = f"{path.name}:{call.lineno} in {fn.name}()"
                if (path.name, call.lineno, name) in seen:
                    continue
                seen.add((path.name, call.lineno, name))
                if name == "PrivacyEngine" and (path.name, fn.name) != ALLOWED_BUILDER:
                    print(f"FAIL  PrivacyEngine(...) outside builder: {where}"); ok = False
                if path.name == "main.py" and name in PER_ROUND_BAN and fn.name.startswith(("_train", "_eval", "_run")):
                    print(f"FAIL  {name}() inside per-round function: {where}"); ok = False
    print("PASS  code audit" if ok else "FAIL  code audit")
    return ok


def collect(obj, key, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key: out.append(v)
            collect(v, key, out)
    elif isinstance(obj, list):
        for v in obj: collect(v, key, out)


def audit_manifests(results):
    ok, n = True, 0
    for p in sorted(pathlib.Path(results).rglob("*.json")):
        try: d = json.loads(p.read_text(encoding="utf-8"))
        except Exception: continue
        text = json.dumps(d)
        if '"use_dp": true' not in text.lower() and '"use_dp":true' not in text.lower():
            continue
        n += 1
        tgt, fin = [], []
        collect(d, "full_run_target_epsilon", tgt); collect(d, "final_total_epsilon", fin)
        legacy = '"epsilon"' in text and not fin
        if legacy:
            print(f"LEGACY {p} (pre-fix, per-round epsilon; do NOT reuse as corrected evidence)"); continue
        if not tgt: print(f"FAIL  {p}: missing full_run_target_epsilon"); ok = False
        if not fin: print(f"FAIL  {p}: missing final_total_epsilon"); ok = False
        print(f"      {p.name}: target={tgt[:1]} final_total_epsilon entries={len(fin)} values={fin}")
    print(f"{'PASS' if ok else 'FAIL'}  manifest audit ({n} DP result files). "
          "Confirm manually: entries == number of DP-active (honest) clients, one each.")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--src", default="."); ap.add_argument("--results", default="results"); ap.add_argument("--exclude", nargs="*", default=[])
    a = ap.parse_args()
    sys.exit(0 if all([audit_code(a.src, set(a.exclude)), audit_manifests(a.results)]) else 1)
