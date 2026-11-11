"""Outside-truth check: run Python sources in the real libraries and compare
each shape and each error of `ndim-lsp check --shapes` with what happened.

    NDIM_DUMP_SOURCES=/tmp/ndim-oracle/src cargo test --lib
    cargo build --release
    oracle/.venv/bin/python oracle/check.py /tmp/ndim-oracle/src

A shape of the tool must equal each shape the name had on that line. An error
of the tool on a line that ran to its end is a false error. "Unknown" from the
tool always passes. A function runs with one array for each jaxtyping
parameter, with a different prime size for each dim name.
"""

import ast
import json
import math
import os
import re
import signal
import subprocess
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp
import torch
import equinox as eqx
import einops
import jaxtyping

ROOT = Path(__file__).resolve().parent.parent
TOOL = os.environ.get("NDIM_LSP", str(ROOT / "target/release/ndim-lsp"))
PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
SHAPE_WORDS = ("shape", "broadcast", "size", "dimension", "dim ", "axis", "rank", "mismatch")


def prelude():
    names = {name: getattr(jaxtyping, name) for name in dir(jaxtyping) if name[0].isupper()}
    names.update(
        np=np, numpy=np, jnp=jnp, jax=jax, torch=torch, nn=torch.nn, F=torch.nn.functional,
        eqx=eqx, equinox=eqx, einops=einops, Tensor=torch.Tensor, math=math,
        jr=jax.random, jrandom=jax.random, __name__="oracle_probe",
    )
    return names


def tool_output(directory):
    out = subprocess.run([TOOL, "check", ".", "--shapes"], cwd=directory, capture_output=True, text=True).stdout
    shapes, errors = defaultdict(list), defaultdict(list)
    for line in out.splitlines():
        if m := re.match(r"\./?(.+?\.py):(\d+): ([A-Za-z_]\w*): \[(.*)\]$", line) or re.match(
            r"(.+?\.py):(\d+): ([A-Za-z_]\w*): \[(.*)\]$", line
        ):
            dims = [d.strip() for d in split_dims(m[4])]
            shapes[m[1].removeprefix("./")].append((int(m[2]), m[3], dims))
        elif m := re.match(r"(.+?\.py):(\d+):\d+: error: (.*)$", line):
            errors[m[1].removeprefix("./")].append((int(m[2]), m[3]))
    return shapes, errors


def split_dims(text):
    parts, depth, current = [], 0, ""
    for ch in text:
        depth += ch in "([{"
        depth -= ch in ")]}"
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    return parts + [current] if current.strip() else parts


def dim_value(token, sizes):
    """The size of one annotation dim, with new names bound to primes."""
    token = token.lstrip("#")
    if token.isdigit():
        return int(token)
    for name in re.findall(r"[A-Za-z_]\w*", token):
        if name not in sizes:
            sizes[name] = PRIMES[len(sizes) % len(PRIMES)]
    try:
        value = eval(token, {"__builtins__": {}}, dict(sizes))
        return int(value) if float(value) == int(value) and value > 0 else None
    except Exception:
        return None


def make_array(library, dtype, shape):
    kind = {"Int": "int32", "UInt": "int32", "Integer": "int32", "Bool": "bool"}.get(dtype.rstrip("0123456789"), "float32")
    if library == "torch":
        return torch.ones(shape).to(getattr(torch, kind))
    if library == "numpy":
        return np.ones(shape, dtype=kind)
    return jnp.ones(shape, dtype=kind)


def argument(annotation, name, default, source, sizes):
    """A value for one parameter, or NotImplemented when none is safe."""
    if annotation is None:
        return ... if default else NotImplemented
    text = ast.unparse(annotation)
    if isinstance(annotation, ast.Subscript) and isinstance(annotation.slice, ast.Tuple) and len(annotation.slice.elts) == 2:
        holder, spec = annotation.slice.elts
        if isinstance(spec, ast.Constant) and isinstance(spec.value, str):
            holder = ast.unparse(holder)
            shape = []
            for token in spec.value.split():
                if token == "..." or token.lstrip("#_").startswith("*"):
                    shape.append(2)
                elif token == "_":
                    shape.append(3)
                else:
                    value = dim_value(token.split("=")[-1] if "=" in token else token, sizes)
                    if value is None:
                        return NotImplemented
                    shape.append(value)
            library = "torch" if "Tensor" in holder else "numpy" if "ndarray" in holder else None
            if library is None:
                library = "torch" if "torch" in source and "jax" not in source and "jnp" not in source else "jax"
            return make_array(library, ast.unparse(annotation.value).split(".")[-1], tuple(shape))
    if text == "int":
        return sizes.get(name, 2)
    if text == "float":
        return 0.5
    if text == "bool":
        return True
    if "PRNGKey" in text or name == "key":
        return jax.random.PRNGKey(0)
    return ... if default else NotImplemented


class Trace:
    def __init__(self, filename):
        self.filename = filename
        self.seen = defaultdict(set)  # (line, name) -> shapes
        self.ints = defaultdict(dict)  # line -> int locals
        self.done = set()  # lines that ran to their end
        self.raised = set()
        self.last = {}

    def __call__(self, frame, event, arg):
        if frame.f_code.co_filename != self.filename:
            return None
        return self.local

    def local(self, frame, event, arg):
        key = id(frame)
        if event == "exception":
            self.raised.add(frame.f_lineno)
        if event in ("line", "return"):
            previous = self.last.get(key)
            if previous is not None and previous not in self.raised:
                self.done.add(previous)
                for name, value in list(frame.f_locals.items()):
                    shape = getattr(value, "shape", None)
                    if isinstance(shape, tuple) and all(isinstance(d, int) for d in shape):
                        self.seen[(previous, name)].add(tuple(int(d) for d in shape))
                    elif type(value) is int:
                        self.ints[previous][name] = value
            self.last[key] = frame.f_lineno if event == "line" else None
        return self.local


class Timeout(Exception):
    pass


def alarm(*_):
    raise Timeout()


def run_file(path):
    try:
        source = path.read_text()
    except UnicodeDecodeError:
        return None
    filename = str(path)
    trace = Trace(filename)
    result = {"functions": 0, "skipped": 0, "raised": [], "bindings": {}}
    try:
        tree = ast.parse(source)
        code = compile(tree, filename, "exec")
    except SyntaxError:
        return None
    namespace = prelude()
    signal.signal(signal.SIGALRM, alarm)

    def guarded(call, where):
        signal.alarm(5)
        quiet = open(os.devnull, "w")
        stdout, stderr, sys.stdout, sys.stderr = sys.stdout, sys.stderr, quiet, quiet
        sys.settrace(trace)
        try:
            call()
        except Timeout:
            result["raised"].append((where, "timeout"))
        except BaseException as error:  # noqa: BLE001 - a probe can raise anything
            line = None
            tb = error.__traceback__
            while tb:
                if tb.tb_frame.f_code.co_filename == filename:
                    line = tb.tb_lineno
                tb = tb.tb_next
            result["raised"].append((line, f"{type(error).__name__}: {str(error)[:160]}"))
        finally:
            sys.settrace(None)
            signal.alarm(0)
            sys.stdout, sys.stderr = stdout, stderr
            quiet.close()

    guarded(lambda: exec(code, namespace), None)
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.decorator_list:
            continue
        function = namespace.get(node.name)
        if not callable(function):
            continue
        sizes, values = {}, []
        parameters = node.args.posonlyargs + node.args.args
        defaults = [None] * (len(parameters) - len(node.args.defaults)) + node.args.defaults
        # Bind the dim names of all annotations first, so an `n: int` gets the size of `n`.
        for parameter in parameters:
            argument(parameter.annotation, parameter.arg, True, source, sizes)
        for parameter, default in zip(parameters, defaults):
            values.append(argument(parameter.annotation, parameter.arg, default is not None, source, sizes))
        while values and values[-1] is ...:
            values.pop()
        if any(value is NotImplemented or value is ... for value in values) or node.args.kwonlyargs or node.args.vararg:
            result["skipped"] += 1
            continue
        result["functions"] += 1
        result["bindings"][(node.lineno, node.end_lineno)] = sizes
        guarded(lambda: function(*values), node.lineno)
    returns = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.returns is not None]
    result["annotated"] = r"\b(" + "|".join(map(re.escape, returns)) + r")\b" if returns else r"$^"
    result["trace"] = trace
    return result


def dims_agree(dims, shape, env):
    """True, False, or None when a dim has a name with no known value."""
    if any(d == "..." or d.lstrip("#_").startswith("*") for d in dims):
        return None
    if len(dims) != len(shape):
        return False
    unknown = False
    for dim, size in zip(dims, shape):
        if dim == "?":
            continue
        try:
            value = eval(dim.replace("/", "/"), {"__builtins__": {}, "min": min, "max": max, "ceil": math.ceil, "floor": math.floor}, env)
        except Exception:
            unknown = True
            continue
        if float(value) != float(size):
            return False
    return None if unknown else True


def main():
    directory = Path(sys.argv[1]).resolve()
    shapes, errors = tool_output(directory)
    os.chdir("/tmp")
    totals = defaultdict(int)
    wrong, false_errors, missed = [], [], []
    for path in sorted(directory.rglob("*.py")):
        if ".venv" in path.parts:
            continue
        key = str(path.relative_to(directory))
        result = run_file(path)
        totals["files"] += 1
        if result is None:
            totals["files_not_python"] += 1
            continue
        trace = result["trace"]
        totals["functions_run"] += result["functions"]
        totals["functions_skipped"] += result["skipped"]
        lines = path.read_text().splitlines()
        text = lambda line: lines[line - 1].strip() if 0 < line <= len(lines) else ""

        def env_at(line):
            env = {}
            for (start, end), sizes in result["bindings"].items():
                if start <= line <= end:
                    env.update(sizes)
            env.update(trace.ints.get(line, {}))
            return env

        for line, name, dims in shapes.get(key, []):
            totals["tool_shapes"] += 1
            observed = trace.seen.get((line, name))
            if not observed:
                continue
            verdicts = [dims_agree(dims, shape, env_at(line)) for shape in observed]
            if any(verdict is False for verdict in verdicts) and re.search(result["annotated"], text(line)):
                # The tool trusts a return annotation. The body broke it.
                totals["shapes_from_a_broken_annotation"] += 1
            elif any(verdict is False for verdict in verdicts):
                totals["shapes_wrong"] += 1
                wrong.append({"file": key, "line": line, "code": text(line), "name": name, "tool": dims, "runtime": sorted(observed), "sizes": env_at(line)})
            elif all(verdict is True for verdict in verdicts):
                totals["shapes_confirmed"] += 1
            else:
                totals["shapes_not_decided"] += 1
        for line, message in errors.get(key, []):
            totals["tool_errors"] += 1
            if line in trace.done:
                totals["errors_false"] += 1
                false_errors.append({"file": key, "line": line, "code": text(line), "tool": message})
            elif any(where == line for where, _ in result["raised"]):
                totals["errors_confirmed"] += 1
        tool_lines = {line for line, _ in errors.get(key, [])}
        for where, message in result["raised"]:
            if where and where not in tool_lines and any(word in message.lower() for word in SHAPE_WORDS) and "Error" in message:
                if not message.startswith(("NameError", "AttributeError", "ImportError", "ModuleNotFoundError")):
                    totals["runtime_shape_errors_without_tool_error"] += 1
                    missed.append({"file": key, "line": where, "code": text(where), "runtime": message})
    report = {"totals": dict(totals), "wrong_shapes": wrong, "false_errors": false_errors, "missed_errors": missed}
    Path("/tmp/ndim-oracle").mkdir(exist_ok=True)
    Path("/tmp/ndim-oracle/report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(dict(totals), indent=1))
    print(f"wrong shapes: {len(wrong)}, false errors: {len(false_errors)}  ->  /tmp/ndim-oracle/report.json")
    return 1 if wrong or false_errors else 0


if __name__ == "__main__":
    sys.exit(main())
