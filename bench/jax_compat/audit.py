"""Compare explicit public exports in two released JAX wheels, without imports.

This is an inventory, not a claim that every exported object returns an array.
It does not detect new methods, signature changes, or changes in semantics.
"""

import argparse
import ast
import io
import json
import urllib.request
import zipfile


MODULES = [
    "jax/__init__.py",
    "jax/numpy/__init__.py",
    "jax/lax/__init__.py",
    "jax/nn/__init__.py",
    "jax/scipy/linalg.py",
    "jax/scipy/special.py",
    "jax/image/__init__.py",
    "jax/random.py",
]


def released_wheel(version):
    with urllib.request.urlopen(f"https://pypi.org/pypi/jax/{version}/json", timeout=30) as response:
        metadata = json.load(response)
    wheel = next(file for file in metadata["urls"] if file["filename"].endswith(".whl"))
    with urllib.request.urlopen(wheel["url"], timeout=60) as response:
        return zipfile.ZipFile(io.BytesIO(response.read()))


def public_exports(source):
    # JAX deliberately spells exports as `from ... import name as name`.
    return {
        alias.asname
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.asname and not alias.asname.startswith("_")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="0.9.2")
    parser.add_argument("--candidate", default="0.11.1")
    args = parser.parse_args()
    before = released_wheel(args.baseline)
    after = released_wheel(args.candidate)
    print(f"Public export audit: JAX {args.baseline} -> {args.candidate}")
    for module in MODULES:
        old = public_exports(before.read(module))
        new = public_exports(after.read(module))
        print(f"{module}\n  added: {', '.join(sorted(new - old)) or '(none)'}")
        print(f"  removed: {', '.join(sorted(old - new)) or '(none)'}")


if __name__ == "__main__":
    main()
