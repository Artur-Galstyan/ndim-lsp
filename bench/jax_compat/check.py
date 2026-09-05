"""Check the Rust compatibility fixtures against the pinned, real JAX release.

Run with a disposable environment (see README.md). No repository files change.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp


def main():
    fixture = json.loads(Path(__file__).with_name("cases.json").read_text())
    if jax.__version__ != fixture["version"]:
        raise SystemExit(f"Expected JAX {fixture['version']}, got {jax.__version__}")

    for case in fixture["cases"]:
        namespace = {}
        exec(fixture["imports"], namespace)
        namespace.update(
            (name, jnp.ones(dims, dtype=jnp.int32))
            for name, dims in case.get("inputs", {}).items()
        )
        try:
            exec(f"{case['bind']} = {case['call']}", namespace)
        except (ValueError, TypeError, IndexError) as error:
            if not case.get("error"):
                raise AssertionError(f"{case['name']}: unexpected runtime error") from error
        else:
            assert not case.get("error"), f"{case['name']}: expected a runtime error"
            for name, expected in case.get("shapes", {}).items():
                value = namespace[name]
                actual = list(value.shape) if hasattr(value, "shape") else None
                assert actual == expected, f"{case['name']}: {name}: {actual} != {expected}"
    print(f"JAX {jax.__version__}: all {len(fixture['cases'])} runtime cases passed")


if __name__ == "__main__":
    main()
