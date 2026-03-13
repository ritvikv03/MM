"""Enforce: no synthetic data anywhere in the backend."""
import ast
import pathlib


def _python_sources():
    root = pathlib.Path("src")
    return list(root.rglob("*.py"))


def test_no_build_stub_functions():
    """No function named _build_stub_* should exist in src/."""
    violations = []
    for path in _python_sources():
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_build_stub"):
                    violations.append(f"{path}::{node.name}")
    assert not violations, f"Stub functions found: {violations}"


def test_no_use_real_data_env_var():
    """USE_REAL_DATA env var must not exist anywhere in src/."""
    violations = []
    for path in _python_sources():
        if "USE_REAL_DATA" in path.read_text():
            violations.append(str(path))
    assert not violations, f"USE_REAL_DATA found in: {violations}"
