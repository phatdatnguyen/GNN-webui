"""Reproduce interpreter instability independently of the app and ML libraries.

Run with the project's Python 3.11 interpreter. Only Python source from the
installed Jinja2 and MarkupSafe packages is copied to a temporary directory.
The worker uses isolated mode without site-packages or third-party C extensions.
"""

import argparse
import faulthandler
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def run_worker(source_directory, seconds):
    faulthandler.enable(all_threads=True)
    sys.path.insert(0, source_directory)
    import jinja2
    import markupsafe

    assert "markupsafe._speedups" not in sys.modules
    print(f"Python: {sys.version}", flush=True)
    print(f"Template source: {jinja2.__file__}", flush=True)
    print(f"Escaping implementation: {markupsafe._escape_inner.__module__}", flush=True)
    environment = jinja2.Environment()
    template = """{% macro row(item) %}<tr><td>{{ item.id }}</td><td>{{ item.label|escape }}</td></tr>{% endmacro %}
{% for group in groups %}<table>{% for item in group %}{{ row(item) }}{% endfor %}</table>{% endfor %}
{% if enabled %}{{ values|select('odd')|list|join(', ') }}{% endif %}"""
    context = dict(enabled=True, values=list(range(40)), groups=[
        [dict(id=i, label="label <&>" + str(i)) for i in range(20)] for _ in range(4)])
    start = time.monotonic()
    count = 0
    while time.monotonic() - start < seconds:
        result = environment.from_string(template + "\n" + str(count)).render(context)
        assert "label &lt;&amp;&gt;" in result
        count += 1
    print(f"Completed {count} iterations in {time.monotonic() - start:.1f}s. "
          "A passing short run does not establish long-term stability.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=20, help="Duration from 1 to 120 seconds (default 20).")
    parser.add_argument("--worker-source", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        parser.error("Use the existing Python 3.11 environment for this diagnostic.")
    if not 1 <= args.seconds <= 120:
        parser.error("--seconds must be between 1 and 120.")
    if args.worker_source:
        run_worker(args.worker_source, args.seconds)
        return 0
    with tempfile.TemporaryDirectory(prefix="gnn-python311-check-") as directory:
        for package in ("jinja2", "markupsafe"):
            source = Path(importlib.metadata.distribution(package).locate_file(package))
            target = Path(directory) / package
            for file in source.rglob("*.py"):
                destination = target / file.relative_to(source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(file, destination)
        environment = os.environ.copy()
        for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "MKLROOT"):
            environment.pop(name, None)
        command = [sys.executable, "-I", "-S", str(Path(__file__).resolve()),
                   "--worker-source", directory, "--seconds", str(args.seconds)]
        print("Running an isolated Python-only check; no model or GPU operations.", flush=True)
        try:
            result = subprocess.run(command, env=environment, timeout=args.seconds + 15)
        except subprocess.TimeoutExpired:
            print("Diagnostic timed out.", file=sys.stderr)
            return 124
        if result.returncode < 0:
            print(f"Interpreter terminated by signal {-result.returncode}.", file=sys.stderr)
            return 128 - result.returncode
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
