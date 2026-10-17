#!/usr/bin/env python3
"""Check API graphs against the target server without submitting GPU work."""
import argparse
import json
import os
from pathlib import Path
import urllib.request


def check(graph, catalog):
    errors = []
    for node_id, node in graph.items():
        kind = node["class_type"]
        if kind not in catalog:
            errors.append(f"{node_id}: missing class {kind}")
            continue
        schema = catalog[kind]
        if schema.get("api_node") or schema.get("python_module", "").startswith("comfy_api_nodes"):
            errors.append(f"{node_id}: partner/API node {kind} is forbidden")
        required = schema["input"].get("required", {})
        specs = required | schema["input"].get("optional", {})
        inputs = node["inputs"]
        for field in required.keys() - inputs.keys():
            errors.append(f"{node_id}.{field}: required input missing")
        for field, value in inputs.items():
            if field not in specs:
                errors.append(f"{node_id}.{field}: unknown input")
                continue
            spec = specs[field]
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int):
                source, slot = value
                if source not in graph or graph[source]["class_type"] not in catalog:
                    errors.append(f"{node_id}.{field}: missing linked source {source}")
                    continue
                outputs = catalog[graph[source]["class_type"]]["output"]
                if slot < 0 or slot >= len(outputs):
                    errors.append(f"{node_id}.{field}: invalid output slot {slot}")
                elif isinstance(spec[0], str) and spec[0] != outputs[slot] and spec[0] != "*":
                    errors.append(f"{node_id}.{field}: {outputs[slot]} does not match {spec[0]}")
                continue
            # Input files are placeholders in templates; model filenames are checked.
            if kind in {"LoadVideo", "LoadImage"}:
                continue
            options = spec[0] if isinstance(spec[0], list) else (spec[1].get("options") if len(spec) > 1 else None)
            if options:
                choices = [option["key"] if isinstance(option, dict) else option for option in options]
                if value not in choices:
                    errors.append(f"{node_id}.{field}: {value!r} absent from {choices}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=os.environ.get("COMFY_URL", "http://127.0.0.1:8191"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    with urllib.request.urlopen(args.server.rstrip("/") + "/object_info", timeout=30) as response:
        catalog = json.load(response)
    root = Path(__file__).resolve().parents[1]
    result = {}
    for path in sorted((root / "workflows").glob("*.json")):
        errors = check(json.loads(path.read_text()), catalog)
        result[path.name] = {"pass": not errors, "errors": errors}
    body = json.dumps({"server": args.server, "registered_classes": len(catalog), "workflows": result}, indent=2) + "\n"
    print(body, end="")
    if args.report:
        args.report.write_text(body)
    raise SystemExit(0 if all(item["pass"] for item in result.values()) else 1)


if __name__ == "__main__":
    main()
