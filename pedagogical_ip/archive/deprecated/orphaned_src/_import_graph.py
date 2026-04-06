"""Build import reference graph for all src/ modules."""
import os, re, sys
from collections import defaultdict

src = os.path.join(os.path.dirname(__file__), "..", "src")

# Collect all module basenames
all_modules = {}
for root, dirs, files in os.walk(src):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in files:
        if f.endswith(".py") and f != "__init__.py":
            rel = os.path.relpath(os.path.join(root, f), src).replace("\\", "/")
            mod_name = f[:-3]
            all_modules[mod_name] = rel

# For each file, find what it imports
imported_by = defaultdict(set)  # module -> set of files that import it
imports_from = defaultdict(set)  # file -> set of modules it imports

for mod_name, rel_path in all_modules.items():
    fpath = os.path.join(src, rel_path)
    try:
        with open(fpath, "r", encoding="utf-8", errors="replace") as fp:
            content = fp.read()
    except:
        continue
    
    # Find all import references to other src modules
    for other_name in all_modules:
        if other_name == mod_name:
            continue
        # Check for from .xxx import or from ..xxx.yyy import
        patterns = [
            f"from.*{other_name}.*import",
            f"import.*{other_name}",
        ]
        for pat in patterns:
            if re.search(pat, content):
                imported_by[other_name].add(rel_path)
                imports_from[rel_path].add(other_name)
                break

# Count references
print("module_name | file_path | imported_by_count | imported_by_files")
for mod_name in sorted(all_modules.keys()):
    rel = all_modules[mod_name]
    refs = imported_by.get(mod_name, set())
    ref_list = ",".join(sorted(refs)) if refs else "(none)"
    print(f"{mod_name}|{rel}|{len(refs)}|{ref_list}")
