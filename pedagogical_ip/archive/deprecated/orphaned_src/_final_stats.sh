#!/bin/bash
cd /mnt/f/SCAI/Learning-agent/pedagogical_ip

echo "=== src/ remaining modules ==="
find src -name '*.py' -not -path '*__pycache__*' -not -name '__init__.py' | wc -l

echo "=== scripts/ remaining ==="
ls scripts/*.py 2>/dev/null | wc -l

echo "=== archive/ breakdown ==="
for d in archive/*/; do
    count=$(find "$d" -type f | wc -l)
    echo "  $d -> $count files"
done

echo "=== archive/ total files ==="
find archive -type f | wc -l

echo "=== results/ remaining ==="
find results -type f | wc -l

echo "=== tests/ ==="
find tests -name '*.py' -not -path '*__pycache__*' | wc -l
