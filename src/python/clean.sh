 
#!/bin/bash

# clean-python.sh
# Cleans Python project directories of caches, build files, and temp data.

set -e

echo "🧹 Cleaning Python project..."

# Remove common cache & build directories
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null
find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null
find . -type d -name "build" -exec rm -rf {} + 2>/dev/null
find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null
find . -type d -name ".tox" -exec rm -rf {} + 2>/dev/null

# Remove virtual environments if present
for venv_dir in venv .venv env; do
  if [ -d "$venv_dir" ]; then
    echo "Removing virtual environment: $venv_dir"
    rm -rf "$venv_dir"
  fi
done

# Remove leftover files
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
find . -type f -name ".coverage" -delete 2>/dev/null
find . -type f -name ".DS_Store" -delete 2>/dev/null

echo "✅ Python project cleaned."
