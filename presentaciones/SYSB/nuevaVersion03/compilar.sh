#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
for archivo in semanas-01-05.qmd semanas-06-10.qmd semanas-11-15.qmd; do
  quarto render "$archivo" --to revealjs
  quarto render "$archivo" --to beamer
done
