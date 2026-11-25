#!/bin/bash
# Full gate reading for prereg-precedent-v1. Reads texts only. Resumable.
cd /archive/experiments/precedent-2026-11-25/stage0
set -a; . ~/.primer.env; set +a
export GATE_LLM_BASE_URL=https://ellm.nrp-nautilus.io/v1 GATE_LLM_TOKEN="$NRP_LLM_TOKEN"
while [ ! -f pilot_agreement.json ]; do sleep 30; done
for pass in 1 2 3; do
  python3 read_gates.py --in cases.jsonl --out gates_gpt-oss.jsonl --model gpt-oss --workers 4 > gates_gpt-oss.log 2>&1 &
  python3 read_gates.py --in cases.jsonl --out gates_gemma.jsonl --model gemma --workers 4 > gates_gemma.log 2>&1 &
  wait
done
echo DONE > gates_full.done
