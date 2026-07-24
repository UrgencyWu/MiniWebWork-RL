# M1.1 Command Log

## Phase: M1.1 Deterministic Procurement Environment

### DB Init & Validation
```bash
python -m miniwebwork.cli init-db          # Create schema, seed 6 suppliers + 24 products
python -m miniwebwork.cli validate-seed    # All valid, 0 errors
python -m miniwebwork.cli validate-tasks   # All 15 tasks valid, unique answers confirmed
python -m miniwebwork.cli status           # Verify row counts
```

### Tests
```bash
python -m pytest tests/ -q                 # 62 passed, 0 failed
```

### Slurm E2E
```bash
sbatch scripts/slurm/m1_1_procurement_e2e.sbatch
# Job 928: FAILED (form submission issue)
# Job 929: FAILED (form submission issue)
# Job 930: COMPLETED (0:0), 18s — URL-based navigation fix
```

### Environment Export
```bash
conda env export -n miniwebwork --no-builds > environment.final.yml
pip freeze > requirements.final.txt
pip check  # vllm+transformers warning (pre-existing, non-blocking)
```
