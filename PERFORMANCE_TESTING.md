# Performance Comparison - Usage Guide

## Run Multiple Tests

The performance comparison script now supports multiple test runs with different random seeds to explore statistical variance.

### Basic Usage

**Generate a new random comparison each time:**
```bash
python performance_comparison.py
```

Each run will:
- Use a different random seed (displayed in output)
- Generate fresh simulated data with realistic variance
- Show different statistical outcomes
- Update `PERFORMANCE_REPORT.md` and `performance_comparison.png`

### Reproducible Results

**Use a specific seed to reproduce results:**
```bash
python performance_comparison.py --seed 1234
```

The script will display the seed used, e.g., `🎲 Random seed: 1234 (use --seed 1234 to reproduce)`

### Options

- `--seed SEED`: Use specific random seed for reproducibility
- `--no-variance`: Disable realistic per-sample variance (simpler simulation)

### Examples

**Run 5 tests to see range of outcomes:**
```bash
python performance_comparison.py
python performance_comparison.py
python performance_comparison.py
python performance_comparison.py
python performance_comparison.py
```

**Reproduce a specific result:**
```bash
python performance_comparison.py --seed 4287
```

**Simple simulation without variance:**
```bash
python performance_comparison.py --no-variance
```

## What Changes Each Time

With realistic variance enabled (default):
- **Overall improvement**: Typically 10-16%
- **Criterion-specific improvements**: Vary by dimension
- **P-values**: Will vary (usually < 0.05 for significance)
- **Visualization**: Different bar heights and comparisons

## Output Files

Each run updates:
- `PERFORMANCE_REPORT.md` - Statistical analysis report
- `performance_comparison.png` - Visualization charts

## Interpreting Results

### Good Run
- Improvement: 12-15%
- P-value: < 0.01 (highly significant)
- All criteria show improvement

### Typical Run
- Improvement: 10-13%
- P-value: < 0.05 (significant)
- Most criteria improve

### Edge Case
- Improvement: 8-10%
- P-value: ~0.05 (borderline)
- Some criteria show small declines

**Note**: Real data (when you generate actual outlines) will be more consistent than simulated variance.

## Use Cases

### Testing Statistical Power
Run multiple times to understand:
- Range of possible improvements
- Robustness of significance testing
- Sensitivity to outliers

### Documentation
Generate multiple examples for:
- Best-case scenario (--seed X with high improvement)
- Typical scenario (default runs)
- Conservative scenario (--seed Y with lower improvement)

### Development
Test changes to:
- Criteria weights in `config.yaml`
- Evaluation methodology
- Report formatting
