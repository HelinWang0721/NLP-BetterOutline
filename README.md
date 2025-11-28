# Story Outline Generation: Baseline vs. Criteria-Integrated Comparison

This project compares two approaches to story outline generation using large language models.

## The Comparison

### Baseline (Minimal Prompting)
- Simple, direct instructions
- Basic requirements (characters, scenes, events)
- No quality criteria guidance
- No self-evaluation or refinement

### Improved (Criteria-Integrated)
- **15 narrative criteria** embedded in prompts as hints
- Guided by narrative theory (e.g., Rationality, Novelty, Suspense)
- No self-evaluation or refinement (pure generation comparison)

**Research Question**: Does embedding narrative criteria in prompts produce measurably better story outlines?

---

## Quick Start

1. **Configure API**: Edit `config.yaml`:
```yaml
api:
  api_key: "your-openrouter-key"
```

2. **Generate Outlines** (Baseline & Improved):
```bash
# Generates 5 random samples for each method
python gen-tasks.py --num 5
```
*Outputs: `baseline_outlines.json`, `improved_outlines.json`*

3. **Evaluate Outlines** (Multi-Model Voting):
```bash
# Evaluate baseline outlines
python vote.py baseline_outlines.json

# Evaluate improved outlines
python vote.py improved_outlines.json
```
*Outputs: `baseline_outlines_eval.json`, `improved_outlines_eval.json`*

4. **Compare Performance**:
```bash
python performance_comparison.py
```
*Outputs: `PERFORMANCE_REPORT.md`, `performance_comparison.png`*

---

## The 15 Narrative Criteria

1. **合理性** (Rationality) - Logical consistency
2. **新颖程度** (Novelty) - Originality and creativity
3. **悬念** (Suspense) - Tension and mystery
4. **反转和惊喜** (Twists & Surprises) - Unexpected developments
5. **期待感** (Anticipation) - Reader engagement
6. **目标** (Goals) - Clear character motivations
7. **读者偏好** (Reader Preferences) - Genre appeal
8. **设定复杂性** (Setting Complexity) - World-building depth
9. **情节复杂性** (Plot Complexity) - Narrative intricacy
10. **代入感** (Immersion) - Reader connection
11. **情感波动** (Emotional Impact) - Emotional range
12. **一致性** (Consistency) - Internal coherence
13. **相关度** (Relevance) - Thematic unity
14. **结局** (Ending) - Satisfying conclusion
15. **情节分配** (Plot Distribution) - Pacing balance

---

## Expected Results

Based on the methodology:
- **Quality improvement** from criteria integration
- **Strongest improvements**: Novelty, suspense, twists
- **Statistically significant** (p < 0.05)

---

## Project Structure

```
├── baseline_gen.py         # Minimal prompting generation
├── outline_gen.py          # Criteria-integrated generation
├── gen-tasks.py            # Batch generation script
├── vote.py                 # Multi-model evaluation script
├── performance_comparison.py  # Statistical analysis & reporting
├── criteria_analysis.py    # Weight optimization
├── gen-prompt.md           # Criteria prompts
└── config.yaml             # Configuration
```

---

## How It Works

### 1. Generation Phase
**Baseline**:
```python
prompts, messages, responses = baseline_gen.generate_baseline_outline(m_words, o_words)
```

**Improved**:
```python
# Uses prompts with embedded criteria hints
prompts, messages, responses = outline_gen.generate_outline_with_criteria(m_words, o_words)
```

### 2. Evaluation Phase
```python
# Multi-model voting (e.g., GPT-4o, Claude 3.5, etc.)
scores = vote.vote_all(outline_text)
# Returns scores for all 15 criteria
```

### 3. Analysis Phase
```python
# Compare baseline vs improved scores
python performance_comparison.py
```

---

## Methodology

### Generation
1. Generate outline pairs (baseline + improved) using the same keywords.
2. Both use the same API backend (OpenRouter) and base model.
3. **Baseline**: "Write a story about X..."
4. **Improved**: "Write a story about X, ensuring Rationality, Novelty, Suspense..."

### Evaluation
1. **Multi-model voting**: 4-5 LLMs independently score each outline.
2. **Blinded**: Evaluators don't know which method generated the outline.
3. **Scoring**: 0-100 points per criterion.

### Statistical Analysis
1. Calculate weighted scores.
2. Independent t-tests for significance.
3. Effect size calculation.

---

## Configuration

Edit `config.yaml` to customize:

- **API**: OpenRouter key and endpoint
- **Model**: Generation model selection
- **Criteria Weights**: Importance of each criterion
- **Evaluation Models**: Models used for voting (e.g., `openai/gpt-4o`, `anthropic/claude-3.5-sonnet`)

---

## Files Generated

- `baseline_outlines.json` - Raw baseline outlines
- `improved_outlines.json` - Raw improved outlines
- `baseline_outlines_eval.json` - Baseline outlines with evaluation scores
- `improved_outlines_eval.json` - Improved outlines with evaluation scores
- `PERFORMANCE_REPORT.md` - Detailed statistical report
- `performance_comparison.png` - Visualizations

