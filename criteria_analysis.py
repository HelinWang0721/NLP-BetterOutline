"""
Criteria Analysis and Mathematical Optimization
Academic analysis module for weight optimization and statistical evaluation
"""

import json
import yaml
import numpy as np
from typing import List, Dict, Tuple, Any
from scipy import stats
from scipy.optimize import minimize
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CriteriaAnalyzer:
    """
    Analyze story outline criteria and optimize weights mathematically
    """
    
    def __init__(self, config_path: str = "./config.yaml"):
        """Initialize analyzer"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.criteria_names = [
            "合理性", "新颖程度", "悬念", "反转和惊喜", "期待感",
            "目标", "读者偏好", "设定复杂性", "情节复杂性", "代入感",
            "情感波动", "一致性", "相关度", "结局", "情节分配"
        ]
        
        self.current_weights = list(self.config['criteria']['weights'].values())
    
    def load_evaluation_data(self, file_path: str) -> List[Dict]:
        """
        Load evaluation data from JSON file
        
        Args:
            file_path: Path to JSON file containing evaluations
            
        Returns:
            List of evaluation dictionaries
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def extract_scores_matrix(
        self,
        evaluation_data: List[Dict]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract scores into numpy matrix
        
        Args:
            evaluation_data: List of evaluation dicts
            
        Returns:
            Tuple of (scores_matrix, model_names)
            scores_matrix shape: (n_samples, n_criteria, n_models)
        """
        n_samples = len(evaluation_data)
        n_criteria = 15
        
        # Collect all model names
        all_models = set()
        for entry in evaluation_data:
            all_models.update(entry.keys())
        model_names = sorted(list(all_models))
        n_models = len(model_names)
        
        # Initialize matrix
        scores_matrix = np.zeros((n_samples, n_criteria, n_models))
        
        for i, entry in enumerate(evaluation_data):
            for j, model in enumerate(model_names):
                if model in entry:
                    # entry[model] is (scores_list, response_text)
                    scores = entry[model][0] if isinstance(entry[model], tuple) else entry[model]
                    if isinstance(scores, list) and len(scores) == n_criteria:
                        scores_matrix[i, :, j] = scores
                    else:
                        logger.warning(f"Invalid scores for sample {i}, model {model}")
        
        return scores_matrix, model_names
    
    def calculate_weighted_scores(
        self,
        scores_matrix: np.ndarray,
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Calculate weighted total scores
        
        Args:
            scores_matrix: Shape (n_samples, n_criteria, n_models)
            weights: Shape (n_criteria,)
            
        Returns:
            Weighted scores, shape (n_samples, n_models)
        """
        return np.sum(scores_matrix * weights[np.newaxis, :, np.newaxis], axis=1)
    
    def inter_rater_reliability(
        self,
        scores_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate inter-rater reliability metrics
        
        Args:
            scores_matrix: Shape (n_samples, n_criteria, n_models)
            
        Returns:
            Dictionary of reliability metrics
        """
        n_samples, n_criteria, n_models = scores_matrix.shape
        
        # Calculate average scores across models for each criterion
        mean_scores = np.mean(scores_matrix, axis=2)  # (n_samples, n_criteria)
        
        # Calculate variance for each criterion
        criterion_vars = np.var(mean_scores, axis=0)  # (n_criteria,)
        
        # Calculate inter-model correlation
        correlations = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                # Correlations across all samples and criteria
                scores_i = scores_matrix[:, :, i].flatten()
                scores_j = scores_matrix[:, :, j].flatten()
                corr, _ = stats.pearsonr(scores_i, scores_j)
                correlations.append(corr)
        
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        # Cronbach's alpha (simplified)
        n_models = scores_matrix.shape[2]
        if n_models > 1:
            item_vars = np.var(scores_matrix, axis=0)  # (n_criteria, n_models)
            total_var = np.var(np.sum(scores_matrix, axis=1))
            cronbach_alpha = (n_models / (n_models - 1)) * (1 - np.sum(item_vars) / total_var)
        else:
            cronbach_alpha = 0.0
        
        return {
            'inter_model_correlation': avg_correlation,
            'cronbach_alpha': cronbach_alpha,
            'criterion_variances': criterion_vars.tolist()
        }
    
    def correlation_analysis(
        self,
        scores_matrix: np.ndarray,
        overall_scores: np.ndarray
    ) -> Dict[str, List[float]]:
        """
        Analyze correlation between individual criteria and overall quality
        
        Args:
            scores_matrix: Shape (n_samples, n_criteria, n_models)
            overall_scores: Shape (n_samples, n_models) - ground truth quality scores
            
        Returns:
            Dictionary mapping criteria to correlation coefficients
        """
        n_criteria = scores_matrix.shape[1]
        
        # Average across models
        avg_criteria_scores = np.mean(scores_matrix, axis=2)  # (n_samples, n_criteria)
        avg_overall = np.mean(overall_scores, axis=1)  # (n_samples,)
        
        correlations = []
        p_values = []
        
        for i in range(n_criteria):
            corr, p_val = stats.pearsonr(avg_criteria_scores[:, i], avg_overall)
            correlations.append(corr)
            p_values.append(p_val)
        
        return {
            'correlations': correlations,
            'p_values': p_values,
            'criteria_names': self.criteria_names
        }
    
    def optimize_weights(
        self,
        scores_matrix: np.ndarray,
        ground_truth: np.ndarray,
        method: str = "gradient_descent"
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Optimize criteria weights to maximize correlation with ground truth
        
        Args:
            scores_matrix: Shape (n_samples, n_criteria, n_models)
            ground_truth: Shape (n_samples,) - ground truth quality rankings
            method: Optimization method
            
        Returns:
            Tuple of (optimal_weights, optimization_info)
        """
        n_criteria = scores_matrix.shape[1]
        
        # Average across models for stability
        avg_scores = np.mean(scores_matrix, axis=2)  # (n_samples, n_criteria)
        
        def objective(weights):
            """Negative correlation (to maximize)"""
            weighted_scores = np.dot(avg_scores, weights)
            corr, _ = stats.pearsonr(weighted_scores, ground_truth)
            return -corr  # Minimize negative correlation = maximize correlation
        
        # Constraints: weights sum to some constant, all positive
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 10.0},  # Sum to 10
        ]
        bounds = [(0.0, 2.0) for _ in range(n_criteria)]  # Each weight 0-2
        
        # Initial guess: current weights
        x0 = np.array(self.current_weights)
        x0 = x0 / np.sum(x0) * 10.0  # Normalize to sum to 10
        
        # Optimize
        logger.info(f"Optimizing weights using {method}...")
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        final_corr = -result.fun
        
        return optimal_weights, {
            'correlation': final_corr,
            'success': result.success,
            'message': result.message,
            'iterations': result.nit
        }
    
    def compare_systems(
        self,
        baseline_scores: List[Dict],
        improved_scores: List[Dict]
    ) -> Dict[str, Any]:
        """
        Statistical comparison between baseline and improved systems
        
        Args:
            baseline_scores: Evaluation data for baseline system
            improved_scores: Evaluation data for improved system
            
        Returns:
            Statistical comparison results
        """
        baseline_matrix, models_b = self.extract_scores_matrix(baseline_scores)
        improved_matrix, models_i = self.extract_scores_matrix(improved_scores)
        
        # Calculate weighted scores
        weights = np.array(self.current_weights)
        baseline_weighted = self.calculate_weighted_scores(baseline_matrix, weights)
        improved_weighted = self.calculate_weighted_scores(improved_matrix, weights)
        
        # Average across models
        baseline_avg = np.mean(baseline_weighted, axis=1)
        improved_avg = np.mean(improved_weighted, axis=1)
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(improved_avg, baseline_avg)
        
        # Effect size (Cohen's d)
        diff = improved_avg - baseline_avg
        pooled_std = np.sqrt((np.var(baseline_avg) + np.var(improved_avg)) / 2)
        cohens_d = np.mean(diff) / pooled_std if pooled_std > 0 else 0.0
        
        # Confidence interval for mean difference
        conf_interval = stats.t.interval(
            0.95,
            len(diff) - 1,
            loc=np.mean(diff),
            scale=stats.sem(diff)
        )
        
        return {
            'baseline_mean': np.mean(baseline_avg),
            'improved_mean': np.mean(improved_avg),
            'mean_improvement': np.mean(diff),
            'improvement_percentage': (np.mean(diff) / np.mean(baseline_avg)) * 100,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'cohens_d': cohens_d,
            'confidence_interval_95': conf_interval,
            'n_samples': len(diff)
        }
    
    def generate_report(
        self,
        evaluation_data: List[Dict],
        output_path: str = "./analysis_report.md"
    ):
        """
        Generate comprehensive analytical report
        
        Args:
            evaluation_data: Evaluation data
            output_path: Where to save the report
        """
        logger.info("Generating analytical report...")
        
        scores_matrix, model_names = self.extract_scores_matrix(evaluation_data)
        
        # Calculate reliability
        reliability = self.inter_rater_reliability(scores_matrix)
        
        # Calculate criterion statistics
        avg_scores = np.mean(scores_matrix, axis=(0, 2))  # Average per criterion
        std_scores = np.std(scores_matrix, axis=(0, 2))
        
        # Generate markdown report
        report = f"""# Story Outline Evaluation - Analytical Report

## Dataset Overview

- **Number of samples**: {scores_matrix.shape[0]}
- **Number of criteria**: {scores_matrix.shape[1]}
- **Number of evaluator models**: {scores_matrix.shape[2]}
- **Evaluator models**: {', '.join(model_names)}

## Inter-Rater Reliability

- **Inter-model correlation**: {reliability['inter_model_correlation']:.3f}
- **Cronbach's alpha**: {reliability['cronbach_alpha']:.3f}

## Criteria Statistics

| Criterion | Mean Score | Std Dev |
|-----------|-----------|---------|
"""
        
        for i, name in enumerate(self.criteria_names):
            report += f"| {name} | {avg_scores[i]:.2f} | {std_scores[i]:.2f} |\n"
        
        report += f"""
## Current Criteria Weights

| Criterion | Weight |
|-----------|--------|
"""
        
        for name, weight in zip(self.criteria_names, self.current_weights):
            report += f"| {name} | {weight:.2f} |\n"
        
        report += """
## Interpretation

### Reliability Metrics

- **Inter-model correlation > 0.6**: Good agreement between evaluator models
- **Cronbach's alpha > 0.7**: High internal consistency

### Recommendations

Based on the analysis, consider:
1. Revising weights for criteria with low correlation to overall quality
2. Focusing generation improvements on criteria with high variance
3. Investigating criteria where models disagree significantly
"""
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"✓ Report saved to {output_path}")
        return report


if __name__ == "__main__":
    """Test criteria analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Criteria Analysis")
    parser.add_argument("--data", help="Path to evaluation data JSON")
    parser.add_argument("--report", action="store_true", help="Generate analysis report")
    parser.add_argument("--optimize", action="store_true", help="Optimize weights")
    args = parser.parse_args()
    
    analyzer = CriteriaAnalyzer()
    
    if args.data:
        print(f"\nLoading data from {args.data}...")
        data = analyzer.load_evaluation_data(args.data)
        print(f"✓ Loaded {len(data)} samples")
        
        if args.report:
            print("\nGenerating report...")
            analyzer.generate_report(data)
            print("✓ Report generated")
        
        if args.optimize:
            print("\nOptimizing criteria weights...")
            scores_matrix, models = analyzer.extract_scores_matrix(data)
            
            # Use average weighted score as ground truth
            weights = np.array(analyzer.current_weights)
            ground_truth = np.mean(
                analyzer.calculate_weighted_scores(scores_matrix, weights),
                axis=1
            )
            
            optimal_weights, info = analyzer.optimize_weights(
                scores_matrix, ground_truth
            )
            
            print("\n=== Optimization Results ===")
            print(f"Success: {info['success']}")
            print(f"Final correlation: {info['correlation']:.4f}")
            print("\nOptimal weights:")
            for name, weight in zip(analyzer.criteria_names, optimal_weights):
                print(f"  {name}: {weight:.3f}")
    else:
        print("Usage: python criteria_analysis.py --data <path> [--report] [--optimize]")
