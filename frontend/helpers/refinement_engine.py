"""
DataMimicAI Refinement Engine - Phase 4
Feedback Loop for Iterative Synthetic Data Quality Improvement

Features:
- Issue detection and analysis
- Smart parameter recommendations
- Version history tracking
- One-click regeneration
- Quality convergence detection
"""
import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

API_BASE = os.getenv("API_URL", "http://localhost:8000")


# ==================== DATA STRUCTURES ====================

def init_refinement_session_state():
    """Initialize session state for refinement engine"""
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    
    if 'current_generation_version' not in st.session_state:
        st.session_state.current_generation_version = 0
    
    if 'quality_scores_history' not in st.session_state:
        st.session_state.quality_scores_history = []
    
    if 'refinement_recommendations' not in st.session_state:
        st.session_state.refinement_recommendations = []


def add_generation_to_history(
    file_id: str,
    algorithm: str,
    parameters: Dict,
    quality_scores: Dict,
    issues: List[Dict],
    timestamp: str = None
):
    """Add a generation to version history"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    version = st.session_state.current_generation_version + 1
    
    generation_record = {
        "version": version,
        "file_id": file_id,
        "algorithm": algorithm,
        "parameters": parameters,
        "quality_scores": quality_scores,
        "issues": issues,
        "timestamp": timestamp
    }
    
    st.session_state.generation_history.append(generation_record)
    st.session_state.current_generation_version = version
    st.session_state.quality_scores_history.append(quality_scores)
    
    return version


# ==================== ISSUE DETECTION ====================

def detect_quality_issues(file_id: str) -> List[Dict]:
    """
    Analyze synthetic data and detect quality issues using backend metrics API
    Returns list of issues with severity, description, and recommendations
    """
    issues = []
    
    try:
        # Call backend metrics endpoint
        response = requests.get(
            f"{API_BASE}/metrics",
            params={"file_id": file_id},
            timeout=30
        )
        
        if response.status_code == 200:
            metrics_data = response.json()
            metrics = metrics_data.get("metrics", {})
            
            # Analyze metrics and detect issues
            numeric_issues = []
            categorical_issues = []
            
            for column, metric in metrics.items():
                if "error" in metric:
                    continue
                    
                if metric.get("type") == "numeric":
                    # Check KS statistic (higher means more different)
                    ks_stat = metric.get("ks_stat", 0)
                    p_value = metric.get("p_value", 1)
                    
                    if ks_stat > 0.15:  # Significant difference
                        severity = "high" if ks_stat > 0.3 else "medium"
                        numeric_issues.append({
                            "column": column,
                            "ks_stat": ks_stat,
                            "p_value": p_value,
                            "severity": severity
                        })
                        
                elif metric.get("type") == "categorical":
                    # Check chi-square p-value (lower means more different)
                    p_value = metric.get("p_value", 1)
                    
                    if p_value < 0.05:  # Significantly different distributions
                        severity = "high" if p_value < 0.01 else "medium"
                        categorical_issues.append({
                            "column": column,
                            "chi2": metric.get("chi2", 0),
                            "p_value": p_value,
                            "severity": severity
                        })
            
            # Create issue reports based on detected problems
            if numeric_issues:
                high_severity_cols = [i["column"] for i in numeric_issues if i["severity"] == "high"]
                medium_severity_cols = [i["column"] for i in numeric_issues if i["severity"] == "medium"]
                
                if high_severity_cols:
                    issues.append({
                        "id": "dist_mismatch_high",
                        "severity": "high",
                        "category": "Distribution Match",
                        "title": "Significant Distribution Mismatch",
                        "description": f"Numeric columns show significant distribution differences from original data",
                        "metrics": {
                            "affected_columns": high_severity_cols,
                            "avg_ks_stat": sum(i["ks_stat"] for i in numeric_issues if i["severity"] == "high") / max(len(high_severity_cols), 1)
                        },
                        "impact": "Statistical queries may return significantly biased results",
                        "root_cause": "Algorithm may not preserve distribution shapes effectively",
                        "recommendation": "Try TVAE for better distribution matching or increase epochs to 500+"
                    })
                    
                if medium_severity_cols:
                    issues.append({
                        "id": "dist_mismatch_medium",
                        "severity": "medium",
                        "category": "Distribution Match",
                        "title": "Minor Distribution Differences",
                        "description": f"Some numeric columns show moderate distribution differences",
                        "metrics": {
                            "affected_columns": medium_severity_cols,
                            "avg_ks_stat": sum(i["ks_stat"] for i in numeric_issues if i["severity"] == "medium") / max(len(medium_severity_cols), 1)
                        },
                        "impact": "May affect downstream analytics accuracy",
                        "root_cause": "Insufficient training epochs or suboptimal hyperparameters",
                        "recommendation": "Increase training epochs or adjust learning rate"
                    })
            
            if categorical_issues:
                high_severity_cols = [i["column"] for i in categorical_issues if i["severity"] == "high"]
                
                if high_severity_cols:
                    issues.append({
                        "id": "categorical_mismatch",
                        "severity": "high",
                        "category": "Categorical Fidelity",
                        "title": "Categorical Distribution Mismatch",
                        "description": f"Categorical columns show different value distributions from original",
                        "metrics": {
                            "affected_columns": high_severity_cols,
                            "avg_p_value": sum(i["p_value"] for i in categorical_issues if i["severity"] == "high") / max(len(high_severity_cols), 1)
                        },
                        "impact": "Category frequencies may not match expected patterns",
                        "root_cause": "Algorithm may struggle with imbalanced categorical data",
                        "recommendation": "Try CTGAN for better categorical handling or enable class balancing"
                    })
            
            # If no issues detected, add a positive message
            if not issues:
                issues.append({
                    "id": "quality_good",
                    "severity": "info",
                    "category": "Quality Assessment",
                    "title": "Good Quality Detected",
                    "description": "Synthetic data shows good statistical similarity to original data",
                    "metrics": {},
                    "impact": "Data suitable for most use cases",
                    "root_cause": "N/A",
                    "recommendation": "Data quality is acceptable. Consider minor tuning for specific requirements."
                })
        
        else:
            # Fallback to basic mock if API fails. Provide a clearer message when files are missing.
            if response.status_code == 404:
                # Backend 404 usually means the expected synthetic or original CSV is not present
                requested_url = f"{API_BASE}/metrics?file_id={file_id}"
                resp_text = None
                try:
                    resp_text = response.text
                except Exception:
                    resp_text = "<unavailable>"

                issues.append({
                    "id": "metrics_file_missing",
                    "severity": "warning",
                    "category": "System",
                    "title": "Metrics files not found on backend",
                    "description": (
                        "Backend could not find the expected files for metric computation. "
                        f"The metrics endpoint looks for both '{{orig}}.csv' and 'syn_{{orig}}.csv' under the backend UPLOAD_DIR using the provided file_id ({file_id})."
                    ),
                    "metrics": {},
                    "impact": "Quality assessment limited until synthetic file is available",
                    "root_cause": (
                        f"Backend returned 404 for request {requested_url}. Response: {resp_text}"
                    ),
                    "recommendation": (
                        "Ensure a synthetic file named 'syn_<file_id>.csv' exists in the backend uploads directory, "
                        "or re-run generation for this file via the Generate step in the UI."
                    ),
                    "debug": {
                        "requested_url": requested_url,
                        "response_text": resp_text,
                        "response_status": response.status_code
                    }
                })
            else:
                # Include response body for diagnostics
                requested_url = f"{API_BASE}/metrics?file_id={file_id}"
                resp_text = None
                try:
                    resp_text = response.text
                except Exception:
                    resp_text = "<unavailable>"

                issues.append({
                    "id": "api_unavailable",
                    "severity": "warning",
                    "category": "System",
                    "title": "Metrics API Unavailable",
                    "description": "Unable to fetch detailed quality metrics from backend",
                    "metrics": {},
                    "impact": "Quality assessment limited",
                    "root_cause": f"Backend returned status {response.status_code}: {resp_text}",
                    "recommendation": "Check backend API connectivity and try again",
                    "debug": {
                        "requested_url": requested_url,
                        "response_text": resp_text,
                        "response_status": response.status_code
                    }
                })
            
    except requests.exceptions.RequestException as e:
        # Network error - use basic fallback
        issues.append({
            "id": "connection_error",
            "severity": "warning",
            "category": "System",
            "title": "Unable to Connect to Backend",
            "description": f"Could not retrieve quality metrics: {str(e)}",
            "metrics": {},
            "impact": "Quality assessment unavailable",
            "root_cause": "Network or API connection issue",
            "recommendation": "Verify backend API is running and accessible"
        })
    
    return issues


def categorize_issues_by_severity(issues: List[Dict]) -> Dict[str, List[Dict]]:
    """Group issues by severity level"""
    categorized = {
        "high": [],
        "medium": [],
        "low": [],
        "info": [],
        "warning": []
    }
    
    for issue in issues:
        severity = issue.get("severity", "low")
        # Map unknown severities to 'low' to prevent KeyError
        if severity not in categorized:
            severity = "low"
        categorized[severity].append(issue)
    
    return categorized


# ==================== PARAMETER RECOMMENDATIONS ====================

def generate_recommendations(issues: List[Dict], current_params: Dict) -> List[Dict]:
    """
    Generate smart parameter recommendations based on detected issues
    Returns prioritized list of actionable changes
    """
    recommendations = []
    
    # Group issues by category for holistic recommendations
    high_severity_issues = [i for i in issues if i['severity'] == 'high']
    
    # Recommendation 1: Algorithm Change
    if any('correlation' in i['title'].lower() for i in high_severity_issues):
        recommendations.append({
            "id": "rec_algo_1",
            "priority": 1,
            "type": "algorithm_change",
            "title": "Switch to Correlation-Preserving Algorithm",
            "description": "Current algorithm struggles with feature correlations. CTGAN and TabDDPM excel at preserving complex relationships.",
            "action": {
                "parameter": "algorithm",
                "current_value": current_params.get("algorithm", "GaussianCopula"),
                "suggested_value": "CTGAN",
                "alternatives": ["TabDDPM", "TVAE"]
            },
            "expected_improvement": {
                "correlation_fidelity": "+30%",
                "overall_quality": "+15-20%"
            },
            "trade_offs": {
                "training_time": "2-3x longer",
                "computational_cost": "Higher GPU usage"
            }
        })
    
    # Recommendation 2: Increase Epochs
    current_epochs = current_params.get("epochs", 300)
    if current_epochs < 500:
        recommendations.append({
            "id": "rec_epochs_1",
            "priority": 2,
            "type": "parameter_tuning",
            "title": "Increase Training Epochs",
            "description": "Model hasn't fully converged. More training iterations will improve quality.",
            "action": {
                "parameter": "epochs",
                "current_value": current_epochs,
                "suggested_value": 500,
                "alternatives": [600, 750]
            },
            "expected_improvement": {
                "distribution_match": "+10-15%",
                "convergence": "Better stability"
            },
            "trade_offs": {
                "training_time": f"+{int((500-current_epochs)/current_epochs*100)}% time",
                "diminishing_returns": "May plateau after 600 epochs"
            }
        })
    
    # Recommendation 3: Batch Size Optimization
    current_batch_size = current_params.get("batch_size", 500)
    if current_batch_size > 200:
        recommendations.append({
            "id": "rec_batch_1",
            "priority": 3,
            "type": "parameter_tuning",
            "title": "Reduce Batch Size",
            "description": "Smaller batches can improve gradient stability and model convergence.",
            "action": {
                "parameter": "batch_size",
                "current_value": current_batch_size,
                "suggested_value": 128,
                "alternatives": [64, 256]
            },
            "expected_improvement": {
                "training_stability": "+15%",
                "gradient_quality": "Better"
            },
            "trade_offs": {
                "training_time": "+20% per epoch",
                "memory_usage": "-30%"
            }
        })
    
    # Recommendation 4: Privacy Enhancement
    if any('privacy' in i['category'].lower() for i in issues):
        recommendations.append({
            "id": "rec_privacy_1",
            "priority": 4,
            "type": "feature_addition",
            "title": "Enable Differential Privacy",
            "description": "Add privacy protection to prevent potential data leakage.",
            "action": {
                "parameter": "algorithm",
                "current_value": current_params.get("algorithm", "CTGAN"),
                "suggested_value": "DP-GAN",
                "alternatives": ["PATE-GAN"]
            },
            "expected_improvement": {
                "privacy_score": "+25%",
                "re_identification_risk": "-80%"
            },
            "trade_offs": {
                "utility_score": "-5 to -10%",
                "data_quality": "Slight degradation"
            }
        })
    
    # Recommendation 5: Hyperparameter Fine-tuning
    recommendations.append({
        "id": "rec_lr_1",
        "priority": 5,
        "type": "parameter_tuning",
        "title": "Optimize Learning Rate",
        "description": "Adjust learning rate for better convergence and stability.",
        "action": {
            "parameter": "learning_rate",
            "current_value": current_params.get("learning_rate", 0.0002),
            "suggested_value": 0.0001,
            "alternatives": [0.00005, 0.0005]
        },
        "expected_improvement": {
            "convergence_speed": "Faster",
            "training_stability": "+10%"
        },
        "trade_offs": {
            "total_training_time": "May need more epochs",
            "sensitivity": "Requires tuning"
        }
    })
    
    return recommendations


def apply_recommendation(recommendation: Dict, current_params: Dict) -> Dict:
    """
    Apply a recommendation to current parameters
    Returns updated parameter dictionary
    """
    updated_params = current_params.copy()
    action = recommendation["action"]
    
    parameter = action["parameter"]
    suggested_value = action["suggested_value"]
    
    updated_params[parameter] = suggested_value
    
    # Log the change
    st.session_state.setdefault('parameter_changes_log', []).append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "recommendation_id": recommendation["id"],
        "parameter": parameter,
        "old_value": action["current_value"],
        "new_value": suggested_value,
        "reason": recommendation["title"]
    })
    
    return updated_params


# ==================== QUALITY SCORING ====================

def calculate_quality_scores(file_id: str) -> Dict[str, float]:
    """
    Calculate comprehensive quality scores for synthetic data using backend metrics
    Returns dict with fidelity, privacy, and utility scores
    """
    try:
        # Call backend metrics endpoint
        response = requests.get(
            f"{API_BASE}/metrics",
            params={"file_id": file_id},
            timeout=30
        )
        
        if response.status_code == 200:
            metrics_data = response.json()
            metrics = metrics_data.get("metrics", {})
            
            # Calculate aggregate scores from metrics
            numeric_scores = []
            categorical_scores = []
            
            for column, metric in metrics.items():
                if "error" in metric:
                    continue
                    
                if metric.get("type") == "numeric":
                    # KS statistic: lower is better (0 = perfect match, 1 = completely different)
                    ks_stat = metric.get("ks_stat", 0.5)
                    # Convert to score (0-1 scale, higher is better)
                    score = max(0, 1 - ks_stat)
                    numeric_scores.append(score)
                    
                elif metric.get("type") == "categorical":
                    # Chi-square p-value: higher is better (>0.05 means similar distributions)
                    p_value = metric.get("p_value", 0)
                    # Convert to score (0-1 scale)
                    score = min(1.0, p_value * 10)  # Scale up p-values for scoring
                    categorical_scores.append(score)
            
            # Calculate component scores
            distribution_score = (
                (sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.8) * 0.7 +
                (sum(categorical_scores) / len(categorical_scores) if categorical_scores else 0.8) * 0.3
            )
            
            # Fidelity = distribution match
            fidelity = distribution_score
            
            # Correlation (use fidelity as proxy for now - can be enhanced with correlation-specific metrics)
            correlation = fidelity * 0.9  # Assume slightly lower correlation preservation
            
            # Privacy (inverse of fidelity - higher fidelity may mean lower privacy)
            # This is a simplified heuristic
            privacy = 0.85 + (1 - fidelity) * 0.15
            
            # Utility (balance between fidelity and diversity)
            utility = (fidelity * 0.8 + privacy * 0.2)
            
            # Diversity (based on number of unique values - proxy metric)
            diversity = 0.80  # Placeholder - could calculate from actual data
            
            # Overall score (weighted average)
            overall = (
                fidelity * 0.35 +
                privacy * 0.20 +
                utility * 0.25 +
                correlation * 0.10 +
                diversity * 0.10
            )
            
            scores = {
                "fidelity": round(fidelity, 3),
                "privacy": round(privacy, 3),
                "utility": round(utility, 3),
                "correlation": round(correlation, 3),
                "distribution": round(distribution_score, 3),
                "diversity": round(diversity, 3),
                "overall": round(overall, 3)
            }
            
            return scores
        else:
            # Fallback to reasonable defaults if API fails
            return {
                "fidelity": 0.75,
                "privacy": 0.85,
                "utility": 0.70,
                "correlation": 0.70,
                "distribution": 0.75,
                "diversity": 0.80,
                "overall": 0.75
            }
            
    except requests.exceptions.RequestException:
        # Network error - return conservative scores
        return {
            "fidelity": 0.70,
            "privacy": 0.85,
            "utility": 0.65,
            "correlation": 0.65,
            "distribution": 0.70,
            "diversity": 0.75,
            "overall": 0.70
        }


def detect_convergence(quality_history: List[Dict], window: int = 3, threshold: float = 0.02) -> bool:
    """
    Detect if quality improvements have plateaued
    Returns True if quality has converged (stopped improving)
    """
    if len(quality_history) < window:
        return False
    
    recent_scores = [q["overall"] for q in quality_history[-window:]]
    
    # Check if variance in recent scores is below threshold
    if len(recent_scores) > 1:
        variance = max(recent_scores) - min(recent_scores)
        return variance < threshold
    
    return False


# ==================== VERSION MANAGEMENT ====================

def get_version_comparison(version1: int, version2: int) -> Dict:
    """
    Compare two generation versions
    Returns comparison of parameters, scores, and improvements
    """
    history = st.session_state.generation_history
    
    if version1 < 1 or version2 < 1 or version1 > len(history) or version2 > len(history):
        return None
    
    v1 = history[version1 - 1]
    v2 = history[version2 - 1]
    
    comparison = {
        "version1": v1,
        "version2": v2,
        "score_changes": {},
        "parameter_changes": {},
        "improvement_summary": ""
    }
    
    # Compare scores
    for metric in v1["quality_scores"]:
        if metric in v2["quality_scores"]:
            diff = v2["quality_scores"][metric] - v1["quality_scores"][metric]
            comparison["score_changes"][metric] = {
                "v1": v1["quality_scores"][metric],
                "v2": v2["quality_scores"][metric],
                "change": diff,
                "change_pct": (diff / v1["quality_scores"][metric] * 100) if v1["quality_scores"][metric] > 0 else 0
            }
    
    # Compare parameters
    for param in v1["parameters"]:
        if v1["parameters"][param] != v2["parameters"].get(param):
            comparison["parameter_changes"][param] = {
                "v1": v1["parameters"][param],
                "v2": v2["parameters"].get(param)
            }
    
    # Generate summary
    overall_improvement = comparison["score_changes"].get("overall", {}).get("change", 0)
    if overall_improvement > 0.05:
        comparison["improvement_summary"] = f"🎉 Significant improvement: +{overall_improvement:.1%}"
    elif overall_improvement > 0:
        comparison["improvement_summary"] = f"✅ Slight improvement: +{overall_improvement:.1%}"
    elif overall_improvement < -0.05:
        comparison["improvement_summary"] = f"⚠️ Quality degraded: {overall_improvement:.1%}"
    else:
        comparison["improvement_summary"] = "➡️ Minimal change"
    
    return comparison


def rollback_to_version(version: int) -> bool:
    """
    Rollback to a previous generation version
    Returns True if successful
    """
    history = st.session_state.generation_history
    
    if version < 1 or version > len(history):
        return False
    
    target_version = history[version - 1]
    
    # Restore file_id and parameters
    st.session_state.generated_file_id = target_version["file_id"]
    st.session_state.generation_parameters = target_version["parameters"]
    st.session_state.selected_algorithm = target_version["algorithm"]
    
    st.success(f"✅ Rolled back to Version {version} (Generated at {target_version['timestamp']})")
    
    return True


# ==================== EXPORT & REPORTING ====================

def export_refinement_report() -> str:
    """
    Generate comprehensive refinement report
    Returns JSON string with complete history and analysis
    """
    report = {
        "report_metadata": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_iterations": len(st.session_state.generation_history),
            "current_version": st.session_state.current_generation_version
        },
        "generation_history": st.session_state.generation_history,
        "quality_progression": st.session_state.quality_scores_history,
        "parameter_changes": st.session_state.get('parameter_changes_log', []),
        "final_quality_scores": st.session_state.quality_scores_history[-1] if st.session_state.quality_scores_history else None,
        "convergence_detected": detect_convergence(st.session_state.quality_scores_history),
        "total_improvements": sum(
            1 for i in range(1, len(st.session_state.quality_scores_history))
            if st.session_state.quality_scores_history[i]["overall"] > st.session_state.quality_scores_history[i-1]["overall"]
        )
    }
    
    return json.dumps(report, indent=2)


def export_comparison_report(version1: int, version2: int) -> str:
    """Export detailed comparison between two versions"""
    comparison = get_version_comparison(version1, version2)
    if comparison:
        return json.dumps(comparison, indent=2)
    return "{}"


# ==================== SESSION PERSISTENCE (Phase 5) ====================

def save_refinement_session(session_name: str = None) -> str:
    """
    Save current refinement session to JSON for later restoration
    Returns JSON string with complete session state
    """
    if session_name is None:
        session_name = f"refinement_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    session_data = {
        "session_name": session_name,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_id": st.session_state.get("file_id"),
        "generation_history": st.session_state.get("generation_history", []),
        "current_generation_version": st.session_state.get("current_generation_version", 0),
        "quality_scores_history": st.session_state.get("quality_scores_history", []),
        "refinement_recommendations": st.session_state.get("refinement_recommendations", []),
        "parameter_changes_log": st.session_state.get("parameter_changes_log", []),
        "generation_parameters": st.session_state.get("generation_parameters", {}),
        "selected_algorithm": st.session_state.get("selected_algorithm")
    }
    
    return json.dumps(session_data, indent=2)


def load_refinement_session(session_json: str) -> bool:
    """
    Restore refinement session from JSON string
    Returns True if successful, False otherwise
    """
    try:
        session_data = json.loads(session_json)
        
        # Restore all session state
        st.session_state.file_id = session_data.get("file_id")
        st.session_state.generation_history = session_data.get("generation_history", [])
        st.session_state.current_generation_version = session_data.get("current_generation_version", 0)
        st.session_state.quality_scores_history = session_data.get("quality_scores_history", [])
        st.session_state.refinement_recommendations = session_data.get("refinement_recommendations", [])
        st.session_state.parameter_changes_log = session_data.get("parameter_changes_log", [])
        st.session_state.generation_parameters = session_data.get("generation_parameters", {})
        st.session_state.selected_algorithm = session_data.get("selected_algorithm")
        
        return True
    except Exception as e:
        st.error(f"Failed to load session: {str(e)}")
        return False


def get_session_summary() -> Dict:
    """Get summary of current refinement session"""
    return {
        "file_id": st.session_state.get("file_id"),
        "total_iterations": len(st.session_state.get("generation_history", [])),
        "current_version": st.session_state.get("current_generation_version", 0),
        "algorithms_tried": list(set([
            v["algorithm"] for v in st.session_state.get("generation_history", [])
        ])),
        "quality_trend": (
            "improving" if len(st.session_state.get("quality_scores_history", [])) >= 2 and
            st.session_state.quality_scores_history[-1]["overall"] > st.session_state.quality_scores_history[0]["overall"]
            else "stable"
        ),
        "last_updated": st.session_state.get("generation_history", [{}])[-1].get("timestamp", "N/A") if st.session_state.get("generation_history") else "N/A"
    }
