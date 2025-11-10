# Feature Engineering Module - README

## Overview

The Feature Engineering module provides intelligent, context-aware transformation suggestions for data preparation and synthetic data generation in DataMimicAI.

## Module Structure

```
feature_engineering/
├── __init__.py
├── feature_suggester.py          # Original AI-driven suggestions
├── utility_suggester.py          # NEW: Utility-focused transformations
├── privacy_suggester.py          # NEW: Privacy-focused transformations
└── conflict_resolver.py          # NEW: Conflict detection & resolution
```

## Components

### 1. UtilitySuggester (`utility_suggester.py`)

**Purpose**: Recommend transformations to improve data quality and model performance.

**Features**:
- Statistical analysis (skewness, outliers, variance)
- 6 transformation types
- Confidence scoring
- Python code generation

**Usage**:
```python
from src.core.feature_engineering.utility_suggester import UtilitySuggester

suggester = UtilitySuggester(df, profile_data=profiling_results)
suggestions = suggester.suggest_utility_transforms()

for sug in suggestions:
    print(f"{sug['column']}: {sug['transformation']} ({sug['confidence']:.0%})")
    code = suggester.get_transform_code(sug)
    print(code)
```

**Transformations**:
- `log_transform` - Reduce skewness
- `sqrt_transform` - Stabilize variance
- `power_transform` - Normalize distribution
- `standard_scaler` - Normalize for convergence
- `robust_scaler` - Handle outliers
- `minmax_scaler` - Scale to [0,1]

### 2. PrivacySuggester (`privacy_suggester.py`)

**Purpose**: Recommend privacy-enhancing transformations to protect sensitive data.

**Features**:
- PII detection integration
- k-anonymity risk assessment
- 6 privacy-enhancing transformations (PETs)
- Risk level classification

**Usage**:
```python
from src.core.feature_engineering.privacy_suggester import PrivacySuggester

suggester = PrivacySuggester(df, pii_report=pii_results, k_anonymity_report=k_results)
suggestions = suggester.suggest_privacy_transforms()

for sug in suggestions:
    print(f"{sug['column']}: {sug['transformation']} (Risk: {sug['risk_level']})")
    code = suggester.get_transform_code(sug)
    print(code)
```

**Transformations**:
- `redact` - Complete removal (for EMAIL, SSN)
- `hash` - One-way hashing (SHA-256)
- `mask` - Partial masking (for NAME, ADDRESS)
- `bin` - Binning (for numeric QIs like Age)
- `generalize` - Rare value grouping (for categorical QIs)
- `suppress` - Column removal (for very high risk)

### 3. ConflictResolver (`conflict_resolver.py`)

**Purpose**: Detect and manage conflicts between utility and privacy suggestions.

**Features**:
- Automatic conflict detection
- Severity calculation (high/medium/low)
- AI-powered recommendations
- Resolution tracking

**Usage**:
```python
from src.core.feature_engineering.conflict_resolver import ConflictResolver

resolver = ConflictResolver()
conflicts = resolver.detect_conflicts(utility_suggestions, privacy_suggestions)

for conflict in conflicts:
    print(f"Column: {conflict['column']}")
    print(f"Severity: {conflict['severity']}")
    print(f"Utility: {conflict['utility_transform']}")
    print(f"Privacy: {conflict['privacy_transform']}")
    print(f"Recommendation: {conflict['recommendation']}")

# User makes decision
resolution = resolver.resolve_conflict(
    column="age",
    chosen_category="privacy",
    chosen_transformation="bin",
    user_note="Compliance requirement"
)

summary = resolver.generate_conflict_summary()
print(f"Total conflicts: {summary['total_conflicts']}")
print(f"High priority: {summary['high_priority_conflicts']}")
```

## Integration with Existing Modules

### Profiler Integration
```python
from src.core.eda.profiling import Profiler
from src.core.feature_engineering.utility_suggester import UtilitySuggester

profiler = Profiler(df)
profile_data = profiler.dataset_profile()

suggester = UtilitySuggester(df, profile_data=profile_data)
suggestions = suggester.suggest_utility_transforms()
```

### PII Scanner Integration
```python
from src.core.eda.pii_scan import PIIScanner
from src.core.feature_engineering.privacy_suggester import PrivacySuggester

scanner = PIIScanner(df)
pii_report = scanner.run_fast_scan()

suggester = PrivacySuggester(df, pii_report=pii_report)
suggestions = suggester.suggest_privacy_transforms()
```

### k-Anonymity Integration
```python
from src.core.eda.privacy import KAnonymityAnalyzer
from src.core.feature_engineering.privacy_suggester import PrivacySuggester

analyzer = KAnonymityAnalyzer(df)
potential_qis = analyzer.identify_potential_qis()
k_report = {"potential_qis": potential_qis}

suggester = PrivacySuggester(df, k_anonymity_report=k_report)
suggestions = suggester.suggest_privacy_transforms()
```

## Complete Workflow Example

```python
import pandas as pd
from src.core.eda.profiling import Profiler
from src.core.eda.pii_scan import PIIScanner
from src.core.eda.privacy import KAnonymityAnalyzer
from src.core.feature_engineering.utility_suggester import UtilitySuggester
from src.core.feature_engineering.privacy_suggester import PrivacySuggester
from src.core.feature_engineering.conflict_resolver import ConflictResolver

# Load data
df = pd.read_csv("data.csv")

# Step 1: Profiling
profiler = Profiler(df)
profile_data = profiler.dataset_profile()

# Step 2: PII Detection
scanner = PIIScanner(df)
pii_report = scanner.run_fast_scan()

# Step 3: k-Anonymity Analysis
analyzer = KAnonymityAnalyzer(df)
potential_qis = analyzer.identify_potential_qis()
k_report = {"potential_qis": potential_qis}

# Step 4: Generate Suggestions
utility_suggester = UtilitySuggester(df, profile_data=profile_data)
utility_suggestions = utility_suggester.suggest_utility_transforms()

privacy_suggester = PrivacySuggester(df, pii_report=pii_report, k_anonymity_report=k_report)
privacy_suggestions = privacy_suggester.suggest_privacy_transforms()

# Step 5: Detect Conflicts
resolver = ConflictResolver()
conflicts = resolver.detect_conflicts(utility_suggestions, privacy_suggestions)
conflict_summary = resolver.generate_conflict_summary()

# Step 6: Separate Non-Conflicting
non_conflicting_utility, non_conflicting_privacy = resolver.get_non_conflicting_suggestions(
    utility_suggestions, privacy_suggestions
)

# Step 7: Display Results
print(f"Utility Suggestions: {len(utility_suggestions)}")
print(f"Privacy Suggestions: {len(privacy_suggestions)}")
print(f"Conflicts: {conflict_summary['total_conflicts']}")
print(f"High Priority Conflicts: {len(conflict_summary['high_priority_conflicts'])}")

# Step 8: Resolve Conflicts (simulated user decision)
for conflict in conflicts:
    if conflict['severity'] == 'high':
        # Prioritize privacy for high-risk conflicts
        resolver.resolve_conflict(
            column=conflict['column'],
            chosen_category='privacy',
            chosen_transformation=conflict['privacy_transform']
        )
    else:
        # User would decide via UI
        pass

# Step 9: Get Resolution History
resolutions = resolver.get_resolution_history()
print(f"Resolved Conflicts: {len(resolutions)}")
```

## Configuration Management

```python
from src.core.feedback_engine import TransformConfigManager

# Create manager
config_manager = TransformConfigManager()

# Add decisions
for suggestion in accepted_suggestions:
    config_manager.add_decision(
        column=suggestion['column'],
        transformation=suggestion['transformation'],
        category=suggestion['category'],
        params=suggestion.get('params', {}),
        reason=suggestion['reason'],
        metadata=suggestion.get('metrics', {})
    )

# Save to file
config_path = config_manager.save_config(file_id="dataset_123")
print(f"Saved to: {config_path}")

# Export as dict
config = config_manager.export_config()

# Load from file
config_manager_new = TransformConfigManager()
success = config_manager_new.load_config(config_path)
```

## Testing

### Unit Tests

```python
# tests/test_utility_suggester.py
def test_skewness_detection():
    df = pd.DataFrame({'col': [1, 2, 2, 3, 100, 200, 500]})
    suggester = UtilitySuggester(df)
    suggestions = suggester.suggest_utility_transforms()
    
    assert any(s['column'] == 'col' and s['transformation'] == 'log_transform' 
               for s in suggestions)

# tests/test_privacy_suggester.py
def test_pii_detection():
    pii_report = {
        "detections": [{"column": "email", "pii_types": ["EMAIL"], "confidence": 0.95}]
    }
    suggester = PrivacySuggester(df, pii_report=pii_report)
    suggestions = suggester.suggest_privacy_transforms()
    
    assert any(s['column'] == 'email' and s['transformation'] == 'redact' 
               for s in suggestions)

# tests/test_conflict_resolver.py
def test_conflict_detection():
    utility = [{"column": "age", "transformation": "log_transform"}]
    privacy = [{"column": "age", "transformation": "bin"}]
    
    resolver = ConflictResolver()
    conflicts = resolver.detect_conflicts(utility, privacy)
    
    assert len(conflicts) == 1
    assert conflicts[0]['column'] == 'age'
```

### Integration Tests

```python
# tests/test_integration.py
def test_full_workflow():
    df = load_test_data()
    
    # Generate all suggestions
    result = run_full_analysis(df)
    
    assert 'utility_suggestions' in result
    assert 'privacy_suggestions' in result
    assert 'conflicts' in result
    
    # Verify conflict detection
    if result['conflicts']:
        assert all('severity' in c for c in result['conflicts'])
        assert all('recommendation' in c for c in result['conflicts'])
```

## Best Practices

### 1. Always Provide Context
```python
# Good - provides profiling data
suggester = UtilitySuggester(df, profile_data=profiling_results)

# Less optimal - will work but might miss opportunities
suggester = UtilitySuggester(df)
```

### 2. Handle Edge Cases
```python
suggestions = suggester.suggest_utility_transforms()

if not suggestions:
    logger.warning("No utility suggestions generated")
    # Handle accordingly
```

### 3. Validate Suggestions
```python
for suggestion in suggestions:
    # Ensure required fields exist
    assert 'column' in suggestion
    assert 'transformation' in suggestion
    assert 'category' in suggestion
    
    # Validate column exists in dataframe
    assert suggestion['column'] in df.columns
```

### 4. Document Decisions
```python
config_manager.add_decision(
    column=column,
    transformation=transformation,
    category=category,
    params=params,
    reason=reason,
    metadata={
        'user': 'analyst_1',
        'timestamp': datetime.now().isoformat(),
        'original_skewness': 2.8,
        'conflict_resolved': True
    }
)
```

## Performance Considerations

- **Caching**: Profile data can be cached to avoid repeated calculations
- **Parallel Processing**: PII scanning can run in parallel with profiling
- **Batch Operations**: Process multiple columns together when possible
- **Memory**: Suggestions are lightweight dictionaries, suitable for large datasets

## Error Handling

```python
try:
    suggestions = suggester.suggest_utility_transforms()
except Exception as e:
    logger.error(f"Failed to generate utility suggestions: {e}")
    suggestions = []
    # Gracefully degrade or notify user
```

## Logging

```python
import logging

logger = logging.getLogger(__name__)

# Log suggestion generation
logger.info(f"Generated {len(suggestions)} utility suggestions")
logger.debug(f"Suggestions: {suggestions}")

# Log conflicts
logger.warning(f"Detected {len(conflicts)} conflicts requiring resolution")
for conflict in conflicts:
    if conflict['severity'] == 'high':
        logger.critical(f"High-priority conflict on column: {conflict['column']}")
```

## Future Enhancements

### Planned Features
- [ ] Custom transformation templates
- [ ] Transformation dependency graph
- [ ] Before/after visualization
- [ ] Automated A/B testing
- [ ] Learning from user decisions
- [ ] Privacy budget tracking

### Extension Points
```python
# Custom transformation rule
class CustomUtilitySuggester(UtilitySuggester):
    def _analyze_column(self, column: str) -> List[UtilitySuggestion]:
        suggestions = super()._analyze_column(column)
        
        # Add custom rule
        if my_custom_condition(self.df[column]):
            suggestions.append(UtilitySuggestion(
                column=column,
                transformation="my_custom_transform",
                reason="Custom business logic",
                confidence=0.75,
                params={"custom_param": value},
                metrics={}
            ))
        
        return suggestions
```

## API Reference

See `Documents/Context_Aware_Feature_Suggestions_Implementation.md` for complete API documentation.

## Support

For issues or questions:
- Check implementation guide
- Review test cases
- Check backend logs
- Consult architecture diagrams

## Credits

Part of DataMimicAI Synthetic Data Generation Platform  
Feature Engineering & Privacy Enhancement Module
