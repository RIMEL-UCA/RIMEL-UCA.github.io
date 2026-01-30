"""
Best Practices Analyzer
Defines rules and scoring for Helm chart best practices compliance.
"""

import re
from typing import Dict, List, Set
from pathlib import Path


# Standard labels recommended by Helm
STANDARD_LABELS = {
    'app.kubernetes.io/name',
    'app.kubernetes.io/instance',
    'app.kubernetes.io/version',
    'app.kubernetes.io/component',
    'app.kubernetes.io/part-of',
    'app.kubernetes.io/managed-by',
    'helm.sh/chart'
}

# Required labels (subset of standard)
REQUIRED_LABELS = {
    'app.kubernetes.io/name',
    'app.kubernetes.io/instance',
    'helm.sh/chart',
    'app.kubernetes.io/managed-by'
}


class BestPracticesAnalyzer:
    """Analyzer for Helm chart best practices compliance."""
    
    def __init__(self):
        self.weights = {
            'templates_structure': 0.30,
            'labels_compliance': 0.20,
            'values_naming': 0.20,
            'documentation': 0.15,
            'advanced_features': 0.15
        }
    
    def analyze_template_structure(self, files: Dict[str, str]) -> Dict:
        """Analyze template structure and naming conventions."""
        template_files = {k: v for k, v in files.items() if k.startswith('templates/')}
        
        # Check for _helpers.tpl
        has_helpers = 'templates/_helpers.tpl' in files
        
        # Check file naming (should be dashed, not camelCase)
        camelcase_files = []
        for file_path in template_files.keys():
            filename = Path(file_path).name
            if filename.startswith('_') or filename in ['NOTES.txt']:
                continue
            # Check for camelCase (has lowercase followed by uppercase)
            if re.search(r'[a-z][A-Z]', filename):
                camelcase_files.append(file_path)
        
        # Check for namespaced template definitions
        namespaced_templates = 0
        non_namespaced_templates = 0
        
        for content in template_files.values():
            # Find all template definitions
            defines = re.findall(r'{{\s*define\s+"([^"]+)"', content)
            for template_name in defines:
                if '.' in template_name:  # Namespaced (e.g., "mychart.fullname")
                    namespaced_templates += 1
                else:
                    non_namespaced_templates += 1
        
        return {
            'has_helpers': has_helpers,
            'template_files_count': len(template_files),
            'camelcase_files': len(camelcase_files),
            'camelcase_files_list': camelcase_files,
            'namespaced_templates': namespaced_templates,
            'non_namespaced_templates': non_namespaced_templates,
            'score': self._score_templates(
                has_helpers, 
                len(camelcase_files), 
                namespaced_templates, 
                non_namespaced_templates
            )
        }
    
    def _score_templates(self, has_helpers: bool, camelcase_count: int, 
                         namespaced: int, non_namespaced: int) -> float:
        """Score template structure (0-1)."""
        score = 0.0
        
        # Has helpers: +40%
        if has_helpers:
            score += 0.4
        
        # No camelCase files: +30%
        if camelcase_count == 0:
            score += 0.3
        
        # Namespaced templates: +30%
        total_templates = namespaced + non_namespaced
        if total_templates > 0:
            score += 0.3 * (namespaced / total_templates)
        else:
            score += 0.3  # No templates defined = OK
        
        return score
    
    def analyze_labels(self, files: Dict[str, str]) -> Dict:
        """Analyze standard labels compliance."""
        template_files = {k: v for k, v in files.items() if k.startswith('templates/') and k.endswith('.yaml')}
        
        labels_found = set()
        files_with_labels = 0
        
        for content in template_files.values():
            file_labels = set()
            # Find label sections
            for label in STANDARD_LABELS:
                if label in content:
                    labels_found.add(label)
                    file_labels.add(label)
            
            if file_labels:
                files_with_labels += 1
        
        required_present = REQUIRED_LABELS.intersection(labels_found)
        
        return {
            'standard_labels_found': list(labels_found),
            'required_labels_present': list(required_present),
            'labels_coverage': len(labels_found) / len(STANDARD_LABELS),
            'required_coverage': len(required_present) / len(REQUIRED_LABELS),
            'files_with_labels': files_with_labels,
            'score': len(required_present) / len(REQUIRED_LABELS)
        }
    
    def analyze_values(self, values_content: str) -> Dict:
        """Analyze values.yaml naming and structure."""
        if not values_content:
            return {
                'total_keys': 0,
                'violations': [],
                'violations_count': 0,
                'max_nesting': 0,
                'has_comments': False,
                'score': 0.0
            }
        
        lines = values_content.split('\n')
        
        # Find all keys and check naming
        violations = []
        keys_found = []
        max_nesting = 0
        has_comments = False
        
        for line in lines:
            # Check for comments
            if '#' in line:
                has_comments = True
            
            # Find YAML keys
            match = re.match(r'^(\s*)([a-zA-Z0-9_-]+):', line)
            if match:
                indent = len(match.group(1))
                key = match.group(2)
                keys_found.append(key)
                
                # Calculate nesting level
                nesting = indent // 2
                max_nesting = max(max_nesting, nesting)
                
                # Check naming conventions
                # Should be: lowercase start, camelCase, no hyphens in middle
                if key[0].isupper():
                    violations.append(f"{key}: starts with uppercase")
                elif '-' in key and not key.startswith('-'):
                    violations.append(f"{key}: contains hyphens")
        
        return {
            'total_keys': len(keys_found),
            'violations': violations[:10],  # First 10
            'violations_count': len(violations),
            'max_nesting': max_nesting,
            'has_comments': has_comments,
            'score': self._score_values(len(violations), len(keys_found), max_nesting, has_comments)
        }
    
    def _score_values(self, violations: int, total_keys: int, max_nesting: int, has_comments: bool) -> float:
        """Score values conventions (0-1)."""
        if total_keys == 0:
            return 1.0
        
        score = 0.0
        
        # No violations: +50%
        violation_ratio = violations / total_keys
        score += 0.5 * (1 - min(violation_ratio, 1.0))
        
        # Reasonable nesting: +25%
        if max_nesting <= 3:
            score += 0.25
        elif max_nesting <= 5:
            score += 0.15
        
        # Has comments: +25%
        if has_comments:
            score += 0.25
        
        return score
    
    def analyze_documentation(self, files: Dict[str, str]) -> Dict:
        """Analyze documentation artifacts."""
        has_readme = any(f.lower() == 'readme.md' for f in files.keys())
        has_notes = 'templates/NOTES.txt' in files
        has_schema = 'values.schema.json' in files
        
        return {
            'has_readme': has_readme,
            'has_notes': has_notes,
            'has_schema': has_schema,
            'score': (
                (0.4 if has_readme else 0.0) +
                (0.3 if has_notes else 0.0) +
                (0.3 if has_schema else 0.0)
            )
        }
    
    def analyze_advanced_features(self, files: Dict[str, str]) -> Dict:
        """Analyze advanced features (tests, CI, hooks)."""
        has_tests = any('tests/' in f for f in files.keys())
        has_ci = any(f in files for f in ['.github/workflows/helm.yaml', '.github/workflows/helm.yml', 
                                           '.gitlab-ci.yml', 'Jenkinsfile'])
        has_hooks = any('templates/hooks/' in f or 'templates/pre-' in f or 'templates/post-' in f 
                        for f in files.keys())
        
        return {
            'has_tests': has_tests,
            'has_ci': has_ci,
            'has_hooks': has_hooks,
            'score': (
                (0.4 if has_tests else 0.0) +
                (0.3 if has_ci else 0.0) +
                (0.3 if has_hooks else 0.0)
            )
        }
    
    def calculate_compliance_score(self, analysis_results: Dict) -> float:
        """Calculate overall compliance score."""
        score = 0.0
        
        for category, weight in self.weights.items():
            if category in analysis_results:
                score += analysis_results[category]['score'] * weight
        
        return score
    
    def analyze_chart(self, files: Dict[str, str]) -> Dict:
        """Perform complete best practices analysis."""
        results = {
            'templates_structure': self.analyze_template_structure(files),
            'labels_compliance': self.analyze_labels(files),
            'values_naming': self.analyze_values(files.get('values.yaml', '')),
            'documentation': self.analyze_documentation(files),
            'advanced_features': self.analyze_advanced_features(files)
        }
        
        results['overall_score'] = self.calculate_compliance_score(results)
        
        return results
