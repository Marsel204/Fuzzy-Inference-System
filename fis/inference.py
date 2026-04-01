"""
inference.py
────────────
Module 2 – Inference

Responsibilities
----------------
* Store a list of fuzzy rules.
* Evaluate the rules against fuzzified inputs to determine the firing strength of each rule.
* Aggregate the fuzzy outputs of the rules (Mamdani max-min inference).
"""

from typing import Dict, List, Tuple

class Rule:
    """A single fuzzy rule (Mamdani style).
    
    Currently supports AND aggregation for antecedents.
    Syntax: IF var1 IS term1 AND var2 IS term2 THEN out_var IS out_term
    """
    def __init__(self, antecedents: Dict[str, str], consequent: Tuple[str, str]):
        """
        Args:
            antecedents: Dict mapping input var name to term name (e.g., {"food": "good", "service": "excellent"})
            consequent: Tuple of (output_var_name, term_name) (e.g., ("tip", "high"))
        """
        self.antecedents = antecedents
        self.consequent = consequent

    def evaluate(self, fuzzified_inputs: Dict[str, Dict[str, float]]) -> float:
        """Evaluate the firing strength of this rule given fuzzified inputs.
        Uses min() for AND operator.
        """
        firing_strength = 1.0 # Identity for min
        for var_name, term_name in self.antecedents.items():
            if var_name not in fuzzified_inputs:
                return 0.0 # Variable not provided
            if term_name not in fuzzified_inputs[var_name]:
                return 0.0 # Term not found
            
            # Apply AND (min)
            firing_strength = min(firing_strength, fuzzified_inputs[var_name][term_name])
            
        return firing_strength

class InferenceModule:
    """Module 2 – Inference."""
    def __init__(self):
        self.rules: List[Rule] = []

    def add_rule(self, rule: Rule) -> "InferenceModule":
        self.rules.append(rule)
        return self

    def infer(self, fuzzified_inputs: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Apply rules to fuzzified inputs and aggregate the results.
        Returns the aggregated fuzzy output for each output variable.
        Format: { output_var_name: { output_term_name: aggregated_firing_strength } }
        Uses max() for aggregation of multiple rules yielding the same consequent.
        """
        aggregated_output: Dict[str, Dict[str, float]] = {}

        for rule in self.rules:
            strength = rule.evaluate(fuzzified_inputs)
            out_var, out_term = rule.consequent
            
            if out_var not in aggregated_output:
                aggregated_output[out_var] = {}
                
            if out_term not in aggregated_output[out_var]:
                aggregated_output[out_var][out_term] = strength
            else:
                # Aggregate using max()
                aggregated_output[out_var][out_term] = max(aggregated_output[out_var][out_term], strength)

        return aggregated_output
