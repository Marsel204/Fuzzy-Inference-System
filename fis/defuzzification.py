"""
defuzzification.py
──────────────────
Module 3 – Defuzzification

Responsibilities
----------------
* Given the aggregated firing strengths for an output variable and the
  LinguisticVariable definition (its terms and universe of discourse),
  compute a single crisp value using the centroid method (Center of Gravity).
"""

import math
from typing import Dict, List
from .fuzzification import LinguisticVariable

class DefuzzificationModule:
    """Module 3 – Defuzzification."""
    
    def __init__(self):
        self._output_variables: Dict[str, LinguisticVariable] = {}

    def add_output_variable(self, variable: LinguisticVariable) -> "DefuzzificationModule":
        """Register an output linguistic variable."""
        self._output_variables[variable.name] = variable
        return self

    def defuzzify(self, inferred_output: Dict[str, Dict[str, float]], num_samples: int = 100) -> Dict[str, float]:
        """
        Defuzzify the inferred fuzzy outputs to produce crisp values.
        Uses Center of Gravity (Centroid) method with numerical integration.

        Args:
            inferred_output: { "tip": { "low": 0.2, "high": 0.8 } }
            num_samples: Number of points to sample across the universe of discourse.

        Returns:
            Dict mapping output variable names to their crisp defuzzified values.
        """
        crisp_outputs: Dict[str, float] = {}

        for var_name, term_strengths in inferred_output.items():
            if var_name not in self._output_variables:
                raise KeyError(f"Output variable '{var_name}' not registered in DefuzzificationModule.")
            
            var = self._output_variables[var_name]
            
            # Discretize the universe of discourse
            step = (var.hi - var.lo) / (num_samples - 1)
            x_values = [var.lo + i * step for i in range(num_samples)]
            y_values = [0.0] * num_samples

            # Evaluate each term's membership function and clip it by its firing strength
            for term_name, strength in term_strengths.items():
                if term_name not in var.terms:
                    continue
                if strength <= 0.0:
                    continue
                
                # Membership function for this term
                mf = var.terms[term_name]
                
                # Clipped (Mamdani min) fuzzy set for this rule consequence
                # Aggregate (union/max) with the overall fuzzy set
                for i, x in enumerate(x_values):
                    term_y = min(strength, mf(x))
                    y_values[i] = max(y_values[i], term_y)

            # Compute centroid (Center of Gravity)
            area = sum(y_values)
            if area == 0.0:
                # Fallback if no rules fired or zero area (e.g. median of range)
                crisp_outputs[var_name] = (var.lo + var.hi) / 2.0
            else:
                centroid = sum(x * y for x, y in zip(x_values, y_values)) / area
                crisp_outputs[var_name] = centroid

        return crisp_outputs
