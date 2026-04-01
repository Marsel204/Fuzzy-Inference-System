"""
fuzzification.py
────────────────
Module 1 – Fuzzification

Responsibilities
----------------
* Store linguistic variables, each with a universe of discourse and a
  dict of term → membership function.
* Accept a crisp input for every input variable and return a nested
  dict: { variable_name: { term_name: degree } }
"""

from __future__ import annotations
from typing import Callable, Dict


# Type alias: any callable that maps float → float ∈ [0, 1]
MembershipFn = Callable[[float], float]


class LinguisticVariable:
    """Represents a single linguistic variable (input OR output).

    Attributes
    ----------
    name   : human-readable label (e.g. "food_quality")
    lo, hi : universe-of-discourse bounds
    terms  : mapping from term label → membership function
    """

    def __init__(self, name: str, lo: float, hi: float) -> None:
        self.name: str                    = name
        self.lo:   float                  = lo
        self.hi:   float                  = hi
        self.terms: Dict[str, MembershipFn] = {}

    # ------------------------------------------------------------------ #
    def add_term(self, label: str, fn: MembershipFn) -> "LinguisticVariable":
        """Register a fuzzy term.

        Parameters
        ----------
        label : term name, e.g. "poor", "average", "excellent"
        fn    : membership function (float) → float in [0, 1]

        Returns self for method-chaining.
        """
        self.terms[label] = fn
        return self

    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        return (
            f"LinguisticVariable(name={self.name!r}, "
            f"range=[{self.lo}, {self.hi}], "
            f"terms={list(self.terms)})"
        )


# ═══════════════════════════════════════════════════════════════════════ #

class FuzzificationModule:
    """Module 1 – Fuzzification.

    Usage example
    -------------
    >>> fm = FuzzificationModule()
    >>> fm.add_variable(food_quality_var)
    >>> fm.add_variable(service_var)
    >>> fuzzy_inputs = fm.fuzzify({"food_quality": 7.5, "service": 8.0})
    >>> # fuzzy_inputs == {"food_quality": {"poor": 0.0, "good": 0.5, ...}, ...}
    """

    def __init__(self) -> None:
        self._variables: Dict[str, LinguisticVariable] = {}

    # ------------------------------------------------------------------ #
    def add_variable(self, variable: LinguisticVariable) -> "FuzzificationModule":
        """Register a linguistic variable.

        Returns self for method-chaining.
        """
        self._variables[variable.name] = variable
        return self

    # ------------------------------------------------------------------ #
    def get_variable(self, name: str) -> LinguisticVariable:
        if name not in self._variables:
            raise KeyError(f"Variable '{name}' not registered.")
        return self._variables[name]

    # ------------------------------------------------------------------ #
    def fuzzify(
        self, crisp_inputs: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Compute membership degrees for all registered input variables.

        Parameters
        ----------
        crisp_inputs : { variable_name: crisp_value }

        Returns
        -------
        { variable_name: { term_name: degree } }
        """
        result: Dict[str, Dict[str, float]] = {}
        for var_name, crisp_value in crisp_inputs.items():
            var = self.get_variable(var_name)
            result[var_name] = {
                term: fn(crisp_value)
                for term, fn in var.terms.items()
            }
        return result

    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        var_names = list(self._variables)
        return f"FuzzificationModule(variables={var_names})"
