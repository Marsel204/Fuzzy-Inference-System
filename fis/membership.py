"""
membership.py
─────────────
Reusable membership-function shapes.

Each function returns a float in [0, 1].
"""

import math


def trimf(x: float, a: float, b: float, c: float) -> float:
    """Triangular membership function.

    Args:
        x: crisp input value
        a: left foot  (μ = 0)
        b: peak       (μ = 1)
        c: right foot (μ = 0)

    Returns:
        Degree of membership in [0, 1].
    """
    if a == b == c:
        return 1.0 if x == a else 0.0
    left  = (x - a) / (b - a) if b != a else float(x >= b)
    right = (c - x) / (c - b) if c != b else float(x <= b)
    return float(max(0.0, min(left, right)))


def trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoidal membership function.

    Args:
        x: crisp input value
        a, b: left rising  edge  (a → b: 0 → 1)
        c, d: right falling edge (c → d: 1 → 0)

    Returns:
        Degree of membership in [0, 1].
    """
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b != a else 1.0
    return (d - x) / (d - c) if d != c else 1.0


def gaussmf(x: float, mean: float, sigma: float) -> float:
    """Gaussian membership function.

    Args:
        x:     crisp input value
        mean:  centre of the bell
        sigma: width parameter (standard deviation)

    Returns:
        Degree of membership in [0, 1].
    """
    return math.exp(-0.5 * ((x - mean) / sigma) ** 2)
