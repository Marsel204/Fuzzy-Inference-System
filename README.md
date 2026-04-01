# Fuzzy Inference System (FIS) in Python 

A pure Python implementation of a Mamdani-style Fuzzy Inference System constructed from scratch using Object-Oriented Principles.


## Architecture

The FIS pipeline consists of three core modules:

1. **Fuzzification Module (`fis/fuzzification.py`)**
   Takes numerical crisp inputs and converts them into fuzzy degrees of membership (between 0 and 1) based on defined linguistic variables and their membership functions (Triangular, Trapezoidal, Gaussian).

2. **Inference Module (`fis/inference.py`)**
   Stores and evaluates a set of logical rules against the fuzzified inputs. It applies the Mamdani max-min inference method:
   * **AND aggregation:** Evaluates the `min()` of antecedent conditions.
   * **OR / Rule aggregation:** Evaluates the `max()` when multiple rules affect the same consequent term.

3. **Defuzzification Module (`fis/defuzzification.py`)**
   Converts the aggregated fuzzy output back into a single crisp, numerical value using the numerical **Centroid (Center of Gravity)** method. 
   *(Note: Discretization is done efficiently without relying on large arrays or numpy).*

## Project Structure

```text
FIS/
├── fis/
│   ├── __init__.py           # Package exports
│   ├── membership.py         # mf functions: trimf, trapmf, gaussmf
│   ├── fuzzification.py      # LinguisticVariable & FuzzificationModule
│   ├── inference.py          # Rule & InferenceModule
│   └── defuzzification.py    # DefuzzificationModule w/ Centroid calculation
└── main.py                   # Tipping Problem example & pipeline usage
```

## How to use

Run the `main.py` test script to see the FIS in action on the classic "Tipping Problem".

```bash
python main.py
```

### Building Your Own FIS Pipeline

The library is designed to be highly modular. Here is a minimal example:

```python
from fis import LinguisticVariable, trimf, FuzzificationModule, InferenceModule, Rule, DefuzzificationModule

# 1. Define Variables
quality = LinguisticVariable("food", 0, 10)
quality.add_term("poor", lambda x: trimf(x, 0, 0, 5))
quality.add_term("good", lambda x: trimf(x, 5, 10, 10))

tip = LinguisticVariable("tip", 0, 30)
tip.add_term("low", lambda x: trimf(x, 0, 0, 15))
tip.add_term("high", lambda x: trimf(x, 15, 30, 30))

# 2. Fuzzification
fuzzifier = FuzzificationModule().add_variable(quality)
fuzzy_inputs = fuzzifier.fuzzify({"food": 8.0})

# 3. Inference
engine = InferenceModule()
engine.add_rule(Rule(antecedents={"food": "good"}, consequent=("tip", "high")))
fuzzy_out = engine.infer(fuzzy_inputs)

# 4. Defuzzification
defuzzifier = DefuzzificationModule().add_output_variable(tip)
crisp_tip = defuzzifier.defuzzify(fuzzy_out)

print("Recommended Tip:", crisp_tip["tip"])
```
