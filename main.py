"""
main.py
───────
Test program for the Fuzzy Inference System.
Based on the classic Tipping problem.
"""

from fis import (
    trimf, trapmf, gaussmf,
    LinguisticVariable, FuzzificationModule,
    Rule, InferenceModule, DefuzzificationModule
)

def build_fis():
    # --------------------------------------------------------- #
    # 1. Define Linguistic Variables & Their Fuzzy Sets
    # --------------------------------------------------------- #
    
    # Input 1: Food Quality [0, 10]
    food_var = LinguisticVariable("food", 0, 10)
    food_var.add_term("poor", lambda x: trapmf(x, -1, 0, 1, 3))
    food_var.add_term("average", lambda x: trapmf(x, 2, 4, 6, 8))
    food_var.add_term("excellent", lambda x: trapmf(x, 7, 9, 10, 11))

    # Input 2: Service Level [0, 10]
    service_var = LinguisticVariable("service", 0, 10)
    service_var.add_term("poor", lambda x: trapmf(x, -1, 0, 1, 4))
    service_var.add_term("good", lambda x: trapmf(x, 3, 5, 5, 7))
    service_var.add_term("excellent", lambda x: trapmf(x, 6, 9, 10, 11))

    # Output: Tip Percentage [0, 30]
    tip_var = LinguisticVariable("tip", 0, 30)
    tip_var.add_term("low", lambda x: trimf(x, 0, 5, 10))
    tip_var.add_term("medium", lambda x: trimf(x, 10, 15, 20))
    tip_var.add_term("high", lambda x: trimf(x, 20, 25, 30))

    # --------------------------------------------------------- #
    # 2. Module 1: Fuzzification
    # --------------------------------------------------------- #
    fuzzifier = FuzzificationModule()
    fuzzifier.add_variable(food_var).add_variable(service_var)

    # --------------------------------------------------------- #
    # 3. Module 2: Inference
    # --------------------------------------------------------- #
    inference_engine = InferenceModule()
    
    # Rule 1: IF food IS poor OR service IS poor THEN tip IS low
    # Splitting OR into separate rules since our module uses AND internally
    inference_engine.add_rule(Rule(antecedents={"food": "poor"}, consequent=("tip", "low")))
    inference_engine.add_rule(Rule(antecedents={"service": "poor"}, consequent=("tip", "low")))
    
    # Rule 2: IF service IS good THEN tip IS medium
    inference_engine.add_rule(Rule(antecedents={"service": "good"}, consequent=("tip", "medium")))
    
    # Rule 3: IF food IS excellent OR service IS excellent THEN tip IS high
    inference_engine.add_rule(Rule(antecedents={"food": "excellent"}, consequent=("tip", "high")))
    inference_engine.add_rule(Rule(antecedents={"service": "excellent"}, consequent=("tip", "high")))
    
    # Rule 4: IF food IS average AND service IS good THEN tip IS medium (Example of AND)
    inference_engine.add_rule(Rule(
        antecedents={"food": "average", "service": "good"}, 
        consequent=("tip", "medium")
    ))

    # --------------------------------------------------------- #
    # 4. Module 3: Defuzzification
    # --------------------------------------------------------- #
    defuzzifier = DefuzzificationModule()
    defuzzifier.add_output_variable(tip_var)
    
    return fuzzifier, inference_engine, defuzzifier

def run_test(fuzzifier, inference_engine, defuzzifier, food_score, service_score):
    print(f"\n--- Testing FIS: Food={food_score}, Service={service_score} ---")
    
    # Step 1
    crisp_inputs = {"food": food_score, "service": service_score}
    fuzzy_inputs = fuzzifier.fuzzify(crisp_inputs)
    print("1. Fuzzified Inputs:")
    for var, terms in fuzzy_inputs.items():
        print(f"   {var}: {terms}")

    # Step 2
    fuzzy_outputs = inference_engine.infer(fuzzy_inputs)
    print("2. Inferred Fuzzy Outputs (Aggregated):")
    print(f"   {fuzzy_outputs}")

    # Step 3
    crisp_outputs = defuzzifier.defuzzify(fuzzy_outputs)
    print("3. Defuzzified Crisp Output:")
    print(f"   Tip Percentage = {crisp_outputs['tip']:.2f}%")
    return crisp_outputs['tip']

if __name__ == "__main__":
    fz, inf, df = build_fis()
    
    # Test cases
    run_test(fz, inf, df, food_score=8.5, service_score=9.0) # Should be high tip (25%)
    run_test(fz, inf, df, food_score=2.0, service_score=3.0) # Should be low tip (5%)
    run_test(fz, inf, df, food_score=5.0, service_score=5.0) # Should be medium tip (15%)
    
    print("\nFIS Pipeline executed successfully.")
