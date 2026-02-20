
import sys
from scipy.stats import norm

def calculate_approval_probability(predicted_arg: float, cutoff_score: float, rmse: float = 13.49) -> float:
    # Copied from src/pas_intelligence/statistics.py
    probability = 1 - norm.cdf(cutoff_score, loc=predicted_arg, scale=rmse)
    return float(probability)

def debug():
    gap = 18.3
    cutoff = 100.0
    predicted = cutoff + gap
    rmse = 13.49
    
    prob = calculate_approval_probability(predicted, cutoff, rmse)
    print(f"Predicted: {predicted}")
    print(f"Cutoff: {cutoff}")
    print(f"Gap: {gap}")
    print(f"RMSE: {rmse}")
    print(f"Probability: {prob}")
    print(f"Chance (%): {prob * 100}")

    # Test inverse case just in case
    prob_inv = calculate_approval_probability(cutoff, predicted, rmse)
    print(f"\nInverse (Pred=Cutoff, Cutoff=Pred):")
    print(f"Probability: {prob_inv}")
    print(f"Chance (%): {prob_inv * 100}")

if __name__ == "__main__":
    debug()
