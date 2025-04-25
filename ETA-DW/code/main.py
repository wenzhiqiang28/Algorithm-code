import numpy as np
from scipy.stats import dirichlet
from .utils import generate_data, bbwm_weights, m_critic_rp_weights, grey_h_convex_correlation


def main():
    num_samples = 10
    num_eye_tracking_features = 5
    num_expert_features = 5

    # Generate simulated data
    eye_tracking_data, expert_scores = generate_data(num_samples, num_eye_tracking_features, num_expert_features)

    # Print generated data
    print("Eye Tracking Data (Simulated):")
    print(eye_tracking_data)
    print("\nExpert Scores (Simulated):")
    print(expert_scores)

    # Compute BBWM weights
    bbwm_weights_result = bbwm_weights(expert_scores)
    print("\nBBWM Weights (Subjective Weights):")
    print(bbwm_weights_result)

    # Compute M-CRITIC-RP weights
    eye_tracking_weights_m_critic, expert_weights_m_critic = m_critic_rp_weights(eye_tracking_data, expert_scores)
    print("\nM-CRITIC-RP Weights (Objective Weights):")
    print("Eye Tracking Data Weights:", eye_tracking_weights_m_critic)
    print("Expert Scores Weights:", expert_weights_m_critic)

    # Calculate Grey-H Convex Correlation
    correlation = grey_h_convex_correlation(eye_tracking_data, expert_scores, eye_tracking_weights_m_critic,
                                            expert_weights_m_critic)
    print("\nGrey-H Convex Correlation:")
    print(correlation)


if __name__ == "__main__":
    main()
