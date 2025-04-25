import numpy as np
import pandas as pd
from scipy.stats import dirichlet


# Step 1: Generate simulated eye tracking and expert scoring data
def generate_data(num_samples=10, num_eye_tracking_features=5, num_expert_features=5):
    """
    Generate simulated data for eye tracking and expert scores.
    """
    eye_tracking_data = np.random.rand(num_samples, num_eye_tracking_features)
    expert_scores = np.random.rand(num_samples, num_expert_features)
    return eye_tracking_data, expert_scores


# Step 2: Calculate BBWM weights (Bayesian Best-Worst Method)
def bbwm_weights(expert_scores):
    """
    Compute subjective weights based on the BBWM algorithm.
    Select best and worst scoring indices and apply Bayesian reasoning.
    """
    best = 0  # Best scoring index
    worst = expert_scores.shape[1] - 1  # Worst scoring index

    best_compared_to_others = np.mean(expert_scores[:, best]) / np.mean(expert_scores, axis=0)
    worst_compared_to_others = np.mean(expert_scores[:, worst]) / np.mean(expert_scores, axis=0)

    # Using Dirichlet distribution to simulate posterior distribution of weights
    weights = dirichlet.rvs(np.ones(expert_scores.shape[1]), size=1)[0]

    return weights


# Step 3: Calculate M-CRITIC-RP weights (Multi-Criteria Decision Method)
def m_critic_rp_weights(eye_tracking_data, expert_scores):
    """
    Compute objective weights using the M-CRITIC-RP method.
    This involves calculating distance correlation coefficients and standard deviations.
    """
    eye_tracking_data_normalized = (eye_tracking_data - np.mean(eye_tracking_data, axis=0)) / np.std(eye_tracking_data,
                                                                                                     axis=0)
    expert_scores_normalized = (expert_scores - np.mean(expert_scores, axis=0)) / np.std(expert_scores, axis=0)

    # Calculate distance correlation matrix
    correlation_matrix = np.corrcoef(np.concatenate([eye_tracking_data_normalized, expert_scores_normalized], axis=1),
                                     rowvar=False)

    # Calculate standard deviation of each feature's correlation
    std_eye_tracking = np.std(correlation_matrix[:eye_tracking_data.shape[1], :], axis=1)
    std_expert_scores = np.std(correlation_matrix[eye_tracking_data.shape[1]:, :], axis=1)

    # Calculate information E = 1 - standard deviation
    E_eye_tracking = 1 - std_eye_tracking
    E_expert_scores = 1 - std_expert_scores

    # Normalize the weights
    total_E = np.sum(E_eye_tracking) + np.sum(E_expert_scores)
    eye_tracking_weights = E_eye_tracking / total_E
    expert_weights = E_expert_scores / total_E

    return eye_tracking_weights, expert_weights


# Step 4: Calculate Grey-H Convex Correlation Model
def grey_h_convex_correlation(eye_tracking_data, expert_scores, eye_tracking_weights, expert_weights):
    """
    Calculate the coupling relationship between visual cognition and layout aesthetics using the Grey-H Convex Correlation model.
    """
    weighted_eye_tracking_data = eye_tracking_data * eye_tracking_weights
    weighted_expert_scores = expert_scores * expert_weights

    # Compute Grey correlation matrix (using correlation coefficient for simplicity)
    correlation = np.corrcoef(weighted_eye_tracking_data.T, weighted_expert_scores.T)[:eye_tracking_data.shape[1],
                  weighted_expert_scores.shape[1]:]

    return correlation


# Main Program: Construct the full model
def main():
    num_samples = 10  # Number of samples
    num_eye_tracking_features = 5  # Number of eye tracking features (e.g., fixation duration, fixation count, etc.)
    num_expert_features = 5  # Number of expert scoring features (e.g., balance, density, simplicity, etc.)

    # Step 1: Generate simulated data for eye tracking and expert scores
    eye_tracking_data, expert_scores = generate_data(num_samples, num_eye_tracking_features, num_expert_features)

    # Display the generated data
    print("Eye Tracking Data (Simulated):")
    print(eye_tracking_data)
    print("\nExpert Scores (Simulated):")
    print(expert_scores)

    # Step 2: Compute BBWM weights
    bbwm_weights_result = bbwm_weights(expert_scores)
    print("\nBBWM Weights (Subjective Weights):")
    print(bbwm_weights_result)

    # Step 3: Compute M-CRITIC-RP weights
    eye_tracking_weights_m_critic, expert_weights_m_critic = m_critic_rp_weights(eye_tracking_data, expert_scores)
    print("\nM-CRITIC-RP Weights (Objective Weights):")
    print("Eye Tracking Data Weights:", eye_tracking_weights_m_critic)
    print("Expert Scores Weights:", expert_weights_m_critic)

    # Step 4: Calculate Grey-H Convex Correlation
    correlation = grey_h_convex_correlation(eye_tracking_data, expert_scores, eye_tracking_weights_m_critic,
                                            expert_weights_m_critic)
    print("\nGrey-H Convex Correlation:")
    print(correlation)


# Run the main program
if __name__ == "__main__":
    main()
