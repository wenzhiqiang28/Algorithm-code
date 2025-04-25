import numpy as np
from scipy.stats import dirichlet


def generate_data(num_samples=10, num_eye_tracking_features=5, num_expert_features=5):
    """
    Generate simulated data for eye tracking and expert scores.
    """
    eye_tracking_data = np.random.rand(num_samples, num_eye_tracking_features)
    expert_scores = np.random.rand(num_samples, num_expert_features)
    return eye_tracking_data, expert_scores


def bbwm_weights(expert_scores):
    """
    Compute subjective weights based on the BBWM algorithm.
    Select best and worst scoring indices and apply Bayesian reasoning.
    """
    best = 0
    worst = expert_scores.shape[1] - 1

    best_compared_to_others = np.mean(expert_scores[:, best]) / np.mean(expert_scores, axis=0)
    worst_compared_to_others = np.mean(expert_scores[:, worst]) / np.mean(expert_scores, axis=0)

    # Using Dirichlet distribution to simulate posterior distribution of weights
    weights = dirichlet.rvs(np.ones(expert_scores.shape[1]), size=1)[0]

    return weights


def m_critic_rp_weights(eye_tracking_data, expert_scores):
    """
    Compute objective weights using the M-CRITIC-RP method.
    This involves calculating distance correlation coefficients and standard deviations.
    """
    eye_tracking_data_normalized = (eye_tracking_data - np.mean(eye_tracking_data, axis=0)) / np.std(eye_tracking_data,
                                                                                                     axis=0)
    expert_scores_normalized = (expert_scores - np.mean(expert_scores, axis=0)) / np.std(expert_scores, axis=0)

    correlation_matrix = np.corrcoef(np.concatenate([eye_tracking_data_normalized, expert_scores_normalized], axis=1),
                                     rowvar=False)

    std_eye_tracking = np.std(correlation_matrix[:eye_tracking_data.shape[1], :], axis=1)
    std_expert_scores = np.std(correlation_matrix[eye_tracking_data.shape[1]:, :], axis=1)

    E_eye_tracking = 1 - std_eye_tracking
    E_expert_scores = 1 - std_expert_scores

    total_E = np.sum(E_eye_tracking) + np.sum(E_expert_scores)
    eye_tracking_weights = E_eye_tracking / total_E
    expert_weights = E_expert_scores / total_E

    return eye_tracking_weights, expert_weights


def grey_h_convex_correlation(eye_tracking_data, expert_scores, eye_tracking_weights, expert_weights):
    """
    Calculate the coupling relationship between visual cognition and layout aesthetics using the Grey-H Convex Correlation model.
    """
    weighted_eye_tracking_data = eye_tracking_data * eye_tracking_weights
    weighted_expert_scores = expert_scores * expert_weights

    correlation = np.corrcoef(weighted_eye_tracking_data.T, weighted_expert_scores.T)[:eye_tracking_data.shape[1],
                  weighted_expert_scores.shape[1]:]

    return correlation
