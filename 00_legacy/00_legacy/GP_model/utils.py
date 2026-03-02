import pickle
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def preprocess_data(filename):
    with open(filename, 'rb') as f:
        samples = pickle.load(f)
    summary = {}
    for param, arr in samples.items():
        flat = np.array(arr).reshape(-1, *arr.shape[2:])  # flatten chains and samples
        mean = flat.mean(axis=0)
        lower = np.percentile(flat, 2.5, axis=0)
        upper = np.percentile(flat, 97.5, axis=0)
        summary[param] = {
            'mean': mean,
            '2.5%': lower,
            '97.5%': upper,
            'samples': flat
        }
    return summary

def visualize_results(x, y, E, summary, title="Gaussian Process Predictions"):
    """
    Plots observed counts and GP posterior predictive mean and 95% CI for Poisson GP.
    x: region indices (1D array)
    y: observed counts (1D array)
    E: exposure (1D array)
    summary: dict with posterior samples for 'f'
    """
    f_samples = summary["f"]["samples"]  # shape: (n_samples, n_regions)
    mu_samples = E * np.exp(f_samples)   # shape: (n_samples, n_regions)
    mean_mu = mu_samples.mean(axis=0)
    lower = np.percentile(mu_samples, 2.5, axis=0)
    upper = np.percentile(mu_samples, 97.5, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, 'r.', markersize=10, label='Observed Counts')
    plt.plot(x, mean_mu, 'b-', label='GP Posterior Mean Rate')
    plt.fill_between(x, lower, upper, color='blue', alpha=0.2, label='95% CI')
    plt.title(title)
    plt.xlabel('Region Index')
    plt.ylabel('Expected Counts')
    plt.legend()
    plt.tight_layout()
    plt.ticklabel_format(style='plain', axis='y')  # <-- Force plain formatting
    plt.ylim(0, max(y.max(), mean_mu.max(), upper.max()) * 1.1)  # <-- Set sensible y-limits
    plt.grid(alpha=0.3)
    plt.savefig("GP_model/gp_posterior_mean_with_CI.png", dpi=300)
    plt.show()

def plot_violin_summary(summary):
    params = [k for k in summary if k != "f"]
    data = [summary[k]['samples'].flatten() for k in params]
    plt.figure(figsize=(1 + 2*len(params), 5))
    plt.violinplot(data, showmeans=True)
    plt.xticks(np.arange(1, len(params)+1), params)
    plt.title("Posterior Distributions (Violin Plots)")
    plt.tight_layout()
    plt.savefig("GP_model/gp_violin_summary.png", dpi=300)
    plt.show()

def save_results(filename, results):
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

def load_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    




if __name__ == "__main__":
    #from CAR_model.uk_lung_car import y, E, X -> changed for standardized X 
    from run_gp_model import y, E, X_standardized as X
    print("First 10 y:", y[:10])
    print("x shape:", np.arange(len(y)).shape, "y shape:", y.shape)
    summary = preprocess_data("gp_mcmc_samples.pkl")
    visualize_results(np.arange(len(y)), y, E, summary, title="GP Posterior Mean Rate with 95% CI")
    plot_violin_summary(summary)
    print("E shape:", E.shape, "E min/max:", E.min(), E.max())
    print("y shape:", y.shape, "y min/max:", y.min(), y.max())
    f_samples = summary["f"]["samples"]
    print("f_samples shape:", f_samples.shape, "mean:", f_samples.mean(), "min:", f_samples.min(), "max:", f_samples.max())
    mu_samples = E * np.exp(f_samples)
    print("First 10 mu_samples mean:", mu_samples.mean(axis=0)[:10])
    print("mu_samples mean:", mu_samples.mean(), "min:", mu_samples.min(), "max:", mu_samples.max())
    print("Summary statistics:")
    for param, stats in summary.items():
        print(f"{param}: mean={stats['mean']}, 2.5%={stats['2.5%']}, 97.5%={stats['97.5%']}")