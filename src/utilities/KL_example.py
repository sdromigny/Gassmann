import numpy as np
import matplotlib.pyplot as plt

###########################################################################################################
### Prior of nuisance and target parameters are independent of each other : p(n,m)=p(n)*p(m) ###

# Define simple models
d_obs = 0.0

# GLOBAL m (used inside p_n)
m_current = 0.0

def likelihood(m, n, sigma):
    return (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-(d_obs - (m+n))**2/(2*sigma**2))

def p_n(n):
    return (1/np.sqrt(2*np.pi)) * np.exp(-(n - m_current)**2/2)

def p_m(m):
    return (1/np.sqrt(2*np.pi)) * np.exp(-m**2/2)


n_grid = np.linspace(-6, 6, 2000)
dn = n_grid[1] - n_grid[0]

m_grid = np.linspace(-4, 4, 500)
dm = m_grid[1] - m_grid[0]

# Different regimes → different gaps
sigma_vals = [0.5, 1.0, 2.0]
colors = ['tab:blue', 'tab:orange', 'tab:green']

# Representative m for left plot
m_ref = 2.0

fig, axes = plt.subplots(1, 2, figsize=(9,3))

# Prior and posterior over the nuisance parameters
ax = axes[0]

m_current = m_ref
ax.plot(n_grid, p_n(n_grid), linewidth=4, linestyle='--', color='k', label='prior')

for sigma, c in zip(sigma_vals, colors):
    m_current = m_ref
    integrand = likelihood(m_ref, n_grid, sigma) * p_n(n_grid)
    posterior = integrand / (np.sum(integrand) * dn)

    ax.plot(n_grid, posterior, color=c, linewidth=2, label=f"posterior with σ={sigma}")

ax.set_xlabel("n")
ax.set_ylabel("Density")
ax.set_title("Prior and Posterior over $\mathbf{n}$")
ax.legend()

# Posterior over model parameters
ax = axes[1]
ax.plot(m_grid, true_post, color='k', linewidth=4, label="true posterior")

for sigma, c in zip(sigma_vals, colors):
    true_post = []
    approx_post = []

    for m in m_grid:
        m_current = m
        integrand = likelihood(m, n_grid, sigma) * p_n(n_grid)
        p_d_given_m = np.sum(integrand) * dn

        true_post.append(p_d_given_m * p_m(m))

        E_log = np.sum(p_n(n_grid) * np.log(likelihood(m, n_grid, sigma))) * dn
        approx_post.append(np.exp(E_log) * p_m(m))

    true_post = np.array(true_post)
    true_post /= np.sum(true_post) * dm

    approx_post = np.array(approx_post)
    approx_post /= np.sum(approx_post) * dm

    ax.plot(m_grid, approx_post, color=c, linewidth=2, label=f"approx σ={sigma}")

# plot true posterior once (reference)
sigma = 1.0
true_post = []

for m in m_grid:
    m_current = m
    integrand = likelihood(m, n_grid, sigma) * p_n(n_grid)
    p_d_given_m = np.sum(integrand) * dn
    true_post.append(p_d_given_m * p_m(m))

true_post = np.array(true_post)
true_post /= np.sum(true_post) * dm

ax.set_xlabel("m")
ax.set_ylabel("Density")
ax.set_title("Posterior over $\mathbf{m}$")


ax.legend()

plt.tight_layout()
plt.show()

###########################################################################################################
### Prior of nuisance and target parameters are dependent of each other : p(n,m)=p(m)*p(n|m) ###

import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ["DejaVu Serif"]  # or 'DejaVu Serif', 'Computer Modern'

# -----------------------------
# Model
# -----------------------------
d_obs = 0.0

def likelihood(m, n, sigma):
    return (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-(d_obs - (m+n))**2/(2*sigma**2))

# NEW: conditional prior p(n | m)
def p_n_given_m(n, m):
    return (1/np.sqrt(2*np.pi)) * np.exp(-(n - m)**2/2)

def p_m(m):
    return (1/np.sqrt(2*np.pi)) * np.exp(-m**2/2)

# -----------------------------
# Grids
# -----------------------------
n_grid = np.linspace(-6, 6, 2000)
dn = n_grid[1] - n_grid[0]

m_grid = np.linspace(-4, 4, 500)
dm = m_grid[1] - m_grid[0]

sigma_vals = [0.5, 1.0, 2.0]
colors = ['tab:blue', 'tab:orange', 'tab:green']

m_ref = 2.0

fig, axes = plt.subplots(1, 2, figsize=(9,3))

# -----------------------------
# LEFT: prior vs posterior over n (conditional!)
# -----------------------------
ax = axes[0]

# conditional prior at m_ref
prior_nm = p_n_given_m(n_grid, m_ref)
ax.plot(n_grid, prior_nm, linewidth=4, linestyle='--', color='k', label='prior p(n|m_ref)')

for sigma, c in zip(sigma_vals, colors):
    integrand = likelihood(m_ref, n_grid, sigma) * p_n_given_m(n_grid, m_ref)
    posterior = integrand / (np.sum(integrand) * dn)

    ax.plot(n_grid, posterior, color=c, linewidth=2, label=f"posterior σ={sigma}")

ax.set_xlabel("n")
ax.set_ylabel("Density")
ax.set_title("b) Prior and Posterior over n")
ax.legend()

# -----------------------------
# RIGHT: posterior over m
# -----------------------------
ax = axes[1]

for sigma, c in zip(sigma_vals, colors):
    true_post = []
    approx_post = []

    for m in m_grid:
        # TRUE marginal likelihood using p(n|m)
        integrand = likelihood(m, n_grid, sigma) * p_n_given_m(n_grid, m)
        p_d_given_m = np.sum(integrand) * dn

        true_post.append(p_d_given_m * p_m(m))

        # Approximation: expectation under p(n|m)
        weights = p_n_given_m(n_grid, m)
        weights /= np.sum(weights) * dn  # normalize (important numerically)

        E_log = np.sum(weights * np.log(likelihood(m, n_grid, sigma))) * dn
        approx_post.append(np.exp(E_log) * p_m(m))

    true_post = np.array(true_post)
    true_post /= np.sum(true_post) * dm

    approx_post = np.array(approx_post)
    approx_post /= np.sum(approx_post) * dm

    # plot both
    ax.plot(m_grid, true_post, linestyle='--', color=c, linewidth=4, label=f"true σ={sigma}")
    ax.plot(m_grid, approx_post, color=c, linewidth=2, label=f"approx σ={sigma}")

ax.set_xlabel("m")
ax.set_ylabel("Density")
ax.set_title("a) Posterior over m")
ax.legend()

plt.tight_layout()
plt.show()
