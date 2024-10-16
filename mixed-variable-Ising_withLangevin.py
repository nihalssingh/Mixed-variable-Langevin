import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(32131)

# Params: Continuous and Binary Ising
N_continuous = 2          # Number of continuous units
N_binary = 2              # Number of binary units
N = N_continuous + N_binary

J = np.random.randn(N, N) 
J = (J + J.T) / 2          
np.fill_diagonal(J, 0)  

h = 0.0  
lambda_pot = 2           # Make it expensive to be unbounded
num_steps = int(2e4)      
burn_in = int(1e4)        # Burn-in period
delta_t = 0.01            # Time step size for Langevin dynamics
temperature = 0.1         # Lower temperature for less exploration of high-energy states
gamma = 0.1               # Friction coefficient
resolution = 0.01          # Resolution for state space discretization
bound = 1.0               # Bound for the continuous variables
boxdim = 2

def potential(s):
    return lambda_pot * (s**2)/2

def potential_derivative(s):
    return lambda_pot * s

def energy(s_cont, s_bin):
    s = np.concatenate([s_cont, s_bin])
    interaction_energy = -0.5 * np.sum(J * np.outer(s, s))
    magnetic_energy = -h * np.sum(s)
    potential_energy = np.sum(potential(s[:N_continuous]))
    return interaction_energy + magnetic_energy + potential_energy

def gradient_energy(s_cont, s_bin):
    """Gradient of the energy function."""
    s = np.concatenate([s_cont, s_bin])
    interaction_grad = -np.dot(J, s)
    magnetic_grad = -h * np.ones_like(s)
    potential_grad = np.concatenate([potential_derivative(s[:N_continuous]), np.zeros(N_binary)])
    return interaction_grad + magnetic_grad + potential_grad

def langevin_step(s_cont, s_bin, aux_vars, velocities):
    """Single step of Langevin dynamics updating each component iteratively."""
    grad = gradient_energy(s_cont, s_bin)
    for i in range(N_continuous):
        noise = np.sqrt(2.0 * temperature * gamma / delta_t) * np.random.randn()
        s_cont[i] = s_cont[i] - gamma * grad[i] * delta_t + noise * np.sqrt(delta_t)
    
    for i in range(N_binary):
        aux_vars[i] += velocities[i] * delta_t
        if np.abs(aux_vars[i]) >= 1.0:
            aux_vars[i] = 0  
            proposed_state = 1 - s_bin[i]
            current_energy = energy(s_cont, s_bin)
            s_bin[i] = proposed_state
            proposed_energy = energy(s_cont, s_bin)
            accept_prob = min(1, np.exp(-(proposed_energy - current_energy)))
            if np.random.rand() >= accept_prob:
                s_bin[i] = 1 - proposed_state  # Reject the proposal and revert back
                
    return s_cont, s_bin, aux_vars

def langevin_sampling():
    s_cont = np.random.randn(N_continuous)  # Initial spin configuration for continuous variables
    s_bin = np.random.randint(2, size=N_binary)  # Initial spin configuration for binary variables
    aux_vars = np.random.uniform(-1, 1, N_binary)  # Auxiliary variables for binary dynamics
    velocities = np.random.randn(N_binary)  # Velocities for auxiliary variables
    samples_cont = []
    samples_bin = []
    energies = []
    for step in range(num_steps):
        s_cont, s_bin, aux_vars = langevin_step(s_cont, s_bin, aux_vars, velocities)  # Update s iteratively
        if step >= burn_in:  # Start recording after burn-in
            samples_cont.append(s_cont.copy())
            samples_bin.append(s_bin.copy())
            energies.append(energy(s_cont, s_bin))
    return np.array(samples_cont), np.array(samples_bin), np.array(energies)

# Run Langevin sampling
start_time = time.time()
samples_cont, samples_bin, energies = langevin_sampling()

# Discretize the samples to bin them by resolution
binned_samples_cont = np.floor(samples_cont / resolution).astype(int)

# Find the most frequent state in the binned samples
unique_states_cont, counts_cont = np.unique(binned_samples_cont, axis=0, return_counts=True)
most_frequent_state_cont = unique_states_cont[np.argmax(counts_cont)]
most_frequent_state_real_cont = most_frequent_state_cont * resolution  # Convert back to real state space

# Find the state with the lowest energy
min_energy_index = np.argmin(energies)
state_with_min_energy_cont = samples_cont[min_energy_index]
state_with_min_energy_bin = samples_bin[min_energy_index]

# Plot histogram of visited binary states
def plot_binary_state_histogram(samples_bin, energies, temperature):
    indices = np.dot(samples_bin, 2**np.arange(samples_bin.shape[1]))
    unique_states, counts = np.unique(indices, return_counts=True)
    
    # Calculate Boltzmann probabilities for unique binary states
    boltzmann_probs = np.zeros_like(counts, dtype=float)
    for i, state in enumerate(unique_states):
        state_bin = [(state >> j) & 1 for j in range(N_binary)]
        state_bin = np.array(state_bin[::-1]) 
        boltzmann_probs[i] = np.exp(-energy(np.zeros(N_continuous), state_bin) / temperature)
    boltzmann_probs /= np.sum(boltzmann_probs)
    
    plt.figure(figsize=(12, 6))
    plt.bar(unique_states, boltzmann_probs, alpha=0.5, label='Boltzmann Probabilities')
    plt.bar(unique_states, counts / np.sum(counts), alpha=0.5, label='Observed Frequencies')
    plt.xlabel('Binary State Index')
    plt.ylabel('Probability')
    plt.title('Histogram of Visited Binary States')
    plt.legend()
    plt.show()

plot_binary_state_histogram(samples_bin, energies, temperature)

# Plot visited continuous states over the course of Langevin dynamics
def plot_continuous_state_distribution(samples_cont, state_with_min_energy_bin):
    plt.figure(figsize=(12, 6))
    sns.kdeplot(x=samples_cont[:, 0], y=samples_cont[:, 1], cmap='viridis', fill=True, bw_adjust=0.5)
    plt.scatter(samples_cont[:, 0], samples_cont[:, 1], color='red', s=1, label='Visited States')
    plt.xlabel('s1')
    plt.ylabel('s2')
    plt.title('Visited Continuous States')
    plt.legend()
    plt.show()

# Fix binary variables to state with minimum energy
fixed_binary_state = state_with_min_energy_bin
plot_continuous_state_distribution(samples_cont, fixed_binary_state)

# Calculate and plot the expected Boltzmann distribution for the continuous variables given fixed binary state
def plot_expected_boltzmann_distribution(J, h, lambda_pot, temperature, resolution, boxdim, fixed_binary_state):
    x = np.arange(-boxdim, boxdim + resolution, resolution)
    y = np.arange(-boxdim, boxdim + resolution, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            s_cont = np.array([X[i, j], Y[i, j]])
            Z[i, j] = np.exp(-energy(s_cont, fixed_binary_state) / temperature)
    
    Z /= np.sum(Z)  # Normalize to get probabilities

    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(Z, cmap='viridis')
    xticks = [0, len(x)//2, len(x)-1]
    yticks = [0, len(y)//2, len(y)-1]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(np.round([x[t] for t in xticks], 2))
    ax.set_yticklabels(np.round([y[t] for t in yticks], 2))
    
    plt.xlabel('s1')
    plt.ylabel('s2')
    plt.title('Expected Boltzmann Distribution for Continuous Variables')
    plt.show()

plot_expected_boltzmann_distribution(J, h, lambda_pot, temperature, resolution, boxdim, fixed_binary_state)

print("--- %s seconds ---" % (time.time() - start_time))

print("Most Frequent State (Discretized):", most_frequent_state_real_cont)
print("State with Minimum Energy (Continuous):", state_with_min_energy_cont)
print("State with Minimum Energy (Binary):", state_with_min_energy_bin)

print("Minimum Energy Value:", energies[min_energy_index])
print("Frequency of Most Frequent State:", counts_cont[np.argmax(counts_cont)])
print("Total Unique States (Continuous):", len(unique_states_cont))
