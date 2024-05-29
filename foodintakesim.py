import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class ModelParameters:
    def __init__(self):
        # Simulation time parameters
        self.t0 = 0  # starting time of simulation (min)
        self.tend = 1440  # ending time of simulation (min)
        self.t_delta = 2  # interval between each simulation (min)
        
        # Parameters for Physical Activity
        self.ETL = 300  # Time lower bound of activity period (min)
        self.ETU = 1260  # Time upper bound of activity period (min)

        # Parameters for Availability of Food
        self.w_snacks = 0.047  # Weight for snacks
        self.w_M_peak = [0.92, 0.45, 0.65, 0.65]  # Weight for meals peaks
        self.t_M_L = [260, 580, 600, 600]  # Time lower bound for meals (minutes)
        self.t_M_U = [1000, 1400, 1400, 1440]  # Time upper bound for meals (minutes)
        self.t_M_mu = [500, 750, 610, 1140]  # Mean time for meals (minutes)
        self.t_M_sigma = [85, 20, 240, 87]  # Standard deviation for meal times (minutes)

        # Initial conditions
        self.L0 = 178  # Initial Ghrelin concentration (pM)
        self.G0 = 5  # Initial Glucose concentration (mM)
        self.S0 = 0  # Initial stomach food content (g)

        # Body Weight
        self.W = 62  # Body Weight (kg)

        # Constants for Appetite
        self.A_max = 300  # Maximum appetite
        self.L_A50 = 120  # Ghrelin plasma concentration when appetite is half of its maximum (pM)
        self.lambda_AG = 0.3  # Factor representing the effect of decreasing glucose concentration on appetite (/mM)

        # Constants for switching probabilities
        self.rho_HA = 0.01  # Conversion factor from habits and appetite weight to probability intensity of switching to eating (/min)
        self.w_A = 0.1  # Relative weight of Appetite (versus Habits) in inducing eating behavior (#)
        self.k_ij0 = 0.001  # Baseline probability intensity of stopping eating (/min)
        self.rho_S_ij = 0.0001  # Conversion factor from Stomach contents to probability intensity of switching to fasting (/min/g)
        self.rho_G_ij = 0.0001  # Conversion factor from Glycemia to probability intensity of switching to fasting (/min/mM)

        # Constants for glucose model
        self.rho_GS = 0.9  # Glucose bioavailability from food (#)
        self.eta_G = 0.2  # Glucose contribution from food (g/g)
        self.V_G = 12.4  # Glucose distribution volume (L)
        self.k_XG = 0.0072  # Glucose apparent linear elimination rate (/min)
        self.k_XGE = 0.0036  # Additional glucose clearance rate during exercise (/min)
        self.k_G = 0.4464  # Rate of constant entry of glucose into plasma (mmol/min)
        self.k_XS = 0.0154033  # Stomach emptying rate (/min)
        self.k_S = 16.5  # Rate of food intake when eating (g/min)

        # Constants for Ghrelin model
        self.k_LS_max = 4.16  # Maximum rate of Ghrelin production (mM/min)
        self.lambda_LS = 0.00462098  # Rate of decrease in Ghrelin secretion due to amount in stomach (/g)
        self.k_XL = 0.02  # Rate constant for Ghrelin clearance (/min)
        self.H_max = 208  # Ghrelin level at maximum (pM)
        self.S_50 = 150  # Stomach content at 50% Ghrelin secretion (g)
        self.H_ss = 178  # Ghrelin level at steady state (pM)
        self.G_ss = 5  # Glucose plasma concentration at steady state (mM)

def E(t, params):
    if params.ETL <= t % 1440 <= params.ETU:
        return 1
    else:
        return 0

def H(t, params):
    snacks = params.w_snacks
    meals = sum(params.w_M_peak[m] * np.exp(-0.5 * ((t - params.t_M_mu[m]) / params.t_M_sigma[m]) ** 2)
                if params.t_M_L[m] <= t <= params.t_M_U[m] else 0 for m in range(4))
    return snacks + meals

def dL_dt(L, t, S, params):
    return params.k_LS_max * np.exp(-params.lambda_LS * S) - params.k_XL * L

def dG_dt(G, t, E, S, params):
    return - (params.k_XG + params.k_XGE * E(t, params)) * G + (params.k_G + params.k_XS * params.eta_G * params.rho_GS * S) / params.V_G

def A(L, G, params):
    return params.A_max * L / (params.L_A50 + L) * np.exp(-params.lambda_AG * G)

def k_ji(t, H, A, params):
    return params.rho_HA * H * (1 + params.w_A * A / params.A_max)

def k_ij(t, S, G, params):
    return params.k_ij0 + params.rho_S_ij * S + params.rho_G_ij * G

def simulate(params):
    np.random.seed(42)  # Set RNG seed for reproducibility
    t_max = params.tend  # Total time in minutes (1 day)
    dt = params.t_delta  # Time step in minutes
    t = np.arange(params.t0, t_max, dt)

    # Initial conditions
    L = params.L0
    G = params.G0
    S = params.S0
    chi_i = 0

    L_values = []
    G_values = []
    S_values = []
    chi_i_values = []
    A_values = []
    H_values = []

    for time in t:
        # Calculate habits (H)
        H_current = H(time, params)
        H_values.append(H_current)

        # Update indicator function chi_i
        u = np.random.uniform()
        if chi_i == 0:
            if u <= 1 - np.exp(-k_ji(time, H_current, A(L, G, params), params) * dt):
                chi_i = 1
        else:
            if u <= 1 - np.exp(-k_ij(time, S, G, params) * dt):
                chi_i = 0

        # Calculate food intake quantity Q
        Q = chi_i * params.k_S * dt

        # Update L, G, S using ODE integrator
        L = odeint(dL_dt, L, [time, time + dt], args=(S, params))[-1]
        G = odeint(dG_dt, G, [time, time + dt], args=(E, S, params))[-1]
        S = S + (-params.k_XS * S + chi_i * params.k_S) * dt

        # Store values
        L_values.append(L)
        G_values.append(G)
        S_values.append(S)
        chi_i_values.append(chi_i)
        A_values.append(A(L, G, params))

    return t, L_values, G_values, S_values, chi_i_values, A_values, H_values


def plot_results(t_hours, L_values, G_values, S_values, chi_i_values, H_values, A_values):
    # Colorblind-friendly colors from the CUD palette
    color_green = '#009E73'
    color_blue = '#0072B2'
    color_red = '#D55E00'
    color_purple = '#CC79A7'
    color_orange = '#E69F00'
    color_brown = '#F0E442'

    # Plotting results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Glucose, Ghrelin, and Stomach Volume on one graph
    ax1.plot(t_hours, G_values, label='Glucose (mM)', color=color_green)
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Glucose (mM)')
    ax1.set_title('Ghrelin, Glucose, and Stomach Volume')
    ax1.legend(loc='upper left')

    ax1b = ax1.twinx()
    ax1b.plot(t_hours, L_values, label='Ghrelin (pM)', color=color_blue)
    ax1b.plot(t_hours, S_values, label='Stomach Volume (g)', color=color_red)
    ax1b.set_ylabel('Ghrelin (pM) and Stomach Volume (g)')
    ax1b.legend(loc='upper right')

    # Adding horizontal bars to the left y-axis (Glucose)
    y_ticks = ax1.get_yticks()
    for y_value in y_ticks:
        ax1.axhline(y=y_value, color='tab:gray', linestyle='--', linewidth=0.5)

    # Intake State, Habits, and Appetite on another graph
    ax2.plot(t_hours, chi_i_values, label='Intake State (chi_i)', color=color_purple)
    ax2.plot(t_hours, H_values, label='Habits (H)', color=color_orange)
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Intake State (chi_i) and Habits (H)')
    ax2.set_title('Intake State, Habits, and Appetite')
    ax2.legend(loc='upper left')

    ax2b = ax2.twinx()
    ax2b.plot(t_hours, A_values, label='Appetite', color=color_brown)
    ax2b.set_ylabel('Appetite')
    ax2b.legend(loc='upper right')

    # Adding horizontal bars to the left y-axis (Intake State and Habits)
    y_ticks = ax2.get_yticks()
    for y_value in y_ticks:
        ax2.axhline(y=y_value, color='tab:gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()


def main():
    params = ModelParameters()
    t, L_values, G_values, S_values, chi_i_values, A_values, H_values = simulate(params)
    
    # Convert time from minutes to hours for display
    t_hours = t / 60
    
    # Plot the results
    plot_results(t_hours, L_values, G_values, S_values, chi_i_values, H_values, A_values)

if __name__ == "__main__":
    main()

