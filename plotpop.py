from main import htot, x
import hoppingalg
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Run simulations and get results
    results = hoppingalg.run_simulations()

    # Extract data
    P_vals = list(results.keys())
    prob_trans_lowers = [results[p][2] for p in P_vals]
    prob_refl_lowers = [results[p][3] for p in P_vals]
    prob_trans_uppers = [results[p][4] for p in P_vals]

    fig, ax = plt.subplots(figsize=(6,5)) 
    ax.plot(P_vals, prob_trans_lowers, color = "#A0C4FF", label="Prob Trans Lower", linewidth=3) 
    ax.plot(P_vals, prob_refl_lowers, color = "#FFB3BA",  label="Prob Refl Lower", linewidth=3) 
    ax.plot(P_vals, prob_trans_uppers, color = "#A67BB5", label="Prob Trans Upper", linewidth=3) 
    ax.set_xlabel(r"momentum $P$") 
    ax.set_ylabel("Probability") 
    ax.set_title("Tully Model I - Transmission and Reflection Probabilities") 
    ax.legend(loc="upper right") 
    fig.tight_layout() 
    fig.savefig('tully1-trans-refl.png', dpi=300) 
    plt.show()
