# IC–IIa — Informational Calculus IIa (Dynamic Coherence Layer)
# Copyright (c) 2025 Adolfo J. Céspedes Jiménez
# Licensed under the MIT License. See the LICENSE file in this repository for details.

IC–IIa — Minimal Reference Implementation (Aligned with Manuscript Example)

This module implements the core dynamic components of the Informational Calculus IIa (IC–IIa)
as described in:

  Céspedes Jiménez, A. J. (2025).
  "IC–IIa: Formal Consolidation of the Informational Dynamic Calculus
   in the Theory of Informational Emergence (TIE)."

It includes a numerical example that reproduces:

    cos(I_s, I_m)  ≈ 0.992
    |∂ᵢ I_s|       ≈ 0.224
    C_t            ≈ 0.733

exactly as shown (rounded) in the manuscript.
"""

from __future__ import annotations
import numpy as np


def sigmoid(x: float) -> float:
    """Logistic function σ(x) = 1 / (1 + e^{-x})."""
    return 1.0 / (1.0 + float(np.exp(-x)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity cos(a, b) for non-zero vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def delta_async(I_s_t: np.ndarray, I_m_t_delta: np.ndarray) -> float:
    r"""
    Asynchronous inter-systemic difference:

        Δ_async(t) = || I_s(t) - I_m(t + δ) ||
    """
    I_s_t = np.asarray(I_s_t, dtype=float)
    I_m_t_delta = np.asarray(I_m_t_delta, dtype=float)
    return float(np.linalg.norm(I_s_t - I_m_t_delta))


def coherence_ct(
    I_s_t: np.ndarray,
    I_m_t_delta: np.ndarray,
    I_s_prev: np.ndarray,
    alpha: float = 1.2,
    beta: float = 0.8,
    gamma: float = 0.0,  # γ = 0 en el ejemplo numérico del manuscrito
) -> float:
    r"""
    Extended dynamic coherence function 𝒞_t (example version):

        𝒞_t = σ( α cos(I_s(t), I_m(t+δ))
                 - β |∂ᵢ I_s(t)|
                 - γ Δ_async(t) )

    Para el ejemplo numérico del manuscrito fijamos γ = 0, de modo que:

        𝒞_t = σ( α cos(I_s, I_m) - β |∂ᵢ I_s| )
    """
    I_s_t = np.asarray(I_s_t, dtype=float)
    I_s_prev = np.asarray(I_s_prev, dtype=float)
    I_m_t_delta = np.asarray(I_m_t_delta, dtype=float)

    # informational differential ∂ᵢ I_s
    dI = I_s_t - I_s_prev
    dI_norm = float(np.linalg.norm(dI))

    # semantic similarity
    sim = cosine_similarity(I_s_t, I_m_t_delta)

    # asynchronous difference (no contribuye si γ = 0)
    delta = delta_async(I_s_t, I_m_t_delta)

    x = alpha * sim - beta * dI_norm - gamma * delta
    return sigmoid(x)


def minimal_repair(
    I_s_t: np.ndarray,
    I_m_t_delta: np.ndarray,
    eta: float = 0.5,
) -> np.ndarray:
    r"""
    Simplified constructive version of the Law of Minimal Repair:

        I_s^repaired(t) = η I_s(t) + (1 − η) I_m(t+δ)
    """
    I_s_t = np.asarray(I_s_t, dtype=float)
    I_m_t_delta = np.asarray(I_m_t_delta, dtype=float)
    eta = float(eta)
    return eta * I_s_t + (1.0 - eta) * I_m_t_delta


def run_single_step_example() -> None:
    """
    Reproduce the numerical example reported in the IC–IIa manuscript.

    Vectors:

        I_s      = (0.6, 0.8)
        I_m      = (0.4942097034, 0.8693427224)
        I_s_prev = (0.7, 0.6)

    With α = 1.2, β = 0.8, γ = 0, we obtain (rounded):

        cos(I_s, I_m)  ≈ 0.992
        |∂ᵢ I_s|       ≈ 0.224
        C_t            ≈ 0.733
    """
    # example vectors aligned with manuscript
    I_s = np.array([0.6, 0.8])
    I_m = np.array([0.494209703436419, 0.869342722422686])
    I_s_prev = np.array([0.7, 0.6])

    alpha = 1.2
    beta = 0.8
    gamma = 0.0
    Phi_low = 0.75  # example threshold, adjust as needed

    print("I_s      =", I_s)
    print("I_m      =", I_m)
    print("I_s_prev =", I_s_prev)

    sim = cosine_similarity(I_s, I_m)
    print(f"\ncos(I_s, I_m) = {sim:.3f}")  # → 0.992

    dI = I_s - I_s_prev
    dI_norm = float(np.linalg.norm(dI))
    print(f"∂ᵢ I_s      = {dI}")
    print(f"|∂ᵢ I_s|    = {dI_norm:.3f}")  # → 0.224

    delta = delta_async(I_s, I_m)
    print(f"Δ_async     = {delta:.3f}")

    C_t = coherence_ct(I_s, I_m, I_s_prev, alpha=alpha, beta=beta, gamma=gamma)
    print(f"\n𝒞_t (before repair) = {C_t:.3f}")  # ≈ 0.733

    if C_t < Phi_low:
        print(f"𝒞_t < Φ_low ({Phi_low:.2f}) → repair triggered.")
        I_s_repaired = minimal_repair(I_s, I_m, eta=0.5)
        print("I_s (repaired) =", I_s_repaired)
        C_t_repaired = coherence_ct(
            I_s_repaired, I_m, I_s_prev,
            alpha=alpha, beta=beta, gamma=gamma
        )
        print(f"𝒞_t (after repair)  = {C_t_repaired:.3f}")
    else:
        print(f"𝒞_t ≥ Φ_low ({Phi_low:.2f}) → no repair applied.")


if __name__ == "__main__":
    run_single_step_example()
