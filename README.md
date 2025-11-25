# IC–IIa — Informational Calculus IIa (Dynamic Coherence Layer)

This repository contains the minimal reference implementation of **IC–IIa**,  
the dynamic coherence calculus of the **Theory of Informational Emergence (TIE)**.

It provides:

- The extended coherence function 𝒞ₜ between **Iₛ** and **Iₘ**.
- The asynchronous inter-systemic difference **Δ_async(t)**.
- A constructive implementation of the **Law of Minimal Repair**.
- A numerical example aligned with the IC–IIa manuscript.

---

## Installation

```bash
git clone https://github.com/AdolfoJCJ/IC-IIa.git
cd IC-IIa
pip install numpy
````

---

## Usage

Run the example:

```bash
python ic_iia.py
```

This will compute:

* cos(Iₛ, Iₘ)
* ∂ᵢ Iₛ and |∂ᵢ Iₛ|
* the extended coherence 𝒞ₜ
* and apply minimal repair if 𝒞ₜ < Φ_low

---

## Files

* `ic_iia.py` — Core implementation + numerical example
* `README.md` — Documentation
* `LICENSE` — MIT License

---

## Reference

If you use this code, please cite:

Céspedes Jiménez, A. J. (2025).
*IC–IIa: Formal Consolidation of the Informational Dynamic Calculus in the Theory of Informational Emergence (TIE)*
Zenodo. https://doi.org/10.5281/zenodo.17691472
---

## License

Released under the **MIT License**.

