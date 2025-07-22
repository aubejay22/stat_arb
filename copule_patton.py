import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class CoefCorrDynamique:
    """
    Estimation d’un modèle de dépendance dynamique (copule gaussienne) selon la spécification de Patton (2006).

    Paramètres
    ----------
    u : array-like
        Pseudo-observations marginales (PIT transformées) pour la première variable.
    v : array-like
        Pseudo-observations marginales (PIT transformées) pour la deuxième variable.
    p : int
        Taille de la fenêtre pour l’estimation locale de la dépendance.

    Attributs
    ---------
    omega : float
        Paramètre constant dans la dynamique de la corrélation.
    alpha : float
        Poids de la dépendance instantanée estimée localement.
    beta : float
        Coefficient d’auto-régression de la corrélation dynamique.
    result : OptimizeResult
        Résultat complet de l’optimisation `scipy.optimize.minimize`.
    """

    def __init__(self, u, v, p):
        self.u = u
        self.v = v
        self.p = p

        self.loglik = None

    @staticmethod
    def logistic_modified(x):
        """Fonction logistique modifiée : tanh(x / 2), pour contraindre ρ ∈ (−1, 1)."""
        return np.tanh(x / 2.0)

    @staticmethod
    def logistic_modified_derivate(x):
        """Dérivé de la Fonction logistique modifiée."""
        return 0.5*(1 - np.tanh(x / 2.0)**2)

    @staticmethod
    def gaussian_copula_log_density(u, v, rho):
        """
        Log-densité de la copule gaussienne bivariée.

        Paramètres
        ----------
        u, v : float
            Observations uniformes (PIT) à un instant t.
        rho : float
            Corrélation instantanée.

        Retourne
        --------
        float
            Log-densité de la copule en (u, v) donné ρ.
        """
        x = norm.ppf(u)
        y = norm.ppf(v)
        denom = 1.0 - rho**2
        return -0.5 * np.log(denom) + (rho * x * y - 0.5 * rho**2 * (x**2 + y**2)) / denom

    def log_likelihood_patton(self, params):
        """
        Log-vraisemblance du modèle dynamique de Patton.

        Paramètres
        ----------
        params : array-like
            Paramètres [omega, alpha, beta].

        Retourne
        --------
        float
            Moins la log-vraisemblance totale (car minimisée).
        """
        omega, beta, alpha = params
        T = len(self.u)
        u_ppf = norm.ppf(self.u)
        v_ppf = norm.ppf(self.v)
        rho = np.zeros(T)
        rho[0] = np.corrcoef(u_ppf[:self.p], v_ppf[:self.p])[0, 1]
        log_lik = 0.0

        for t in range(1, T):
          #PEUT ETRE ERREURE ICI DANS LA BOUCLE VU RANGE(1, T) peut etre je dois faire self.gaussian_copula_log_density(self.u[t-1], self.v[t-1], rho[t])???
          #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
          # EST CE QU'ON DOIT VRAIMENT BOUCLÉ SUR T OU SUR LEN(U_PPF)
            empirical_corr = (
                np.mean(u_ppf[:t] * v_ppf[:t]) if t < self.p else
                np.mean(u_ppf[t - self.p:t] * v_ppf[t - self.p:t])
            )
            rho[t] = CoefCorrDynamique.logistic_modified(omega + beta * rho[t - 1] + alpha * empirical_corr)
            log_ct = CoefCorrDynamique.gaussian_copula_log_density(self.u[t], self.v[t], rho[t])
            if not np.isfinite(log_ct):
                return 1e10
            log_lik += log_ct

        return -log_lik


    def gradient_log_likelihood_patton(self, params):
        """
        Gradient de la Log-vraisemblance du modèle dynamique de Patton.

        Paramètres
        ----------
        params : array-like
            Paramètres [omega, alpha, beta].

        Retourne
        --------
        float
            array de 3 dimension
        """
        omega, beta, alpha = params
        T = len(self.u)
        u_ppf = norm.ppf(self.u)
        v_ppf = norm.ppf(self.v)
        rho = np.zeros(T)
        rho[0] = np.corrcoef(u_ppf[:self.p], v_ppf[:self.p])[0, 1]
        log_lik = np.zeros(3)

        for t in range(1, T):
            empirical_corr = (
                np.mean(u_ppf[:t] * v_ppf[:t]) if t < self.p else
                np.mean(u_ppf[t - self.p:t] * v_ppf[t - self.p:t])
            )
            rho[t] = CoefCorrDynamique.logistic_modified(omega + beta * rho[t - 1] + alpha * empirical_corr)
            log_ct = CoefCorrDynamique.gaussian_copula_log_density(self.u[t], self.v[t], rho[t])
            derivate = CoefCorrDynamique.logistic_modified_derivate(omega + beta * rho[t - 1] + alpha * empirical_corr)

            log_lik += np.array([1, rho[t-1], empirical_corr])*derivate * (rho[t]/(1-rho[t]**2) + u_ppf[t]*v_ppf[t]*(1 - rho[t]**2 - 2*derivate*rho[t]**2)/(1-rho[t]**2)**2
                                                                           + rho[t] * (u_ppf[t]**2 + v_ppf[t]**2) * (2*rho[t]**2 - 1) / (1-rho[t]**2)**2
                                                                           )
        return -log_lik

    
    def _fit(self, eps: float = 1e-3):
        """
        Estime (ω, β, α) par maximum de vraisemblance sous la contrainte
        β + α ≤ 1 − eps.

        Retour
        ------
        OptimizeResult
            Objet renvoyé par `scipy.optimize.minimize`.
        """

        # -----------------------------
        # 1) point de départ (ω, β, α)
        # -----------------------------
        theta0 = np.array([0.0, 0.90, 0.05])   # ω = 0, β = 0.90, α = 0.05

        # -----------------------------
        # 2) bornes individuelles
        #    indices : 0 = ω, 1 = β, 2 = α
        # -----------------------------
        bounds = Bounds(
            lb=[-5.0, 1e-6, 1e-6],   # ω, β, α
            ub=[ 5.0, 0.99, 0.99]
        )

        # ---------------------------------------------------
        # 3) contrainte stationnarité  : β + α ≤ 1 − eps
        #    -> mêmes indices (1 et 2) mais inversés par
        #       rapport à la version précédente
        # ---------------------------------------------------
        constr = NonlinearConstraint(
            fun=lambda t: t[1] + t[2] - (1.0 - eps),  # β + α − (1−eps) ≤ 0
            lb=-np.inf,
            ub=0.0
        )

        # ---------------------------------------------------
        # 4) optimisation interior-point « trust-constr »
        # ---------------------------------------------------

        theta0 = np.array([0.0, 0.90, 0.05])     # ω, β, α

        # 2) bornes individuelles
        bounds = Bounds(
            lb=[-5.0, 1e-6, 1e-6],               # ω, β, α
            ub=[ 5.0, 0.99, 0.99]
        )

        # 3) contrainte non linéaire  β + α ≤ 1 − eps  (forme “ineq” pour SLSQP)
        constr = {
            'type': 'ineq',
            # fun >= 0  <=>  1 - eps - (β+α)  ≥ 0
            'fun': lambda t: (1.0 - eps) - (t[1] + t[2]),
            # jacobienne associée
            'jac': lambda t: np.array([0.0, -1.0, -1.0])
        }

        # 4) appel SLSQP
        return minimize(
            fun         = self.log_likelihood_patton,         # −log-vraisemblance
            jac         = self.gradient_log_likelihood_patton,# son gradient (avec signe −)
            x0          = theta0,
            method      = 'SLSQP',
            bounds      = bounds,
            constraints = [constr],
            options     = dict(ftol=1e-9, maxiter=500, disp=False)
        )

    def get_params(self):
      result = self._fit()
      if not result.success:
        print(f"Optimisation Patton non convergée : {result.message}")
        self.loglik = None
        return None, None, None
      else:
        self.loglik = -result.fun
        return result.x

    def get_likelihood(self):
      return self.loglik

    def get_marginals(self):
      return self.u, self.v
