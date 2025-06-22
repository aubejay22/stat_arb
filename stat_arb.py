
from google.colab import files
import pandas as pd

# Ouvre une fenêtre pour choisir le(s) fichier(s)
uploaded = files.upload()

# Supposons que votre fichier s’appelle « data.csv »
df = pd.read_csv('us_banks.csv')

df.head()        # aperçu des premières lignes

from datetime import datetime

full_df =df



# Étape 1 : conversion explicite en datetime (si ce n'est pas déjà le cas)
full_df['date'] = pd.to_datetime(full_df['date'], errors='coerce')
# Étape 2 : enlever le fuseau horaire SANS conversion d'heure
# (on enlève tzinfo mais on garde l'heure locale telle quelle)
full_df['date'] = full_df['date'].apply(lambda dt: dt.replace(tzinfo=None) if pd.notna(dt) else dt)


# Étape 2: sélectionner les colonnes d'intérêt
cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
full_df = full_df[cols]

# Étape 3: pivot en colonnes multiples
df_wide = full_df.pivot(index='date', columns='symbol', values=['open', 'high', 'low', 'close', 'volume'])

# Étape 4: réordonner les colonnes si tu veux (facultatif)
df_wide = df_wide.reorder_levels([1, 0], axis=1).sort_index(axis=1)

# Étape 5: fixer l'index datetime
df_wide.index = pd.to_datetime(df_wide.index)

import numpy as np
import pandas as pd
import openturns as ot


from arch import arch_model
from statsmodels.stats.diagnostic import het_arch
import openturns as ot

import numpy as np
from scipy.stats import norm, t
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


from itertools import combinations

class PairSelector:

  def __init__(self, df, is_stationnaire, is_dynamic_patton, p=25):

    #liste de tuple de toutes les pairs possibles
    self.pairs = list(combinations(df.columns.tolist(), 2))

    self.copule_dict = {col : Copule(is_stationnaire, is_dynamic_patton, p) for col in self.pairs}

    self.best_pair = self.find_best_pair(df)

  def find_best_pair(self, df):
    pairs = dico = {cle: None for cle in self.pairs}

    for pair in self.pairs:

      #extraire rendement de la pair
      j, k = pair
      returns = df[[j, k]]

      #train copule et extraire likelihood
      self.copule_dict[pair].train(returns)
      pairs[pair] = self.copule_dict[pair].get_likelihood()
    return max(pairs, key=pairs.get)

  def get_bestpair(self):
    return self.best_pair

  def get_copule(self):
    #return l'object copule meilleur pair pour pas refaire calcul
    #et l'utiliser directement dans le backtest
    return self.copule_dict[self.best_pair]




class Garch:
    """
    Univariate GARCH(1,1) volatility modeling with Student-t innovations.

    Handles training, forecasting, and standardized residual updates for multiple assets.
    """

    def __init__(self):
        """
        Initializes internal structures to store GARCH models, residuals,
        volatilities, and estimated degrees of freedom.
        """
        self.garch_models = None
        self.standardized_residuals = None

        self.vol_val = None
        self.last_residu = {}

        self.v = None  # Degrees of freedom of the Student-t distribution
        self.garch_is_converged = {}  # Convergence status of each GARCH model

        self.loglikelihood = {}

    def reset_attribut(self):
        """
        Resets internal attributes for re-training on new data.
        """
        self.garch_models = {}
        self.standardized_residuals = {}
        self.vol_val = {}
        self.last_residu = {}
        self.v = {}
        self.garch_is_converged = {}
        self.loglikelihood = {}

        self.scale = 1000 #scale data to help to numerical optimizer to converge

    def _train_garch(self, df):
        """
        Trains a GARCH(1,1)-t model for each time series.

        If the model fails to converge, falls back to constant volatility.

        :param df: DataFrame with one asset per column.
        """
        for target in df.columns:
            residu = df[target] #*self.scale
            model = arch_model(residu, mean='Zero', vol='GARCH', p=1, q=1, dist='t', rescale=False)
            garch_result = model.fit(disp='off')

            if garch_result.optimization_result.success:
                self.garch_is_converged[target] = True
                sigma_t = garch_result.conditional_volatility
                self.standardized_residuals[target] = residu / sigma_t
                self.garch_models[target] = garch_result
                self.vol_val[target] = sigma_t.iloc[-1]
                self.last_residu[target] = residu.iloc[-1]
                self.v[target] = garch_result.params['nu']
                self.loglikelihood[target] = garch_result.loglikelihood
            else:
                print("⚠️ GARCH did not converge")
                self.garch_is_converged[target] = False
                sigma_t = np.std(residu, ddof=1)
                self.standardized_residuals[target] = residu / sigma_t
                self.garch_models[target] = sigma_t
                self.v[target] = 5  # default fallback
                self.loglikelihood[target] = None

        self.standardized_residuals = pd.DataFrame(self.standardized_residuals)

    def garch_pred(self, stock, *args):
        """
        Predicts next-period volatility σ_{t+1} for a given asset.

        :param stock: Asset name (string)
        :param args: (epsilon_t, sigma_t) if model converged
        :return: Predicted volatility at t+1
        """

        if self.garch_is_converged[stock]:
            epsilon, sigma = args[:2]
            garch = self.garch_models[stock]
            params = garch.params.to_dict()
            return np.sqrt((params['omega'] + params['alpha[1]'] * epsilon**2 + params['beta[1]'] * sigma**2))
        else:
            return self.garch_models[stock]

    def _nouveau_residu(self, rendement):
        """
        Computes next-period standardized residuals for each asset.

        Uses GARCH prediction and observed returns.

        :param rendement: pd.Series of asset returns at t+1
        :return: pd.Series of standardized residuals
        """
        nouveau_residu_standardise = {}
        for asset, asset_return in rendement.items():
            if not self.garch_is_converged[asset]:
                nouveau_residu_standardise[asset] = asset_return / self.garch_pred(asset)
                continue

            residu = self.last_residu[asset]
            vol = self.vol_val[asset]
            vol_t1 = self.garch_pred(asset, residu, vol)

            #asset_return *= self.scale
            nouveau_residu_standardise[asset] = asset_return  / vol_t1
            self.last_residu[asset] = asset_return
            self.vol_val[asset] = vol_t1

        return pd.Series(nouveau_residu_standardise)

    def get_standardized_residuals(self):
        """
        :return: DataFrame of standardized residuals.
        """
        return self.standardized_residuals

    def get_doff(self):
        """
        :return: Dictionary of estimated degrees of freedom for each asset.
        """
        return self.v

    def get_loglikelihood(self):
      """
        :return: Dictionary of maximum of the loglikelihood function for each asset.
      """
      return self.loglikelihood




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




class Copule:
    def __init__(self, is_stationnaire, is_dynamic_patton, p=25):
        """
        Initialise la classe Copule permetant calculer le mispricing index au temps t

        :attribut stationnaire_ordre_2:  type::bool mettre à False si on veut répliquer seulement le papier et à True si on veut calculer avec ajout (arma-garch)
        :attribut distribution: cdf du portefeuille induite par une copule de bernstein
        p : le parametre pour le model de patton
        """
        self.is_stationnaire = is_stationnaire
        self.is_dynamic_patton = is_dynamic_patton
        self.garch = Garch()
        self.distribution = None
        self.copule = None
        self.marginals = None
        self.d_of_f = {} #liste degrés de libertés des lois de student
        self.rho_list = []
        self.rho_t_moins_1 = 0

        #données utiles pour compute likelihood
        self.training_data = None
        self.param = None

        if is_dynamic_patton:
          self.likelihood_patton = None
          self.is_stationnaire = True
          self.p = p
          self.omega, self.beta, self.alpha = 0, 0, 0
          self.u, self.v = [], [] #les u et v out of sample que je vais devoir stocker en mémoire.
          self.rho_list = []



    def train(self, df):
        """
        Entraîne la copule (Bernstein) sur les données. Si on veut que série soit stationnaire ordre 2,
        on ajuste d’abord les modèles ARMA et GARCH pour
        obtenir des résidus standardisés, puis on construit la copule sur ces résidus. Sinon on calcule
        la copule seulement sur les rendements comme dans le papier

        :param df: DataFrame contenant les données à utiliser pour l’entraînement.
        """
        if self.is_stationnaire:
          self.garch.reset_attribut()
          self.garch._train_garch(df)
          self.d_of_f = self.garch.get_doff()
          self._train_copule(self.garch.get_standardized_residuals())

          #training data
          self.training_data = self.garch.get_standardized_residuals()

        else:
          self._train_copule(df)

          #training data
          self.training_data = df



    def estimation_marginal(self, asset_name, asset_return, v):
      """
      Estime une loi de Student t pour les marges

      Parameters:
          asset_name : string, nom de l'actif
          asset_return : np.series,  Série de rendements (1D).
          v : int, Degrés de liberté de la loi de Student.

      Returns:
          ot.Student

      """
      if (not self.is_stationnaire and not self.is_dynamic_patton) or ( v is None):
        v , _, _ = t.fit(asset_return, floc=0.0, fscale=1.0)
        self.d_of_f[asset_name] = v
      return ot.Student(v, 0, 1)


    def PIT(self, x, y):
      #probability integral transform
       u = np.array([self.marginals[0].computeCDF(xi) for xi in x])
       v = np.array([self.marginals[1].computeCDF(yi) for yi in y])

       CLIP = 1e-6
       u_raw = np.clip(u, CLIP, 1-CLIP)
       v_raw = np.clip(v, CLIP, 1-CLIP)
       return u_raw, v_raw


    def _train_copule(self, data):
        """
        Construit une copule de Bernstein basée sur les rendements ou les rendements transformer pour atteindre stationarité ordre 2.
        Chaque variable suit une marginale estimée normal si stationaire ordre 2 sinon empirique.
        Ensuite, on créer la cdf du portefeuille dans l'attribut distribution

        :param data: pd.DataFrame contenant les échantillons (une colonne par actif).
        """
        data = data.dropna()

        if data.shape[0] == 0:
            raise ValueError("Plus de données après suppression des NaN. Impossible d’entraîner la copule.")

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #vérifier ici pourquoi je prends des samples ???
        #pourquoi pas prendre x ou y ?
        x, y = data.iloc[:,0].to_numpy(), data.iloc[:,1].to_numpy()

        if self.is_dynamic_patton:
          #entrainé le model dynamique
          #premiere étape les marginals
          self.marginals = [self.estimation_marginal(asset_name, data[asset_name], self.d_of_f[asset_name]) for asset_name in data.columns]

          #estimation parametre dynamique copule
          u, v = self.PIT(x, y)
          self.rho_t_moins_1 = np.corrcoef(u, v)[0, 1]
          copule_dyn =  CoefCorrDynamique(u, v, self.p)
          self.omega, self.beta, self.alpha = copule_dyn.get_params()
          self.likelihood_patton = copule_dyn.get_likelihood()


        elif self.is_stationnaire: #selement marge stationnaire
          self.marginals = [self.estimation_marginal(asset_name, data[asset_name], self.d_of_f[asset_name]) for asset_name in data.columns]

          #estimation parametre copule
          u, v = self.PIT(x, y)
          z_u, z_v = norm.ppf(u), norm.ppf(v)
          self.rho_t_moins_1 = np.corrcoef(z_u, z_v)[0, 1]
          self.copule_static()#création copule static
          """
          self.copule_static(data)#création copule static
          """
        else:
          #self.marginals = [self.estimation_marginal(asset_name, data[asset_name], None) for asset_name in data.columns]
          self.marginals = [self.estimation_marginal(asset_name, data[asset_name], None) for asset_name in data.columns]
          #estimation parametre copule
          u, v = self.PIT(x, y)
          z_u, z_v = norm.ppf(u), norm.ppf(v)
          self.rho_t_moins_1 = np.mean(z_u * z_v)
          self.copule_static()#création copule static




    def rho_t(self, u, v):
      if self.is_dynamic_patton:#modele dynamic de patton donc coefficient varie
        if self.omega is None:
          #l'optimisation de patton n'a pas convergé on utilise parametre static
          return self.rho_t_moins_1
        self.u.append(u); self.v.append(v)

        # première obs
        if len(self.u) == 1:
            return self.rho_t_moins_1

        CLIP = 1e-6
        u_raw = np.clip(self.u[-self.p:-1], CLIP, 1-CLIP)
        v_raw = np.clip(self.v[-self.p:-1], CLIP, 1-CLIP)

        u_ppf = norm.ppf(u_raw)
        v_ppf = norm.ppf(v_raw)
        ma = np.mean(u_ppf * v_ppf)

        rho = CoefCorrDynamique.logistic_modified(
            self.omega + self.beta * self.rho_t_moins_1 + self.alpha * ma
        )
        rho = np.clip(rho, -1+1e-6, 1-1e-6)
        self.rho_t_moins_1 = rho
        return rho

      else: #copule static
        return self.rho_t_moins_1 #rho calculer entrainement coefficient de corelation de la veille


    def copule_t(self, u, v):
      #ce code va servir a paramétriser la copule pour le temps t et ainsi la distribution lorsque on est dans le cas dynamique
      #u, v les pseudo observations out of sample

      #rho de t estimer model patton
      rho = self.rho_t(u, v)
      self.rho_list.append(rho)
      R = ot.CorrelationMatrix(2)
      R[0, 1] = rho
      R[1, 0] = rho  # inutile mais plus explicite

      # 2. Construire la copule normale avec cette matrice
      self.copule = ot.NormalCopula(R)
      self.distribution = ot.JointDistribution(self.marginals, self.copule)


    def copule_static(self):
      self.rho_list.append(self.rho_t_moins_1)
      R = ot.CorrelationMatrix(2)
      R[0, 1] = self.rho_t_moins_1
      R[1, 0] = self.rho_t_moins_1

      # 2. Construire la copule normale avec cette matrice
      self.copule = ot.NormalCopula(R)
      self.distribution = ot.JointDistribution(self.marginals, self.copule)
    """
    def copule_static(self, data):
      sample = ot.Sample(data.to_numpy().tolist())
      self.copule = copula = ot.NormalCopulaFactory().build(sample)
      self.distribution = ot.JointDistribution(self.marginals, self.copule)
      #self.rho_t_moins_1 = float(self.copule.getParameter()[0])
    """

    def get_likelihood(self):
        """
        Return the log-likelihood of `data` under the current copula model.
        If `data` is None, use the training sample.
        """

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #valider bon ddl et scale !
        log_likelihood = lambda x, v : np.sum(t.logpdf(x, df=v, loc=0, scale=np.sqrt((v-2)/v)))

        def garch_marginal(marge):
          #vérifie si les garch on bien convergé.
          #sinon calcul la loglikelihood d'une t-student static
          for asset, lh in marge.items():
            if lh is None:
              marge[asset] = log_likelihood(self.training_data[asset], self.d_of_f[asset])
          return marge


        if self.is_dynamic_patton and self.likelihood_patton is not None:
          pdf_vals = [np.exp(self.likelihood_patton)]
          loglikelihood_marginals = self.garch.get_loglikelihood()
          loglikelihood_marginals = garch_marginal(loglikelihood_marginals)

        else:

          #PIT
          x, y = self.training_data.iloc[:, 0], self.training_data.iloc[:, 1]
          u, v = self.PIT(x, y)
          uv_pairs = np.column_stack((u, v))
          copule = self.distribution.getCopula()

          copula_pdf_vals = np.array([copule.computePDF(list(pair)) for pair in uv_pairs])
          # 6. Éviter log(0) (protection numérique)
          pdf_vals = np.clip(copula_pdf_vals, 1e-300, None)


          if self.is_stationnaire or (self.is_dynamic_patton and self.likelihood_patton is None):#marginal si garch
            loglikelihood_marginals = self.garch.get_loglikelihood()
            loglikelihood_marginals = garch_marginal(loglikelihood_marginals)

          else: #marge static
            loglikelihood_marginals = {col : log_likelihood(self.training_data[col], self.d_of_f[col]) for col in self.training_data.columns}

        return np.log(pdf_vals).sum() + sum(loglikelihood_marginals.values())



    '''
    def probabilité_conditionnelle(self, data):
        """
        Approximation de P(U <= u | V = v)
        via différence finie sur v = G(y).
        """
        epsilon = 0.01
        valeurs = data.values.tolist()

        # PIT : u = F(x), v = G(y)
        u = [self.distribution.getMarginal(i).computeCDF(valeurs[i]) for i in range(len(valeurs))]

        v = u[1]  # v = G(y)

        u_eps_plus = u.copy()
        u_eps_minus = u.copy()

        # ✅ Corrige ici : bouger directement v
        u_eps_plus[1] = min(v + epsilon, 1.0)
        u_eps_minus[1] = max(v - epsilon, 0.0)

        cdf_plus = self.copule.computeCDF(u_eps_plus)
        cdf_minus = self.copule.computeCDF(u_eps_minus)

        try:
            result = (cdf_plus - cdf_minus) / (2 * epsilon)
        except ZeroDivisionError:
            result = 0.0

        return result
    '''

    def probabilité_conditionnelle(self, data, epsilon=0.01):
      """
      Calcule simultanément
          • P(U ≤ u | V = v)
          • P(V ≤ v | U = u)
      par dérivée numérique centrale sur la copule C(u, v).

      Paramètres
      ----------
      data : pd.Series
          Série à deux composantes (valeurs x et y observées).
      epsilon : float
          Pas de dérivation (différence finie centrale).

      Retour
      ------
      tuple(float, float)
          (prob_U_cond_V, prob_V_cond_U)
      """

      # 1) transformation PIT  (u = F_X(x),  v = F_Y(y))
      vals = data.values.tolist()
      uv    = [self.distribution.getMarginal(i).computeCDF(vals[i])
              for i in range(2)]
      u, v  = uv                         # dépaquetage

      # ------------------------------------------------------------------
      # 2) P(U ≤ u | V = v)  =  ∂C(u,v)/∂v
      # ------------------------------------------------------------------
      uv_plus, uv_minus = uv.copy(), uv.copy()
      uv_plus[1]  = min(v + epsilon, 1.0)
      uv_minus[1] = max(v - epsilon, 0.0)

      cdf_plus_v  = self.copule.computeCDF(uv_plus)
      cdf_minus_v = self.copule.computeCDF(uv_minus)
      prob_u_cond_v = (cdf_plus_v - cdf_minus_v) / (2 * epsilon)

      # ------------------------------------------------------------------
      # 3) P(V ≤ v | U = u)  =  ∂C(u,v)/∂u
      # ------------------------------------------------------------------
      uv_plus, uv_minus = uv.copy(), uv.copy()
      uv_plus[0]  = min(u + epsilon, 1.0)
      uv_minus[0] = max(u - epsilon, 0.0)

      cdf_plus_u  = self.copule.computeCDF(uv_plus)
      cdf_minus_u = self.copule.computeCDF(uv_minus)
      prob_v_cond_u = (cdf_plus_u - cdf_minus_u) / (2 * epsilon)

      return prob_u_cond_v, prob_v_cond_u


    def MI_t(self, data):
        if self.is_stationnaire:
          data = self.garch._nouveau_residu(data)
          x, y = data.iloc[0], data.iloc[1]

          if self.is_dynamic_patton:
            #je dois appliqué la probability integral transforme
            #pour obtenir u = F(X), v = F(Y)
            u, v = self.marginals[0].computeCDF(x), self.marginals[1].computeCDF(y)
            self.copule_t(u, v)

        return self.probabilité_conditionnelle(data)




import backtrader as bt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



class PairTradingStrategy(bt.Strategy):
    params = dict(
        training_day=1,
        trading_day=1,
        timeframe=5,             # minutes
        trade_condition=0.99,    # seuil sur le MI
        marge_stationnaire=True,
        dynamic=False,
        stocks=[]                # noms des datafeeds
    )

    def __init__(self):
        # datafeeds indexés par nom
        self.stocks = {name: self.getdatabyname(name) for name in self.params.stocks}
        self.stock_a, self.stock_b = None, None       # mis à jour par PairSelector
        self.copule = None

        self.pair_history = []     # historique des paires sélectionnées
        self.trade_results = []    # PnL de chaque trade
        self.trade_durations = []  # durée de chaque position (en heures)

        self.trade_log = []
        self.mispricing = []
        self.active_stock = False
        self.trade_direction = None

        self.jour_precedent = None
        self.nb_jours_depuis_training = 0
        self.start_strat = False
        self.close_mkt = False

        self.equity_curve = []
        self.dates = []

        self.nb_chandelles = int((6.5 * 60) / self.params.timeframe)
        self.tradesize = self.broker.getvalue()

        self.tp = self.sl = 0
        self.pnl_trade = []
        self.pnl_start = 0

    # ------------------------------------------------------------------ #
    #  Utilitaires
    # ------------------------------------------------------------------ #
    def market_close(self):
        t = self.datas[0].datetime.time(0)
        self.close_mkt = (t.hour == 15 and t.minute == 55)

    def count_days_since_training(self):
        d = self.datas[0].datetime.date(0)
        if self.jour_precedent is None:
            self.jour_precedent = d
        elif d != self.jour_precedent:
            self.nb_jours_depuis_training += 1
            self.jour_precedent = d

    def convert_to_return(self, kind='df', subset=None):
        if subset is None:
            subset = self.stocks.keys()
        if kind == 'df':
            df = pd.DataFrame({
                name: [self.stocks[name].close[-i]
                       for i in range(self.params.training_day * self.nb_chandelles, 0, -1)]
                for name in subset
            })
            df = df.pct_change().dropna() * 100
            return df
        elif kind == 'series':
            return pd.Series({
                name: ((self.stocks[name].close[0] / self.stocks[name].close[-1]) - 1) * 100
                for name in subset
            })
        else:
            raise ValueError("kind must be either 'df' or 'series'")

    # ------------------------------------------------------------------ #
    #  Entraînement et sélection de la meilleure paire
    # ------------------------------------------------------------------ #
    def select_pair_and_train(self):
        train_df = self.convert_to_return('df')
        selector = PairSelector(train_df,
                                self.params.marge_stationnaire,
                                self.params.dynamic)
        self.stock_a, self.stock_b = selector.get_bestpair()
        self.copule = selector.get_copule()
        self.pair_history.append((self.stock_a, self.stock_b))

    def fit_copule(self):
        self.market_close()
        self.count_days_since_training()

        if not self.start_strat:
            if self.nb_jours_depuis_training >= self.params.training_day and self.close_mkt:
                self.select_pair_and_train()
                self.start_strat = True
                self.nb_jours_depuis_training = 0
            else:
                return True  # pas encore prêt à trader

        elif self.nb_jours_depuis_training >= self.params.trading_day and self.close_mkt:
            self.select_pair_and_train()
            self.nb_jours_depuis_training = 0

    # ------------------------------------------------------------------ #
    #  Gestion des positions
    # ------------------------------------------------------------------ #
    def spread_return(self):
        da = self.stocks[self.stock_a]
        db = self.stocks[self.stock_b]
        ret_a = (da.close[0] - da.close[-1]) / da.close[-1]
        ret_b = (db.close[0] - db.close[-1]) / db.close[-1]
        return ret_a - ret_b

    def open_trade(self, longue, short):
        price_long = longue.close[0]
        price_short = short.close[0]
        qty_long = int(self.tradesize / price_long)
        qty_short = int(self.tradesize / price_short)

        self.buy(data=longue, size=qty_long)
        self.sell(data=short, size=qty_short)
        self.active_stock = True

        self.entry_datetime = self.datas[0].datetime.datetime(0)
        self.entry_prices = dict(
            long_asset=longue._name, long_price=price_long,
            short_asset=short._name, short_price=price_short,
            qty_long=qty_long, qty_short=qty_short
        )

        r = abs(self.spread_return())
        self.tp = self.sl = r
        self.pnl_trade = []
        self.pnl_start = self.broker.getvalue()

    def close_trade(self):
        exit_datetime = self.datas[0].datetime.datetime(0)

        duration = (exit_datetime - self.entry_datetime).total_seconds() / 3600
        self.trade_durations.append(duration)

        pnl = self.broker.getvalue() - self.pnl_start
        self.trade_results.append(pnl)

        self.trade_log.append({
            'entry_time': self.entry_datetime,
            'exit_time': exit_datetime,
            'long_asset': self.entry_prices['long_asset'],
            'short_asset': self.entry_prices['short_asset'],
            'entry_long_price': self.entry_prices['long_price'],
            'entry_short_price': self.entry_prices['short_price'],
            'qty_long': self.entry_prices['qty_long'],
            'qty_short': self.entry_prices['qty_short'],
            'pnl': pnl
        })

        self.close(data=self.stock_a)
        self.close(data=self.stock_b)
        self.active_stock = False
        self.tp = self.sl = 0
        self.pnl_trade = []
        self.trade_direction = None

    # ------------------------------------------------------------------ #
    #  Boucle principale de Backtrader
    # ------------------------------------------------------------------ #
    def next(self):
        if self.fit_copule():
            return

        data = self.convert_to_return('series', [self.stock_a, self.stock_b])
        """
        mispricing = self.copule.MI_t(data)
        self.mispricing.append(mispricing)

        if not self.active_stock:
            if mispricing >= self.params.trade_condition:
                self.open_trade(self.stocks[self.stock_b], self.stocks[self.stock_a])
                self.trade_direction = "short"
            elif mispricing <= 1 - self.params.trade_condition:
                self.open_trade(self.stocks[self.stock_a], self.stocks[self.stock_b])
                self.trade_direction = "long"
        """
        p_u_cond_v , p_v_cond_u = self.copule.MI_t(data)
        if not self.active_stock:
            if p_u_cond_v >= self.params.trade_condition and  p_v_cond_u <= 1 - self.params.trade_condition:
                self.open_trade(self.stocks[self.stock_b], self.stocks[self.stock_a])
                self.trade_direction = "short"
            elif p_u_cond_v <= 1 - self.params.trade_condition and p_v_cond_u >= 1 - self.params.trade_condition:
                self.open_trade(self.stocks[self.stock_a], self.stocks[self.stock_b])
                self.trade_direction = "long"

        else:
            if self.close_mkt:
                self.close_trade()
                return

            self.pnl_trade.append(self.spread_return())
            pnl = sum(self.pnl_trade)

            if self.trade_direction == "long":
                if pnl <= -self.sl or pnl >= self.tp:
                    self.close_trade()

            else:
                if pnl >= self.sl or pnl <= -self.tp:
                    self.close_trade()


        self.equity_curve.append(self.broker.getvalue())
        self.dates.append(self.datas[0].datetime.date(0))

    # ------------------------------------------------------------------ #
    #  Analyse des performances
    # ------------------------------------------------------------------ #
    def stop(self):
        equity = pd.Series(self.equity_curve, index=self.dates)
        returns = equity.pct_change().dropna()

        # Tracer la courbe de rendement cumulé
        cumulative_returns = (1 + returns).cumprod()
        plt.figure(figsize=(12, 5))
        plt.plot(cumulative_returns, label="Rendement cumulé")
        plt.xlabel("Période")
        plt.ylabel("Croissance du portefeuille")
        plt.title("Évolution du portefeuille au cours du temps")
        plt.grid(True)
        plt.legend()


        annual_factor = np.sqrt(252 * (6.5 * 60 / self.params.timeframe))
        volatility = returns.std() * annual_factor
        sharpe = returns.mean() / returns.std() * annual_factor
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        print(f"Volatilité annualisée : {volatility:.4f}")
        print(f"Sharpe ratio : {sharpe:.4f}")
        print(f"Drawdown maximal : {max_drawdown:.4f}")

        if self.pair_history:
            print("\nFréquence des paires sélectionnées :")
            print(pd.Series(self.pair_history).value_counts())

        if self.trade_results:
            results = pd.Series(self.trade_results)
            avg_gain = results[results > 0].mean()
            avg_loss = results[results < 0].mean()
            avg_duration = np.mean(self.trade_durations)
            total_days = (self.dates[-1] - self.dates[0]).days
            trades_per_day = len(results) / total_days if total_days > 0 else np.nan

            print(f"\nNombre moyen de trades par jour : {trades_per_day:.2f}")
            print(f"Gain moyen : {avg_gain:.4f}")
            print(f"Perte moyenne : {avg_loss:.4f}")
            print(f"Durée moyenne des positions (h) : {avg_duration:.2f}")
        else:
            print("\nAucun trade effectué.")



"""
symbols = ['JPM', 'BAC', 'C', 'WFC', 'USB',
           'PNC', 'TFC', 'GS', 'MS', 'SCHW']
"""
"""
symbols = ['JPM', 'BAC', 'C', 'WFC',
           'GS', 'MS']
"""

#symbols = ['JPM', 'BAC']

symbols = ['MS', 'JPM']

# ------------------------------------------------------------------
# 2) Fonctions utilitaires (identiques, mais génériques)
# ------------------------------------------------------------------
def prepare_for_backtrader(df_wide, symbol):
    """
    Extrait le sous-DataFrame df_wide[symbol] (colonnes OHLCV)
    et ajoute openinterest = 0 pour Backtrader.
    """
    df = df_wide[symbol].copy()
    required = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required):
        missing = [c for c in required if c not in df.columns]
        raise ValueError(f"{symbol} – colonnes manquantes : {missing}")
    df['openinterest'] = 0
    df = df.dropna()
    return df[['open', 'high', 'low', 'close', 'volume', 'openinterest']]

def filter_last_days(df, n_days):
    cutoff = df.index.max() - timedelta(days=n_days)
    return df[df.index >= cutoff]

# ------------------------------------------------------------------
# 3) Construction du moteur Backtrader
# ------------------------------------------------------------------
cerebro = bt.Cerebro()
cerebro.cheat_on_close = True

# a) ajouter chaque flux de données
for sym in symbols:
    data = prepare_for_backtrader(df_wide, sym)
    # ↓ optionnel – filtrer les 250 derniers jours, par ex.
    # data = filter_last_days(data, 250)
    feed = bt.feeds.PandasData(dataname=data, name=sym)
    cerebro.adddata(feed, name=sym)

# b) paramètre pour la stratégie
#    ––> ici, on lui passe toute la liste des noms ajoutés
cerebro.addstrategy(
    PairTradingStrategy,
    stocks=symbols            # ou list(combinations(symbols, 2)) si ta stratégie attend des paires
)

# ------------------------------------------------------------------
# 4) Paramètres broker et lancement
# ------------------------------------------------------------------
cerebro.broker.setcash(100_000)
cerebro.broker.setcommission(leverage=10.0)

results = cerebro.run()

