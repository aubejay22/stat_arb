from garch import Garch
from copule_patton import CoefCorrDynamique
import openturns as ot
import numpy as np
import pandas as pd
from scipy.stats import norm, t
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from abc import ABC, abstractmethod



class CopuleBase(ABC):

    def __init__(self):
        self.distribution = None
        self.copule = None
        self.marginals = None
        self.d_of_f = {} #liste degrés de libertés des lois de student
        self.rho_t_moins_1 = 0

    
    @abstractmethod
    def fit(self, df):
        """
        Entraîne la copule sur les données. Doit être implémentée par les sous-classes.
        :param df: DataFrame contenant les données à utiliser pour l’entraînement.
        """
        pass

    @abstractmethod
    def _train_copule(self, data):
        """
        Code paramétrise la copule et marge a partir données pris en argument data.
        """
        pass

    @abstractmethod
    def estimation_marginal(self, asset_name, asset_return, v):
        """
        Estime une loi de Student t pour les marges.
        Doit être implémentée par les sous-classes.
        """
        pass
    
    
    @abstractmethod
    def MI_t(self, data):
        """
        Calcule le mispricing index au temps t.
        :param data: pd.DataFrame contenant les échantillons (une colonne par actif).
        :return: probabilité conditionnelle P(U ≤ u | V = v) et P(V ≤ v | U = u)
        """
        pass
    

    def PIT(self, x, y):
        """
        Effectue la transformation PIT (Probability Integral Transform).
        :param x: Série de rendements de l'actif 1.
        :param y: Série de rendements de l'actif 2.
        :return: u, v  les valeurs transformées uniformes.
        """
        # S'assurer que x et y sont itérables
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)

        u = np.array([self.marginals[0].computeCDF(xi) for xi in x])
        v = np.array([self.marginals[1].computeCDF(yi) for yi in y])

        CLIP = 1e-6
        u_raw = np.clip(u, CLIP, 1-CLIP)
        v_raw = np.clip(v, CLIP, 1-CLIP)
        return u_raw, v_raw
    

    def copule_static(self):
        R = ot.CorrelationMatrix(2)
        R[0, 1] = self.rho_t_moins_1
        R[1, 0] = self.rho_t_moins_1
        self.copule = ot.NormalCopula(R)
        self.distribution = ot.JointDistribution(self.marginals, self.copule)
    
   
    def probabilité_conditionnelle(self, data, epsilon=0.01):
        """
        Calcule simultanément :
            • P(U ≤ u | V = v)
            • P(V ≤ v | U = u)
        par dérivée numérique centrale sur la copule C(u, v).
        """

        # 1. Extraire les scalaires (float natifs)
        x, y = float(data.iloc[0]), float(data.iloc[1])
        u, v = self.PIT(x, y)
        u = float(np.asarray(u).item())
        v = float(np.asarray(v).item())
        uv = [u, v]

        # 2. ∂C/∂v
        uv_plus  = uv.copy()
        uv_minus = uv.copy()
        uv_plus[1]  = min(v + epsilon, 1.0)
        uv_minus[1] = max(v - epsilon, 0.0)

        cdf_plus_v  = self.copule.computeCDF(uv_plus)
        cdf_minus_v = self.copule.computeCDF(uv_minus)
        prob_u_cond_v = (cdf_plus_v - cdf_minus_v) / (2 * epsilon)

        # 3. ∂C/∂u
        uv_plus  = uv.copy()
        uv_minus = uv.copy()
        uv_plus[0]  = min(u + epsilon, 1.0)
        uv_minus[0] = max(u - epsilon, 0.0)

        cdf_plus_u  = self.copule.computeCDF(uv_plus)
        cdf_minus_u = self.copule.computeCDF(uv_minus)
        prob_v_cond_u = (cdf_plus_u - cdf_minus_u) / (2 * epsilon)

        return prob_u_cond_v, prob_v_cond_u



class CopuleStatic(CopuleBase):
    def __init__(self):
        super().__init__()


    def estimation_marginal(self, asset_name, asset_return, v):
        v , _, _ = t.fit(asset_return, floc=0.0, fscale=1.0)
        self.d_of_f[asset_name] = v
        return ot.Student(v, 0, 1)
    

    def fit(self, df):
        self._train_copule(df)

    
    def _train_copule(self, data):
        data = data.dropna()

        if data.shape[0] == 0:
            raise ValueError("Plus de données après suppression des NaN. Impossible d’entraîner la copule.")

        x, y = data.iloc[:,0].to_numpy(), data.iloc[:,1].to_numpy()

        self.marginals = [self.estimation_marginal(asset_name, data[asset_name], None) for asset_name in data.columns]
        u, v = self.PIT(x, y)
        z_u, z_v = norm.ppf(u), norm.ppf(v)
        self.rho_t_moins_1 = np.corrcoef(z_u, z_v)[0, 1]
        self.copule_static()#création copule static
   

  
    def MI_t(self, data):
        return self.probabilité_conditionnelle(data)
    





class CopuleGarch(CopuleBase):
    def __init__(self):
        super().__init__()
        self.garch = Garch()

    def estimation_marginal(self, asset_name, asset_return, v):
        if v is None:
            v , _, _ = t.fit(asset_return, floc=0.0, fscale=1.0)
            self.d_of_f[asset_name] = v
        return ot.Student(v, 0, 1)
    

    def fit(self, df):
        self.garch.reset_attribut()
        self.garch._train_garch(df)
        self.d_of_f = self.garch.get_doff()
        self._train_copule(self.garch.get_standardized_residuals())

    
    def _train_copule(self, data):
        data = data.dropna()

        if data.shape[0] == 0:
            raise ValueError("Plus de données après suppression des NaN. Impossible d’entraîner la copule.")

        x, y = data.iloc[:,0].to_numpy(), data.iloc[:,1].to_numpy()

        self.marginals = [self.estimation_marginal(asset_name, data[asset_name], self.d_of_f[asset_name]) for asset_name in data.columns]
        u, v = self.PIT(x, y)
        z_u, z_v = norm.ppf(u), norm.ppf(v)
        self.rho_t_moins_1 = np.corrcoef(z_u, z_v)[0, 1]
        self.copule_static()#création copule static
   
    
    def MI_t(self, data):
        data = self.garch._nouveau_residu(data)
        return self.probabilité_conditionnelle(data)






class CopuleGarchSemiParametric(CopuleBase):
    def __init__(self):
        super().__init__()
        self.garch = Garch()

    def estimation_marginal(self, asset_name, asset_return, v):
        return ot.KernelSmoothing().build(asset_return)
        
    

    def fit(self, df):
        self.garch.reset_attribut()
        self.garch._train_garch(df)
        self.d_of_f = self.garch.get_doff()
        self._train_copule(self.garch.get_standardized_residuals())

    
    def _train_copule(self, data):
        data = data.dropna()
        if data.shape[0] == 0:
            raise ValueError("Plus de données après suppression des NaN. Impossible d’entraîner la copule.")

        sample = ot.Sample(data.values.tolist())
        self.marginals = [
                          self.estimation_marginal(asset_name, sample.getMarginal(i), self.d_of_f[asset_name])
                          for i, asset_name in enumerate(data.columns)
                         ]
        
        self.copule = ot.NormalCopulaFactory().build(sample)
        self.distribution = ot.JointDistribution(self.marginals, self.copule)
    

    def MI_t(self, data):
        data = self.garch._nouveau_residu(data)
        return self.probabilité_conditionnelle(data)






class CopulePatton(CopuleBase):
    def __init__(self):
        super().__init__()
        self.garch = Garch()
        self.p = 10 #paramètre pour le modèle de Patton
        self.omega, self.beta, self.alpha = 0, 0, 0
        self.u, self.v = [], []

    def estimation_marginal(self, asset_name, asset_return, v):
        if v is None:
            v , _, _ = t.fit(asset_return, floc=0.0, fscale=1.0)
            self.d_of_f[asset_name] = v
        return ot.Student(v, 0, 1)
    

    def fit(self, df):
        self.garch.reset_attribut()
        self.garch._train_garch(df)
        self.d_of_f = self.garch.get_doff()
        self._train_copule(self.garch.get_standardized_residuals())

    
    def _train_copule(self, data):
        data = data.dropna()

        if data.shape[0] == 0:
            raise ValueError("Plus de données après suppression des NaN. Impossible d’entraîner la copule.")

        x, y = data.iloc[:,0].to_numpy(), data.iloc[:,1].to_numpy()

        self.marginals = [self.estimation_marginal(asset_name, data[asset_name], self.d_of_f[asset_name]) for asset_name in data.columns]
        #estimation parametre dynamique copule
        u, v = self.PIT(x, y)
        self.rho_t_moins_1 = np.corrcoef(u, v)[0, 1]
        copule_dyn =  CoefCorrDynamique(u, v, self.p)
        self.omega, self.beta, self.alpha = copule_dyn.get_params()
   

    def rho_t(self, u, v):
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


    def copule_t(self, u, v):
        #ce code va servir a paramétriser la copule pour le temps t et ainsi la distribution lorsque on est dans le cas dynamique
        #u, v les pseudo observations out of sample

        #rho de t estimer model patton
        rho = self.rho_t(u, v)
        R = ot.CorrelationMatrix(2)
        R[0, 1] = rho
        R[1, 0] = rho  

        # 2. Construire la copule normale avec cette matrice
        self.copule = ot.NormalCopula(R)
        self.distribution = ot.JointDistribution(self.marginals, self.copule)

    
    def MI_t(self, data):
        data = self.garch._nouveau_residu(data)
        x, y = data.iloc[0], data.iloc[1]
        u, v = self.PIT(x, y)
        self.copule_t(u, v)

        return self.probabilité_conditionnelle(data)
  