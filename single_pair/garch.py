from arch import arch_model
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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