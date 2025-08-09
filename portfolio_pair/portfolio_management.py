import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
import cvxpy as cp


class DynamicPairPortfolio:
    def __init__(self, df_equity, window=22, top_n=3, sharpe_min=0, rebalance_freq=1, MeanVarianceOpt=True):
        """
        Paramètres :
        - df_equity : DataFrame avec les equity curves (index = date, colonnes = paires)
        - window : nombre de jours pour la fenêtre de Sharpe ratio
        - top_n : nombre de paires à sélectionner
        - sharpe_min : Sharpe minimum requis
        - rebalance_freq : fréquence de rebalancement en jours
        """
        self.window = window
        self.top_n = top_n
        self.sharpe_min = sharpe_min
        self.rebalance_freq = rebalance_freq
        self.MeanVarianceOpt = MeanVarianceOpt

        self.df_equity = df_equity.resample('1D').last().pct_change()

        # Pré-calcul des ratios de Sharpe glissants
        rolling_mean = self.df_equity.rolling(window=window).mean()
        rolling_std = self.df_equity.rolling(window=window).std()
        self.df_sharpe = (rolling_mean / rolling_std).shift(1)


        # Résultats à remplir après le backtest
        self.daily_returns = None
        self.cumulative_returns = None
        self.weights_history = []


        #éxécution du code 
        self.run_backtest()
        self.plot_cumulative_returns()
        self.print_performance()

    def _select_top_pairs(self, date):
        """
        Sélectionne les meilleures paires selon le Sharpe ratio pré-calculé à la date donnée.
        """
        if date not in self.df_sharpe.index:
            print("WTF!!")
            return [], pd.Series()

        sharpe_today = self.df_sharpe.loc[date]
        sharpe_today = sharpe_today[sharpe_today > self.sharpe_min]
        top_pairs = sharpe_today.sort_values(ascending=False).head(self.top_n)

        if len(sharpe_today) == 0:
            return [], pd.Series()
        
        if not self.MeanVarianceOpt:
            weights = pd.Series(1 / len(top_pairs), index=top_pairs.index)
            return top_pairs.index, weights
        
        else:
            """
            pos = self.df_sharpe.index.get_loc(date)
            sharpe_period = self.df_sharpe.iloc[pos - self.window:pos, ]
            
            #restreint au stock choisit
            sharpe_period = sharpe_period[top_pairs.index]
        
            #cov matrix
            S = sharpe_period.cov()
            ef = EfficientFrontier(sharpe_today[top_pairs.index], S, weight_bounds=(0, 1))
            """

            pos = self.df_equity.index.get_loc(date)
            return_period = self.df_equity.iloc[pos - self.window:pos, ]
            
            #restreint au stock choisit
            return_period = return_period[top_pairs.index]
        
            #cov matrix
            S = return_period.cov()
            mu = return_period.mean()
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            
            # Calcul des poids optimaux
            weights_dict = ef.max_sharpe()  # ef.min_volatility() 
            cleaned_weights = ef.clean_weights()  # arrondit et remplace les poids ≈ 0 ou ≈ négatifs
            weights = pd.Series(cleaned_weights)

            #weights = pd.Series(weights_dict)
            return weights.index, weights

        

    def run_backtest(self):
        """
        Lance le backtest dynamique en sélectionnant les meilleures paires
        à chaque date de rebalancement et en calculant les rendements du portefeuille.
        """
        dates = self.df_sharpe.index[self.window:]
        returns_list = []
        current_weights = pd.Series(dtype=float)

        for i, date in enumerate(dates):
            if i % self.rebalance_freq == 0:
                selected_pairs, current_weights = self._select_top_pairs(date)
                self.weights_history.append((date, current_weights))

            if len(current_weights) == 0:
                returns_list.append(0)
                continue

            rets_today = self.df_equity.loc[date, current_weights.index]
            portfolio_return = rets_today.dot(current_weights)
            returns_list.append(portfolio_return)

        self.daily_returns = pd.Series(returns_list, index=dates)
        self.cumulative_returns = (1 + self.daily_returns).cumprod() 
        return self.cumulative_returns

    def plot_cumulative_returns(self):
        """
        Trace le rendement cumulé du portefeuille.
        """
        if self.cumulative_returns is None:
            raise ValueError("Backtest not run yet.")
        plt.figure(figsize=(10, 5))
        self.cumulative_returns.plot()
        plt.title("Rendement cumulé du portefeuille dynamique")
        plt.ylabel("Rendement cumulé")
        plt.xlabel("Date")
        plt.grid(True)
        plt.show()

    def print_performance(self):
        """
        Affiche les statistiques de performance du portefeuille.
        """
        if self.daily_returns is None:
            raise ValueError("Backtest not run yet.")
        total_return = self.cumulative_returns.iloc[-1] -1
        volatility = self.daily_returns.std() * np.sqrt(252)
        sharpe = self.daily_returns.mean() / self.daily_returns.std() * np.sqrt(252)
        max_drawdown = (self.cumulative_returns.cummax() - self.cumulative_returns).max()

        print("\n📊 Statistiques de performance du portefeuille stat-arb :")
        print(f"Total Return       : {total_return * 100:.2f} %")
        print(f"Annual Volatility  : {volatility * 100:.2f} %")
        print(f"Sharpe Ratio       : {sharpe:.2f}")
        print(f"Max Drawdown       : {max_drawdown * 100:.2f} %")

    def get_weights_history(self):
        """
        Retourne la liste des pondérations appliquées à chaque date de rebalancement.
        """
        return self.weights_history
