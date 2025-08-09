from single_pair.backtest2 import PairTradingStrategy
from single_pair.dependence_modeling import *
from itertools import combinations
import backtrader as bt
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm



class PortfolioBacktest:
   
    def __init__(self, stocks, df, fees, params, nom_fichier_result):
        self.df = df
        self.fees = fees
        self.params = params
        self.nom_fichier_result = nom_fichier_result

        # Créer toutes les combinaisons possibles de paires d'actifs
        self.pairs = list(combinations(stocks, 2))
        
        self.results_each_pair = {}
        
        

    def backtest(self, pair):
        cerebro = bt.Cerebro()
        for sym in pair:
            data = PortfolioBacktest.prepare_for_backtrader(self.df, sym)
            feed = bt.feeds.PandasData(dataname=data, name=sym)
            cerebro.adddata(feed, name=sym)

        cerebro.addstrategy(PairTradingStrategy, **self.params)
        cerebro.broker.setcash(100_000)
        cerebro.broker.setcommission(leverage=1.01, commission=self.fees)

        start = cerebro.run()

        #return serie de L'évolution equity curve
        return start[0].backtest_result


    def run(self):
        #run du backtest 
        for pair in tqdm(self.pairs, desc="Backtesting pairs"):
            result = self.backtest(pair)
            self.results_each_pair[pair] = result

        df = pd.DataFrame.from_dict(self.results_each_pair)
        df.to_pickle(self.nom_fichier_result + ".pkl")
        
        return df
   
    @staticmethod
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
        return df[['open', 'high', 'low', 'close', 'volume', 'openinterest']]