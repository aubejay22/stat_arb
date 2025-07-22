from backtest import PairTradingStrategy
from dependence_modeling import *
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

    


class PairTradingStrategy(bt.Strategy):
    params = dict(
        training_day=1,
        trading_day=1,
        timeframe=5,             # minutes
        trade_condition=0.99,    # seuil sur le MI
        statistical_model=CopuleGarch,
        profit_target=0.002,     # 0.2% de gain, si opportunité rapport pas plus que ca on ne la prend pas
        edge_buffer = 6
    )
    
    def __init__(self):
        #noms des deux stocks
        self.stock_a, self.stock_b = self.datas[0]._name, self.datas[1]._name
        
        #noms des stocks avec datafeeds
        self.stocks = {
                        self.stock_a: self.datas[0],
                        self.stock_b: self.datas[1]
                      }
        
        #Création de la copule
        self.copule = self.params.statistical_model()
        
        self.backtest_result = None #pnl fin backtest a extraire pour analyser rendement strat
        self.metrics = {}  #métrics du backtest nb trade, gain moyenne, etc

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
        #derniere bougie a 15h55 et le close c'Est à 16h
        dt = self.datas[0].datetime.datetime(0)
        self.close_mkt = (dt.hour == 15 and dt.minute == 55)


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
            n = self.params.training_day * self.nb_chandelles
            closes = np.column_stack([self.stocks[name].close.get(size=n) for name in subset])
            opens  = np.column_stack([self.stocks[name].open.get(size=n)  for name in subset])
            
            #indice a éliminer a cause du buffer
            idx = (np
               .arange(n)              
               .reshape(self.params.training_day, self.nb_chandelles)     
               [:, self.params.edge_buffer : self.nb_chandelles-self.params.edge_buffer] 
               .ravel())                          
            
            # -------- sous-échantillonnage -----------
            closes_use = closes[idx]
            opens_use  = opens [idx]

            returns = (closes_use - opens_use) / opens_use * 100
            return pd.DataFrame(returns, columns=subset)


        elif kind == 'series':
            return pd.Series({
                name: (((self.stocks[name].close[0] - self.stocks[name].open[0]) / self.stocks[name].open[0]))  * 100
                for name in subset
            })
        else:
            raise ValueError("kind must be either 'df' or 'series'")

    def is_edge_bar(self):
        """
        Retourne True si la bougie courante (self.datas[0]) est
        dans les `buffer` premières ou dernières bougies de la journée.
        """
        dt = self.datas[0].datetime.datetime(0)
        timeframe = self.params.timeframe
        buffer = self.params.edge_buffer

        # ouverture et fermeture de la séance
        session_open = 9 * 60 + 30  # 9h30 = 570
        session_close = 16 * 60     # 16h00 = 960
        session_len = session_close - session_open  # 390 min

        nb_bars = session_len // timeframe

        now_min = dt.hour * 60 + dt.minute
        idx = (now_min - session_open) // timeframe

        return idx < buffer or idx >= nb_bars - buffer



    # ------------------------------------------------------------------ #
    #  Entraînement de la copule
    # ------------------------------------------------------------------ #
    def fit_copule(self):
        self.market_close()
        self.count_days_since_training()

        if not self.start_strat:
            if self.nb_jours_depuis_training >= self.params.training_day-1 and self.close_mkt:
                #!!!!!!!!!!!!!!!!!
                train_df = self.convert_to_return('df')
                self.copule.fit(train_df)
                self.start_strat = True
                self.nb_jours_depuis_training = 0
            else:
                return True  # pas encore prêt à trader

        elif self.nb_jours_depuis_training >= self.params.trading_day-1 and self.close_mkt:
            train_df = self.convert_to_return('df')
            self.copule.fit(train_df)
            self.nb_jours_depuis_training = 0



    # ------------------------------------------------------------------ #
    #  Gestion des positions
    # ------------------------------------------------------------------ #
    def spread_return(self):
        da = self.stocks[self.stock_a]
        db = self.stocks[self.stock_b]
        ret_a = (da.close[0] - da.open[0]) / da.open[0]
        ret_b = (db.close[0] - db.open[0]) / db.open[0]
        return ret_a - ret_b

    def open_trade(self, longue, short):
        r = abs(self.spread_return())
        if r < self.params.profit_target: 
            return
        
        price_long = longue.close[0]
        price_short = short.close[0]
        qty_long = int(self.tradesize / price_long)
        qty_short = int(self.tradesize / price_short)
        

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        #    wtf j'ai buy avant acheté ??impossible xd

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.buy(data=longue, size=qty_long)
        self.sell(data=short, size=qty_short)
        self.active_stock = True

        self.entry_datetime = self.datas[0].datetime.datetime(0)
        self.entry_prices = dict(
            long_asset=longue._name, long_price=price_long,
            short_asset=short._name, short_price=price_short,
            qty_long=qty_long, qty_short=qty_short
        )

    
        self.tp = self.sl = r
        self.pnl_trade = []
        self.pnl_start = self.broker.getvalue()


    def close_trade(self, end_of_the_day=None):
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

        if end_of_the_day is not None: #fin de la journée trading cloturé à 16h
          self.close(data=self.stock_a, exectype=bt.Order.Close)
          self.close(data=self.stock_b, exectype=bt.Order.Close)
        else:
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
        
        is_edge = self.is_edge_bar()
        if is_edge:
          #la dans ce code on attend juste qu'au close pour fermer position
          if self.active_stock and  self.close_mkt:
            self.close_trade('fin de journée! fermer toutes les positions')
            self.equity_curve.append(self.broker.getvalue())
            self.dates.append(self.datas[0].datetime.datetime(0))
          return

        data = self.convert_to_return('series', [self.stock_a, self.stock_b])
        p_u_cond_v , p_v_cond_u = self.copule.MI_t(data)
        if not self.active_stock:
          if not is_edge:  #anciennement close_mkt 
            if p_u_cond_v >= self.params.trade_condition and  p_v_cond_u <= 1 - self.params.trade_condition:
                self.open_trade(self.stocks[self.stock_b], self.stocks[self.stock_a])
                self.trade_direction = "short"
            elif p_v_cond_u >= self.params.trade_condition and p_u_cond_v <= 1 - self.params.trade_condition:
                self.open_trade(self.stocks[self.stock_a], self.stocks[self.stock_b])
                self.trade_direction = "long"

        else:#en position
            self.pnl_trade.append(self.spread_return())
            pnl = sum(self.pnl_trade)

            if self.trade_direction == "long":
                if pnl <= -self.sl or pnl >= self.tp:
                    self.close_trade()

            else:
                if pnl >= self.sl or pnl <= -self.tp:
                    self.close_trade()

        self.equity_curve.append(self.broker.getvalue())
        self.dates.append(self.datas[0].datetime.datetime(0))

    # ------------------------------------------------------------------ #
    #  Analyse des performances
    # ------------------------------------------------------------------ #
    def stop(self):
        
        # Tracer la courbe de rendement cumulé
        self.backtest_result = pd.Series(self.equity_curve, index=self.dates)
         
       
        if self.trade_results:
            results = pd.Series(self.trade_results)
            avg_gain = results[results > 0].mean()
            avg_loss = results[results < 0].mean()
            avg_duration = np.mean(self.trade_durations)
            total_days = (self.dates[-1] - self.dates[0]).days
            trades_per_day = len(results) / total_days if total_days > 0 else np.nan


            self.metrics['avg_gain'] = avg_gain
            self.metrics['avg_loss'] =  avg_loss
            self.metrics['avg_duration'] = avg_duration
            self.metrics['trades_per_day'] = trades_per_day
        else:
            print("\nAucun trade effectué.")








