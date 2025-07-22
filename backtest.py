from portfolio import PairSelector
from dependence_modeling import *
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
        statistical_model=CopuleGarch,
        stocks=[],                # noms des datafeeds
        profit_target=0.002,     # 0.2% de gain, si opportunité rapport pas plus que ca on ne la prend pas
        edge_buffer = 6
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

        
        self.backtest_result = None
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
    #  Entraînement et sélection de la meilleure paire
    # ------------------------------------------------------------------ #
    def select_pair_and_train(self):
        train_df = self.convert_to_return('df')
        selector = PairSelector(train_df,
                                self.params.statistical_model)
        self.stock_a, self.stock_b = selector.get_bestpair()
        self.copule = selector.get_copule()
        self.pair_history.append((self.stock_a, self.stock_b))


    def fit_copule(self):
        self.market_close()
        self.count_days_since_training()

        if not self.start_strat:
            if self.nb_jours_depuis_training >= self.params.training_day-1 and self.close_mkt:
                self.select_pair_and_train()
                self.start_strat = True
                self.nb_jours_depuis_training = 0
            else:
                return True  # pas encore prêt à trader

        elif self.nb_jours_depuis_training >= self.params.trading_day-1 and self.close_mkt:
            self.select_pair_and_train()
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
            self.dates.append(self.datas[0].datetime.date(0))
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
        plt.show()

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

            results.plot.hist(bins=100, edgecolor='black')
            plt.title("Distribution du P&L")
            plt.xlabel("$")
            plt.ylabel("Fréquence")
            plt.grid(True)
            plt.show()
        else:
            print("\nAucun trade effectué.")


         