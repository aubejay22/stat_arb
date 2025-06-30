
from itertools import combinations
from dependence_modeling import *



class PairSelector:
    """
    Cette classe sélectionne automatiquement la meilleure paire d'actifs
    en fonction critere
    """

    def __init__(self, df, copule):
        """
        Initialise l'objet PairSelector.

        Paramètres
        ----------
        df : pandas.DataFrame
            DataFrame contenant les rendements des actifs (chaque colonne représente un actif).
        copule : class
            Classe de la copule à utiliser (ex. CopuleStatic, CopuleGarch, CopulePatton).
            Elle doit pouvoir être instanciée sans argument : copule().
        """
        # Créer toutes les combinaisons possibles de paires d'actifs
        self.pairs = list(combinations(df.columns.tolist(), 2))

        # Dictionnaire contenant une instance de la copule pour chaque paire
        self.copule_dict = {pair: copule() for pair in self.pairs}

        # Déterminer la meilleure paire selon la log-vraisemblance
        self.best_pair = self.find_best_pair(df)

    def find_best_pair(self, df):
        """
        Évalue chaque paire en entraînant la copule sur ses données, 
        puis retourne la paire ayant la plus grande log-vraisemblance.

        Paramètres
        ----------
        df : pandas.DataFrame
            DataFrame contenant les rendements des actifs.

        Retour
        ------
        tuple
            La meilleure paire d'actifs, sous forme de tuple (actif_1, actif_2).
        """
        scores = {pair: None for pair in self.pairs}

        for pair in self.pairs:
            j, k = pair
            returns = df[[j, k]]

            # Entraîner la copule et calculer la vraisemblance
            self.copule_dict[pair].fit(returns)
            scores[pair] = self.copule_dict[pair].rho_t_moins_1 

        return max(scores, key=scores.get)

    def get_bestpair(self):
        """
        Retourne la meilleure paire sélectionnée.
        """
        return self.best_pair

    def get_copule(self):
        """
        Retourne l'objet copule associé à la meilleure paire.

        Cela permet de réutiliser directement l'objet copule dans un backtest
        sans avoir à le recalculer.

        Retour
        ------
        object
            Instance de la copule associée à la meilleure paire.
        """
        return self.copule_dict[self.best_pair]
