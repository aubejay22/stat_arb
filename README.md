# stat_arb

This project implements a statistical arbitrage strategy using copulas.

Trading decisions now rely solely on conditional probabilities computed by the
copula. The orientation of the input symbols (e.g. `['JPM', 'MS']` vs
`['MS', 'JPM']`) no longer impacts which positions are taken.
