from typing import Tuple


def kelly_fraction(
    predicted_probability: float, current_market_probability: float, alpha: float
) -> float:
    """
    Return the Kelly fraction of bankroll to wager.
    If > 0, buy YES; if < 0, buy NO; if = 0, no bet.

    alpha is an optional 'fractional Kelly' scaling (0 < alpha < 1).
    """
    p = predicted_probability
    c = current_market_probability

    # If you think the true probability is higher than the market's price, buy YES.
    if p > c:
        # Kelly fraction for a yes-bet in a fixed-odds market
        f = (p - c) / (1 - c)
        return alpha * f

    # If you think the true probability is lower than the market's price, buy NO.
    elif p < c:
        # Kelly fraction for a no-bet is negative => we return a negative fraction
        f = (c - p) / c
        return -alpha * f

    # If p == c, no edge => no bet
    else:
        return 0.0


def kelly_bet(
    predicted_probability: float,
    current_market_probability: float,
    alpha: float,
    bankroll: float,
    max_bet_amount: int,
) -> Tuple[int, str]:
    """
    Return the amount to bet and the side to bet on (YES or NO).
    """
    f = kelly_fraction(predicted_probability, current_market_probability, alpha)
    bet = bankroll * f
    if bet > 0:
        return min(bet, max_bet_amount), "YES"
    elif bet < 0:
        return min(-bet, max_bet_amount), "NO"
    else:
        return 0, "NO"


def test():
    predicted_probability = 0.6
    current_market_probability = 0.4
    alpha = 1.0
    kelly = kelly_fraction(predicted_probability, current_market_probability, alpha)
    print(f"Kelly: {kelly}")
    bankroll = 1000
    max_bet_amount = 100
    bet_amount, side = kelly_bet(
        predicted_probability,
        current_market_probability,
        alpha,
        bankroll,
        max_bet_amount,
    )
    print(f"Bet amount: {bet_amount}, side: {side}")


if __name__ == "__main__":
    test()
