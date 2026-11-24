# Mirrors and Markets

**Reflexivity evidence of retail traders' influence in meme stocks**

M.Sc dissertation — NEOMA Business School, M.Sc Finance & Big Data.
Supervisor: Gabriel Giménez-Roche. Submitted 31 August 2024.

📄 **[Read the dissertation (PDF)](dissertation/Simons_NEOMA_MSc_dissertation_Mirrors-and-Markets.pdf)**

## Research question

Can George Soros's reflexivity theory and sentiment analysis explain the price
movements of meme stocks, and how do retail investors' behaviours amplify
these effects? GameStop (GME), 2020–2021, is the case study.

Hypotheses tested:

1. Reflexivity — feedback loops between perceptions and prices — is a
   significant driver of meme-stock price fluctuations.
2. Social-media sentiment predicts meme-stock price movements.
3. Retail behaviour, seen through trading-volume spikes, amplifies the effect
   of sentiment on prices.

## Data

| Series | Source | Period |
|---|---|---|
| GME daily close and volume | Yahoo Finance (`yfinance`) | Jan 2020 – Dec 2021 |
| VIX | Yahoo Finance (`^VIX`) | Jan 2020 – Dec 2021 |
| Reddit sentiment: daily mean VADER compound score of post titles in r/GME, r/DDintoGME, r/GMEJungle | *GameStop Reddit Dataset*, Han & Iliadis, Temple University (public domain) | Jan – Dec 2021 |

501 daily observations after alignment.
[`data/daily_reddit_sentiment.csv`](data/daily_reddit_sentiment.csv) gives
the daily sentiment used in the analysis, overall and per subreddit, with post
counts. Raw posts (with usernames) are not redistributed.

## Method

Descriptive statistics and time-series plots → correlation matrix → OLS
regression of the GME close on volume, sentiment and VIX → wavelet analysis of
the sentiment / price relationship across time scales.

Script: [`analysis/mirrors_and_markets_analysis.py`](analysis/mirrors_and_markets_analysis.py)
(see the notes at the top of the file).

## Main results (as reported in the dissertation)

- Strong negative correlation between sentiment and price (−0.897), weak
  correlation between price and volume (0.046), −0.462 between price and VIX.
- OLS: R² = 0.820 (F = 752.5). Sentiment (−48.18) and volume are significant
  at the 1% level; VIX is not (p = 0.125).
- Wavelet coherence between sentiment and price ranges from 0 to 0.54 and is
  stronger at longer time scales; the phase pattern suggests sentiment tends
  to lead price during major events.
- The dissertation concludes that the results support the reflexivity and
  sentiment hypotheses, and discusses implications for retail investors,
  market surveillance and financial education.

## Known limitations — revisited in 2026

A re-examination of this work identified several weaknesses: price modelled
in levels (non-stationary), strong multicollinearity in the OLS specification
(condition number 2.7 × 10⁸), sentiment back-filled before the start of the
Reddit data (January 2021), and VADER scores saturating at the ceiling on
WallStreetBets language — which can produce the negative sentiment/price
correlation above.

The follow-up paper addresses these points with FinBERT sentiment, return-based
specifications and GARCH-MIDAS models:
[`paper-2026/paper-2-memestocks.pdf`](paper-2026/paper-2-memestocks.pdf) —
*Mirrors and Markets: Quantifying Reflexivity in Meme Stocks via Sentiment
Cascades and GARCH-MIDAS — GameStop, 2020–2024* (draft, May 2026).

## Repository

```
dissertation/  final dissertation (PDF)
analysis/      analysis script (last saved version, 1 Sep 2024)
data/          daily Reddit sentiment used as input
paper-2026/    2026 follow-up paper (draft)
```

## Data credit

Han, J. & Iliadis, A. *GameStop Reddit Dataset* (2023), Temple University —
public domain.
