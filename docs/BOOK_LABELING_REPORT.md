# Book-Based Auto-Labeling Report

**Generated:** 2026-03-08 19:19:32
**Source:** `J:\E-Books` directory structure

## Method

Images extracted from trading books carry the book title in their filename.
By matching book titles to their topic directory in J:\E-Books, we assign
topic-level labels to all images from that book.

## Results Summary

| Metric | Count |
|---|---|
| Total images | 32,136 |
| Traceable to a book | 16,515 |
| Anonymous (no book ref) | 15,621 |
| Books matched to topic | 0 |
| Books unmatched | 0 |
| Images labeled | 14,948 |
| Images unclassified | 1,567 |
| Images subsampled out | 0 |

## Topic Distribution (labeled images)

| Topic | Count |
|---|---|
| other | 8,010 |
| technical_analysis | 3,473 |
| chart_analysis | 482 |
| day_trading | 448 |
| algo_trading | 270 |
| supply_demand | 253 |
| crypto | 212 |
| strategies | 193 |
| price_action | 184 |
| elliott_wave | 138 |
| smart_money | 132 |
| sniper_trading | 131 |
| ichimoku | 110 |
| renko | 76 |
| trading_setup | 75 |
| breakout | 72 |
| order_blocks | 68 |
| rsi | 66 |
| futures_options | 60 |
| wyckoff | 60 |
| scalping | 55 |
| forex | 53 |
| bollinger_bands | 50 |
| price_volume | 46 |
| swing_trading | 42 |
| moving_average | 40 |
| sentiment | 38 |
| harmonics | 25 |
| short_term | 21 |
| median_line | 20 |
| hedging | 17 |
| quants | 10 |
| trading_psychology | 9 |
| fibonacci | 6 |
| investment | 2 |
| fundamental_analysis | 1 |

## Next Steps

1. Review topic assignments for accuracy
2. Run clustering on remaining unlabeled/anonymous images
3. Train vision model with updated labels
