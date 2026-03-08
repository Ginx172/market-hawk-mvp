# Auto-Labeling Report -- CLIP Zero-Shot Classification

**Generated:** 2026-03-08 16:51:14
**Model:** CLIP ViT-B-32 (laion2b_s34b_b79k)
**Manifest:** `K:\_DEV_MVP_2026\Market_Hawk_3\data\chart_dataset_manifest.csv`

## Configuration

| Parameter | Value |
|---|---|
| HIGH confidence threshold | margin > 0.1 |
| MEDIUM confidence threshold | margin 0.05 - 0.1 |
| LOW confidence threshold | margin < 0.05 |
| Text prompts | 16 categories |

## Results Summary

| Metric | Count | % |
|---|---|---|
| Total processed | 94,531 | 100% |
| Pre-labeled (skipped) | 1,656 | - |
| HIGH confidence (trusted) | 10 | 0.0% |
| MEDIUM confidence (needs review) | 257 | 0.3% |
| LOW confidence (unlabeled) | 94,264 | 99.7% |
| Total auto-labeled | 267 | 0.3% |

**Average confidence score:** 0.3034
**Average margin:** 0.0117
**Elapsed time:** 3697.6s

## Label Distribution (auto-labeled only)

| Pattern | Count |
|---|---|
| support_resistance | 126 |
| generic_chart | 62 |
| head_and_shoulders | 26 |
| candlestick | 22 |
| fibonacci | 9 |
| channel | 8 |
| cup_and_handle | 7 |
| elliott_wave | 3 |
| wedge | 2 |
| trend | 1 |
| bullish_flag | 1 |

## Text Prompts Used

| Label | Prompt |
|---|---|
| head_and_shoulders | a stock chart showing head and shoulders pattern |
| double_top | a stock chart showing double top pattern |
| double_bottom | a stock chart showing double bottom pattern |
| ascending_triangle | a stock chart showing ascending triangle pattern |
| descending_triangle | a stock chart showing descending triangle pattern |
| bullish_flag | a stock chart showing bullish flag pattern |
| bearish_flag | a stock chart showing bearish flag pattern |
| cup_and_handle | a stock chart showing cup and handle pattern |
| wedge | a stock chart showing wedge pattern |
| channel | a stock chart showing channel pattern |
| support_resistance | a stock chart showing support and resistance levels |
| fibonacci | a stock chart showing fibonacci retracement |
| candlestick | a stock chart showing candlestick pattern |
| trend | a stock chart showing trend line analysis |
| elliott_wave | a stock chart showing elliott wave pattern |
| generic_chart | a generic stock price chart with no specific pattern |

## Next Steps

1. Review images in `data/review_queue.csv` (MEDIUM confidence)
2. Manually verify a sample from HIGH confidence labels
3. Retrain vision model with updated labels
