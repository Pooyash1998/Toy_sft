# Results for IBM Granite 4.0 350M

Fine-tuned on UltraFeedback (5000 samples). Evaluated on MMLU, CommonsenseQA, MLQA (en-en exact match), MathQA.

## Numbers

| Task | Baseline | Exp1 (lr=2e-4, r=16, 1ep) | Exp2 (lr=5e-5, r=16, 3ep) | Exp3 (lr=1e-4, r=32, 3ep) |
|------|----------|--------------------------|--------------------------|--------------------------|
| MMLU | 33.8% | 26.9% (−6.9pp) | 28.2% (−5.6pp) | 26.6% (−7.2pp) |
| CommonsenseQA | 29.7% | 23.1% (−6.6pp) | 26.5% (−3.3pp) | 23.7% (−6.1pp) |
| MLQA (EM) | 14.6% | 11.1% (−3.5pp) | **19.2% (+4.6pp)** | **18.7% (+4.1pp)** |
| MathQA | 35.7% | 27.5% (−8.2pp) | 30.1% (−5.6pp) | 29.0% (−6.7pp) |


Fine-tuning hurts everything except MLQA. MMLU, CommonsenseQA, and MathQA all drop after SFT regardless of config. This is hinting a catastrophic forgetting maybe because UltraFeedback is preference/chat data adn so the model drifts away from what it already knew.

MLQA is the odd one out. Exp2 and Exp3 both improve it after fine-tuning (+4.6pp and +4.1pp). My guess is that UltraFeedback has enough reading comprehension style exchanges that some of it transfers to extractive QA. Exp1 destroys MLQA too because the LR is too aggressive.

Exp2 was the best imo. lowest forgetting on every task. Alos Exp3 shows that bumping the rank to 32 doesn't help if the LR is also higher. Exp1 with LR=2e-4 is just too aggressive for one epoch.

## Next experiments i would try if time allows

- **Exp4**: lr=1e-5, r=16, 3 epochs to keep pushing LR down
- **Exp5**: lr=5e-5, r=16, 1 epoch to check if exp2 is overtraining at 3 epochs
- **Exp6**: lr=5e-5, r=64, 3 epochs maybe try higher rank again when i find the the right LR