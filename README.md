# Real-Time Multilingual Abusive Language Detection

In the current age of the internet, with an approximate total of 82 billion messages sent daily worldwide, manual moderation is no longer a feasible process. This is exactly the problem that this project tries to solve, especially given new findings showing that as little as 3.1% of extremely toxic users create up to 33% of a platform's total content. This imbalance leads to a massive volume of abusive behavior on a daily basis that requires automated intervention.

## Objective
- Decrease churn rates and safeguard brand safety in global markets through automated detection of abusive content prior to rendering to end-users.
- Perform multi-label text classification on continuous streaming data across various languages.

## Metrics
- **Primary metrics:** PR-AUC (Area Under the Precision-Recall Curve) and Macro F1-score. Since toxic or abusive comments are highly imbalanced (as most internet activity is not toxic), standard Accuracy or even ROC-AUC can be very misleading.
- **Secondary metrics:** Calibration errors. We need our softmax predictions to be calibrated correctly so that human Trust & Safety (T&S) teams can assign dynamic confidence thresholds (e.g., auto-delete anything above 0.95 and send for human review for scores between 0.70 and 0.94).

## Resources & References
- [Statista: Most Common Languages on the Internet](https://www.statista.com/statistics/262946/most-common-languages-on-the-internet/?srsltid=AYCX9vLLl-xpzBBoX63BDPmXAej-m1vpMVDk5KKhBY-QdBtYzLK3foE0)
- [DeadZones: How Many Text Messages Are Sent Per Day? (2023)](https://www.deadzones.com/2023/04/how-many-text-messages-are-sent-per-day.html)
- [StudyFinds: Americans Overestimate How Many People Post Harmful Content](https://studyfinds.com/americans-overestimate-how-many-people-post-harmful-content/)