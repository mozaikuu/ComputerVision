Discussion
The optimized models showed markedly higher accuracy and lower error rates than baseline CNNs. For example, a PSO-based feature-selection approach achieved 98% classification accuracy on the pneumonia dataset
, while our GA-optimized VGG16 reached ~97% training accuracy and ~94% testing accuracy (error ≈0.072)
. These gains illustrate how metaheuristic tuning can systematically improve model capacity. Similarly, ACO-driven feature selection has yielded high accuracy in lung X-ray classification (up to 98.4% with SVM/ANN classifiers

), and Bayesian hyperparameter tuning is also known to boost deep network performance. In our study, the GA-optimized VGG16 notably outperformed unoptimized benchmarks, confirming prior observations that genetic and swarm-based search can discover more effective hyperparameter/configuration sets
. Importantly, these optimizations also reduce overfitting and improve generalizability. The GA approach explicitly “avoids overtraining” by pruning less useful parameters and focusing on robust architectures
. As a result, the optimized models generalized well to unseen data and across datasets
. This robustness is crucial in medical imaging, where patient populations and imaging conditions vary widely. In practical terms, higher and more reliable accuracy reduces false negatives in clinical screening. Together, the evidence suggests that metaheuristically-optimized CNNs (whether via GA, PSO, ACO, or Bayesian search) can yield significantly better diagnostic performance, making automated pneumonia detection both more accurate and more trustworthy in real-world settings
