# KNOWN_BUGS
1. [x] use max_rounds instead of max_generations.
2. [x] allow deterministic action when evaluation.
3. [ ] add prep_training and prep_rollout.

# ENHANCEMENT
1. [ ] eval rollouts can also be used as training data.
2. [x] tensorboard graph should use num_steps not num_rollouts as x-axis.
3. [ ] plot grad norm.
4. [ ] tensorboard graph with time as x-axis.

# CHECK
1. [ ] check the sync-training performance, especially the evaluation performance.

# REFACTOR
1. [ ] remove modified mappo codes.