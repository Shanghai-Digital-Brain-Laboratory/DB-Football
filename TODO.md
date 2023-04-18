# KNOWN_BUGS
1. [x] use max_rounds instead of max_generations.
2. [x] allow deterministic action when evaluation.
3. [x] add prep_training()=>train() and prep_rollout()=>eval().
4. [x] misuse of obs and state. see return_compute, for example.
5. [x] use actor.formard or critic.forward directly. instead we should use interfaces of policy.
6. [ ] DQN codes need to be updated like mappo.

# ENHANCEMENT
1. [ ] eval rollouts can also be used as training data.
2. [x] tensorboard graph should use num_steps not num_rollouts as x-axis.
3. [ ] plot grad norm.
4. [ ] tensorboard graph with time as x-axis.
5. [ ] learning rate decay.
6. [ ] optimization: allow right controllable to be 0.
7. [ ] allow sampling steps instead of episodes.
8. [ ] add some classic ma-envs.
9. [ ] allow not to call value function again in return computation.
10. [ ] allow composition of configs.

# CHECK
1. [ ] check the sync-training performance, especially the evaluation performance.
2. [ ] do we support mini-batch now?

# REFACTOR
1. [ ] remove modified mappo codes.
2. [ ] make a model a class rathe than a module. merge similar models. maybe use registry?
3. [ ] mappo's loss codes.
4. [ ] maybe we should add a dedicated scheduler and runner for cooperative tasks.