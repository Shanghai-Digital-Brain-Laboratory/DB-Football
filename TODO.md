# KNOWN_BUGS
1. [x] use max_rounds instead of max_generations.
2. [x] allow deterministic action when evaluation.
3. [x] add prep_training()=>train() and prep_rollout()=>eval().
4. [x] misuse of obs and state. see return_compute, for example.
5. [x] use actor.formard or critic.forward directly. instead we should use interfaces of policy.
6. [ ] DQN codes need to be updated like mappo.
7. [ ] the usage of active mask.
8. [ ] to implement the new_gae_trace.

# ENHANCEMENT
1. [ ] eval rollouts can also be used as training data.
2. [x] tensorboard graph should use num_steps not num_rollouts as x-axis.
3. [ ] plot grad norm.
4. [ ] tensorboard graph with time as x-axis.
5. [ ] learning rate decay.
6. [ ] (just set to 1 now, the speed woundn't be much different.) optimization: allow right controllable to be 0.
7. [ ] allow sampling steps instead of episodes.
8. [ ] add some classic ma-envs.
9. [x] allow not to call value function again in return computation.
10. [ ] allow composition of configs.
11. [ ] true single-agent version of MAT.
12. [ ] add data shape def and check.
13. [ ] (?) re-add double clip (Tencent).
14. [ ] true v-trace, refer to IMPALA.

# CHECK
1. [x] check the sync-training performance, especially the evaluation performance.
2. [ ] do we support mini-batch now? => we need to move move_to_gpu (also, return computation) inside mini-batch iteration.
3. [ ] the implementation of RNN data generator.

# REFACTOR
1. [x] remove modified mappo codes. 
2. [ ] make a model a class rathe than a module. merge similar models. maybe use registry?
3. [ ] mappo's loss codes.
4. [ ] maybe we should add a dedicated scheduler and runner for cooperative tasks.
5. [x] maybe reshape in return_compute is unnecessary
6. [ ] use consistent names. for example, "share_obs"="states"!
7. [ ] refactor the implementation of popart(value normalization).