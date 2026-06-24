*** 测试向量dqn ***
python experiments/train_vector_dqn.py --episodes 1000 --N 20 --max_steps 5 --budget 3 --no_disconnect
增加测试轮数
python experiments/train_vector_dqn.py --episodes 1000 --N 50 --max_steps 5 --budget 3 --initial_failure_strategy highest_load -- protect_strength 2.0 --protect_duration 5 --no_disconnect
打开edge disconnection
python experiments/train_vector_dqn.py --episodes 2000 --N 50 --max_steps 5 --budget 3

python experiments/evaluate_baselines.py --episode 200 --include_dqn --dqn_path outputs/checkpoints/vector_dqn

*** 查看DQN逐步动作 ***
python experiments/inspect_policy_behavior.py
python experiments/inspect_policy_behavior.py --policy dqn --episodes 10 --dqn_path outputs/checkpoints/vector_dqn_best.pt --N 50 --max_steps 10 --budget 3 --no_disconnect --initial_failure_strategy highest_load --protect_strength 2.0 --protect_duration 5