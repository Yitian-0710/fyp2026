*** 测试向量dqn ***
python experiments/train_vector_dqn.py --episodes 200 --N 20 --max_steps 5 --budget 3 --no_disconnect
增加测试轮数
python experiments/train_vector_dqn.py --episodes 1000 --N 20 --max_steps 5 --budget 3 --no_disconnect
打开edge disconnection
python experiments/train_vector_dqn.py --episodes 2000 --N 50 --max_steps 5 --budget 3