
To run the DQN model, following the instruction below:

Copy pointnav_with_softSPL.yaml to folder: habitat-lab/configs/tasks/

Copy dqn_pointnav_example.yaml to folder: habitat-lab/habitat_baselines/config/pointnav/

Copy dqn_trainer.py to: habitat-lab/habitat_baselines/rl/ppo/

Run the following code:
!python -u habitat_baselines/run.py --exp-config habitat_baselines/config/pointnav/dqn_pointnav_example.yaml --run-type train
