
# embodiedAI
CS-7643 Habitat AI group project

Two folders were created as there are some very slight differences in the code (ex: the import for the baseline utils). It would be nice to refactor it to 1 set of files, but only if we have time.

Prereqs:
Setup habitat-challenge, habitat-lab, and habitat_baselines following the instructions on their GitHub pages.

 - Habitat Challenge: https://github.com/facebookresearch/habitat-challenge/blob/master/README.md
 - Habitat Lab: https://github.com/facebookresearch/habitat-lab/blob/master/README.md
 - Habitat Lab Baselines:  https://github.com/facebookresearch/habitat-lab/blob/master/habitat_baselines/README.md

Before moving on, confirm you can at least run the random agent tests in habitat-challenge, and be able to train using the default hyperparameters using the base PPO model in habitat-lab/habitat_baselines.

Unzip gibson v2 training data https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v2/pointnav_gibson_v2.zip to habitat-lab/data/datasets/pointnav/gibson/v2


To run training:
Copy code from habitat-lab folder in embodiedAI, and replace the files in habitat-lab/habitat_baselines with them. Copy over the challenge_pointnav2020.local.rgbd.yaml from the challenge configs to habitat-lab/configs/tasks

deepVO_cnn.py should go to habitat_baselines/rl/models
config replaces the one in habitat_baselines/config/pointnav
policy file replaces the one in habitat_baselines/rl/ppo

Once all the files have been replaced run the below command in habitat-lab:

    python -u habitat_baselines/run.py --exp-config habitat_baselines/config/pointnav/ppo_pointnav_example.yaml --run-type train

Checkpoint models are stored in habitat-lab/data/new_checkpoints

To run challenge test:

Copy code from habitat-challenge folder in embodiedAI and drop it in your setup habitat-challenge folder.
Once done run the following commands in your setup habitat-challege folder:

    #Builds Docker container
    docker build . --file Pointnav_DeepVO_PPO.Dockerfile -t pointnav_submission
    #Runs container on local Challenge test set
    ./test_locally_pointnav_rgbd.sh --docker-name pointnav_submission


