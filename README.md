# embodiedAI
CS-7643 Habitat AI group project

Two folders were created as there are some very slight differences in the code (ex: the import for the baseline utils). It would be nice to refactor it to 1 set of files, but only if we have time. Habitat-challenge contains the code to test the agent against the local Habitat Sim test set. Habitat-lab contains the code needed for training the agent.
## Setup

### Prereqs
Setup habitat-challenge, habitat-lab, and habitat_baselines following the instructions on their GitHub pages. **Running Habitat requires a Linux OS because of Nvidia Docker dependency**

 - Habitat Challenge: https://github.com/facebookresearch/habitat-challenge/blob/master/README.md
 - Habitat Lab: https://github.com/facebookresearch/habitat-lab/blob/master/README.md
 - Habitat Lab Baselines:  https://github.com/facebookresearch/habitat-lab/blob/master/habitat_baselines/README.md

Before moving on, confirm you can at least run the random agent tests in habitat-challenge, and be able to train using the default hyperparameters using the base PPO model in habitat-lab/habitat_baselines.

Unzip gibson v2 training data https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v2/pointnav_gibson_v2.zip to habitat-lab/data/datasets/pointnav/gibson/v2


### Training
Copy code from habitat-lab folder in embodiedAI, and replace the files in habitat-lab/habitat_baselines with them. Copy over the challenge_pointnav2020.local.rgbd.yaml from the challenge configs to habitat-lab/configs/tasks

deepVO_cnn.py should go to habitat_baselines/rl/models
config replaces the one in habitat_baselines/config/pointnav
policy file replaces the one in habitat_baselines/rl/ppo

Once all the files have been replaced run the below command in habitat-lab to start training:

    python -u habitat_baselines/run.py --exp-config habitat_baselines/config/pointnav/ppo_pointnav_example.yaml --run-type train

Training can also be run in the background so closing the console wont interrupt the process:

    #running in background
    nohup python -u habitat_baselines/run.py --exp-config habitat_baselines/config/pointnav/ppo_pointnav_example.yaml --run-type train &
    #killing the process
    #1 find pid using below command
    ps ax
    #2 kill the process using the pid
    kill -9 <pid>

Checkpoint models are stored in habitat-lab/data/new_checkpoints

### Testing

Copy code from habitat-challenge folder in embodiedAI and drop it in your setup habitat-challenge folder.
Once done run the following commands in your setup habitat-challege folder:

    #Builds Docker container
    docker build . --file Pointnav_DeepVO_PPO.Dockerfile -t pointnav_submission
    #Runs container on local Challenge test set
    ./test_locally_pointnav_rgbd.sh --docker-name pointnav_submission

If you want to change the RGBD models out all that is required is to move the pretrained model into the habitat-challenge directory, and change the Dockerfile to place it in the image. If you want to run the Blind agent you must also change deep_vo to force is_blind to True.
