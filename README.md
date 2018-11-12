# banana-navigation
A reinforcement learning exercise to navigate to avoid blue bananas and get yellow ones.

## Project Details

The objective of this project is to show how, through using the DQN algorithm, an agent is trained to navigate through to collect yellow bananas while avoiding blue ones.

#### 1. The environment and state space

Our environment is a Unity environment made by the good people in Udacity!

In this environment, every step returns a state as a list with `37` values. The best part is that neither I nor the agent even know what these 37 values represent (_besides what the project initiator brief i.e. 'velocity' and 'ray-based perception of objects'_) and neither do we need to. What's more important is that these `37` values are consistent with the state of the environment for us to be able to learn something.

#### 2. The action spaces

The simulation contains a single agent that navigates a large environment.  At each time step, it has four actions at its disposal:
- `0` - walk forward 
- `1` - walk backward
- `2` - turn left
- `3` - turn right

#### 3. The reward
A reward of `+1` is provided for collecting a yellow banana, and a reward of `-1` is provided for collecting a blue banana. 


## Getting Started
* Do `pip install -r requirements.txt` to set up your environment.
* Run jupyter notebook and open up `Report.ipynb`to see how to train the agent and get its performance.

### Instructions
The project is split into 2 main parts: __training__ and __playing__.

#### Training
`Navigation.ipynb` shows how training is done and the learning progress (using episodic scores) leading up to the final intelligent agent. 

#### Playing
A python script (`play.py`) will then be used to import the agent's learned weights and used to visualize the agent playing through the environment.

Existing successful saved model weights to be used in `play.py` are found in `./data/network-weights/{dqn-default.pth, doubledqn-default.pth}` 

#### Extra:
Check out `./_misc/` to see the video recording of the agent's progress. 
