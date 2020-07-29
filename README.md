[Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236.pdf)

# Questions from paper
- They explain three sources of instability in Q learning. The third is "correlations between action-values (Q) and target values `r + gamma * max_a' Q(s', a')` - explain this
- Explain the loss function in words
```L(theta_i) = E_{experience} [(r + gamma * max_a' Q_target(s', a') - Q_policy(s, a)**2]
```
> The loss is the squared difference between the policy net's prediction for the value of action a in state s, and the reward received for that action plus a discounted portion of the target net's predicted max reward at state s'
> In other words, we want to make a good prediction of the reward we have actually observed (r), and a stabilized estimate of the reward at one step in the future.
> Or said another way - we know that a correctly computed Q function must obey the Bellman equation!


# Algorithm

```
Initialize replay memory D to capacity N
initialize action-value function Q with random weights
# Q maps ? to ?
for episode in range(M):
  initialize sequence s1 = {x1} and preprocessed sequence phi1 = phi(s1)
  for t in range(T):
    with prob. eps, select random action
    otherwise select a_t = max_a Q(phi(s_t), a)

    execute action a_t, observe reward r_t, image x_{t+1}

    set s_{t+1} = ( s_t, a_t, x_{t+1} )
    preprocess phi_{t+1} = phi(s_{t+1})
    store transition (phi_t, a_t, r_t, phi_{t+1}) in D

    sample random minibatch of transitions (phi_j, a_j, r_j, phi_{j+1}) from D

    if phi_{j+1} is an end state:
      set y_j = r_j
    else:
      set y_j = r_j + gamma * max_a' Q(phi_{j+1}, a')

    Perform a gradient step on (y_j - Q(phi_j, a_j)^2 # see eq. 3
    # Do we need to hand-code that gradient, or just use this squared loss and call backward??
```

# Preprocessing
- Atari frames are 210x160 (128 color = 7 bit color)
- Preprocessing:
  - from RGB to grayscale
  - downsample to 110x84
  - crop to 84x84 region in playing area
- phi (preprocessing) is applied to last 4 frames of a history
  - ...so input is B x 4 x 84 x 84 tensor?
- Using AlexNet? 
- Forward pass of network simultaneously predicts the value of all possible action states. Action is not part of the network input.
- Model:
```python
  A = 4 # number of valid actions
  B = 128 # batch size
  x = torch.randn(B, 4, 84, 84)
  model = nn.Sequential(
    nn.Conv2d(4, 16, 8, stride=4), # Padding??
    nn.ReLU(inplace=True),
    nn.Conv2d(16, 32, 4, stride=2),
    nn.ReLU(inplace=True),
    nn.Flatten(),
    nn.Linear(z, 256),
    nn.ReLU(inplace=True),
    nn.Linear(256, A)
  )
```

# Hyperparams
- Game rewards are rescaled:
  - "we fixed all positive rewards to be 1 and all negative rewards to be -1, leaving 0 rewards unchanged"
- RMSProp
- batch size 32
- epsilon annealed from 1 to 0.1 over the first 1e6 frames, fixed at 0.1 afterwards
- total training 1e7 frames
- replay buffer 1e6 frames
- agent sees 1 of every k frames, and repeats last action during skipped frames
  - set k=4, except space invaders uses k=3
