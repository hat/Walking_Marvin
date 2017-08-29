# Walking_Marvin
Using OpenAI Gym to teach Marvin how to walk

[GymAI Docs](https://gym.openai.com/docs)

# Thought Processes

After messing with the code of examples I have come to realize
that AI is all about saving previous iterations and the reward
value that came with that decision. The goal is to then test
a random decision against the save decision and if it is better
you then update the saved decision.

# Observations

Each set of actions is an array of 2 different float values (Ex. [-0.01040708 -0.03782682 -0.02203254  0.00013583]). There is a variety of different mathematical matrice equations
catered to the evolution strategy to find different float values that may be best. You then
go about getting one of those values and sending it as a action then comparing the reward
value and saving it if the reward is better than the previous iterations reward.