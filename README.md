# Walking_Marvin
Using OpenAI Gym to teach Marvin how to walk

[GymAI Docs](https://gym.openai.com/docs)

# Thought Processes

After messing with the code of examples I have come to realize
that AI is all about saving previous iterations and the reward
value that came with that decision. The goal is to then test
a random decision against the save decision and if it is better
you then update the saved decision.