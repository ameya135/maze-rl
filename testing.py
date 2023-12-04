
from maze import (
    create_env,
    create_model,
    train,
    save_model_as_pythorch,
)

model = create_model()
env = create_env()
train(agent=model, env=env, learning_rate=0.01, hidden_size=32, num_cycles=10)
save_model_as_pythorch(model, "output")
