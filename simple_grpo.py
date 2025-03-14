import numpy as np


# Simulate a policy (e.g., a simple probability distribution over actions 0-9)
class SimplePolicy:
    def __init__(self):
        # Initialize logits for 10 actions (numbers 0-9), roughly uniform
        self.logits = np.ones(10) * 0.1

    def get_probs(self):
        # Convert logits to probabilities using softmax
        exp_logits = np.exp(self.logits)
        return exp_logits / exp_logits.sum()

    def sample(self):
        # Sample an action based on current probabilities
        probs = self.get_probs()
        return np.random.choice(10, p=probs)


# Reward function: how close is the guess to the target (7)?
def reward_function(action, target=7):
    return -abs(action - target)  # Negative distance as reward (higher is better)


# GRPO update step
def grpo_update(policy, actions, rewards, learning_rate=0.1, kl_penalty=0.01):
    # Step 1: Compute group-based advantage
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards) + 1e-6  # Avoid division by zero
    advantages = (rewards - mean_reward) / std_reward  # Normalized advantage

    # Step 2: Compute old probabilities (reference policy)
    old_probs = policy.get_probs()

    # Step 3: Gradient update with KL penalty (simplified)
    for i, action in enumerate(actions):
        # Policy gradient: advantage * log(prob)
        grad = advantages[i] * old_probs[action]
        # Add KL-like regularization (simplified penalty)
        kl_approx = np.sum((old_probs - policy.get_probs()) ** 2)  # Rough KL proxy
        policy.logits[action] += learning_rate * (grad - kl_penalty * kl_approx)


# Main training loop
policy = SimplePolicy()
num_steps = 200
group_size = 4  # Number of samples per group

for step in range(num_steps):
    # Step 1: Generate multiple actions (group sampling)
    actions = [policy.sample() for _ in range(group_size)]

    # Step 2: Score actions with reward function
    rewards = [reward_function(action) for action in actions]

    # Step 3: Update policy using GRPO
    grpo_update(policy, actions, rewards)

    # Print progress
    if step % 10 == 0:
        avg_reward = np.mean(rewards)
        probs = policy.get_probs()
        print(f"Step {step}: Avg Reward = {avg_reward:.2f}, Prob of 7 = {probs[7]:.3f}")

# Final policy probabilities
print("Final probabilities:", policy.get_probs())
