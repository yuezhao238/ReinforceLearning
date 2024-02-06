# ReinforceLearning
## 1. DQN
$Q'(s, a) = Q(s, a) + \alpha \cdot \left(r + \gamma \cdot \max_{a'}Q\left(s', a'\right) - Q\left(s, a\right)\right)$

## 2. SARSA
$Q'(s, a) = Q(s, a) + \alpha \cdot \left(r + \gamma \cdot Q\left(s', a'\right) - Q\left(s, a\right)\right)$

## 3. SARSALambda
$Q'(s, a) = Q(s, a) + \alpha \cdot E(s, a) \cdot \left(r + \gamma \cdot Q\left(s', a'\right) - Q\left(s, a\right)\right)$

$E'(s, a) = \gamma \cdot \lambda \cdot E(s, a) + 1$

## 4. Actor-Critic
$V'(s) = V(s) + \alpha \cdot \left(r + \gamma \cdot V\left(s'\right) - V\left(s\right)\right)$

$\theta' = \theta + \alpha \cdot \left(r + \gamma \cdot V\left(s'\right) - V\left(s\right)\right) \cdot \nabla \log \pi(a|s, \theta)$
