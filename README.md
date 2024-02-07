# ReinforceLearning
FOR STUDY AND RESEARCH, PERFOMANCE NOT GUARANTEED

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

## 5. A2C
$A(s, a) = Q(s, a) - V(s)$

$V'(s) = V(s) + \alpha \cdot \left(r + \gamma \cdot V\left(s'\right) - V\left(s\right)\right)$

$\theta' = \theta + \alpha \cdot A(s, a) \cdot \nabla \log \pi(a|s, \theta)$

## 6. TRPO
$\theta' = \theta + \alpha \cdot \nabla \log \pi(a|s, \theta) \cdot A(s, a) + \beta \cdot \nabla \left(\nabla \log \pi(a|s, \theta) \cdot A(s, a)\right)$

## 7. PPO
$\theta' = \theta + \alpha \cdot \min\left(\rho(\theta) \cdot A(s, a), \text{clip}\left(\rho(\theta), 1-\epsilon, 1+\epsilon\right) \cdot A(s, a)\right) \cdot \nabla \log \pi(a|s, \theta)$
