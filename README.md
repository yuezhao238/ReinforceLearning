# ReinforceLearning
## 1. DQN
$Q'(s, a) = Q(s, a) + \alpha \cdot \left(r + \gamma \cdot \max_{a'}Q\left(s', a'\right) - Q\left(s, a\right)\right)$

## 2. SARSA
$Q'(s, a) = Q(s, a) + \alpha \cdot \left(r + \gamma \cdot Q\left(s', a'\right) - Q\left(s, a\right)\right)$

## 3. SARSALambda
$Q'(s, a) = Q(s, a) + \alpha \cdot E(s, a) \cdot \left(r + \gamma \cdot Q\left(s', a'\right) - Q\left(s, a\right)\right)$
