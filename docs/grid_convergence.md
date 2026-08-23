# Estudio de convergencia de malla

| Caso | D | nx | ny | Re | U_inf | nu | tau | steps | t* | Lr/D | Cd | Cl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D=20 | 20 | 420 | 200 | 20 | 0.05 | 0.05 | 0.65 | 2000 | 5 | 1.00 | 2.465 | ≈ 0 |
| D=40 | 40 | 840 | 400 | 20 | 0.025 | 0.05 | 0.65 | 8000 | 5 | 0.95 | 2.426 | ≈ 0 |

En ambos casos, el número de Reynolds se mantuvo constante en `Re = 20`, al igual que la viscosidad cinemática `nu = 0.05` y el tiempo de relajación `tau = 0.65`. También se mantuvo el mismo tiempo adimensional, `t* = t U_inf / D = 5`.

Al duplicar la resolución del cilindro de `D = 20` a `D = 40`, `Cd` cambió aproximadamente un 1.6 %, mientras que `Lr/D` cambió de `1.00` a `0.95`. Esto sugiere una sensibilidad relativamente pequeña al refinamiento entre estas dos resoluciones, pero no constituye todavía una demostración completa de independencia de malla porque solo se dispone de dos niveles de resolución.
