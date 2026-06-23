The goal is not to classify gait disorders.
The goal is not to detect diseases.
The goal is to connect movement data to simulation.


Working idea:
- Use publicly available gait data from healthy and pathological populations to identify characteristic gait patterns.
- Use those patterns to create distinct controller variants.
- Run those controller variants in a shared simulation environment.
- Measure stability and robustness.
- Compare how different populations respond under the same conditions.


Potential contribution:
- Many papers study gait datasets, gait classification, simulation, or locomotion controllers separately.
- Fewer papers combine all of these components into a single framework.

## Stability Analysis Framework

The project will evaluate controller variants using multiple complementary definitions of locomotor stability.

Candidate metrics currently include:
- Lyapunov Exponents
- Floquet Multipliers
- Basin of Support
- Total Body Angular Momentum
- Traditional locomotion metrics (success rate, distance traveled, completed strides)
- Potentially Uncontrolled Manifold Analysis


The project will not assume that a single stability metric fully characterizes locomotion stability. Instead, results will be compared across metrics to identify which measures are most sensitive to different pathological gait patterns.