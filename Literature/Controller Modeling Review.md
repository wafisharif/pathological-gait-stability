How have previous researchers modeled pathological gait in simulation?

Goal:
Determine which controller modifications are most justified for representing Parkinson's disease, Huntington's disease, ALS, and stroke.

Main conclusion

Researchers usually model pathological gait in simulation by changing mechanics, actuation strength, timing, feedback control, delays, or noise. They usually do not directly train a full disease-specific controller from raw clinical data. That supports our plan: use real gait datasets to extract population gait signatures, then create controller variants by modifying a baseline controller.

How researchers have modeled gait/control in simulation
1. Reflex-based neuromuscular walking models

Geyer and Herr developed a muscle-reflex model of human walking where gait emerges from muscle reflexes interacting with legged mechanics. The model reproduced human-like walking dynamics and muscle activity, tolerated ground disturbances, and adapted to slopes without manually changing parameters. This is important because it shows that walking can be modeled through feedback/reflex control rather than only pre-planned joint trajectories.

For our project, this suggests controller variants can modify:

feedback gains
feedback delays
muscle/actuator strength
stance/swing timing rules
disturbance response behavior
2. Feedback and feedforward control

Later studies built on the Geyer-Herr model and tested combinations of feedback and feedforward control during perturbations. One study modified only the neuronal control while keeping the body model fixed, which is directly relevant to us because our project also wants to compare controller variants under the same simulated body/environment.

For our project, this supports the idea that pathology variants should not just change “stride time.” They may also need changes in:

corrective feedback strength
anticipatory control
delayed responses
sensory reliability
motor noise
3. Perturbation and robustness simulations

Several walking simulation papers test how controllers handle disturbances like rough terrain or step-down perturbations. These studies often evaluate whether the controller can keep walking, recover after perturbation, or maintain stable dynamics.

For our project, this supports using perturbation tests as part of the stability comparison, not just normal walking.

Disease-specific modeling notes
Stroke

Stroke is the best-supported pathology for simulation modeling. Researchers often model post-stroke gait through unilateral weakness, activation deficits, asymmetry, and impaired muscle groups.

Knarr et al. simulated post-stroke activation deficits by constraining muscle activation capacity in plantar flexors, dorsiflexors, and hamstrings. They found that models could compensate for isolated impairments but could not recreate normal gait when several muscle groups were impaired at once. This is directly relevant to our controller variants because it supports modeling stroke through reduced activation/strength in specific muscle groups.

Santos et al. used predictive simulation for post-stroke gait abnormalities such as drop foot, stiff-knee gait, and knee hyperextension. This supports the idea that stroke gait can be represented through specific biomechanical impairments rather than a vague “stroke controller.”

A newer RL musculoskeletal simulation study induced asymmetric gait by reducing right-leg muscle strength to 75%, 50%, and 25% of baseline. As weakness increased, gait asymmetry increased, especially in ankle motion, stance timing, and loading. This is extremely relevant because it gives us a very concrete implementation idea: a stroke-like controller can be created through unilateral actuation-strength reduction.

Project mapping:

Stroke signature: asymmetry, altered stance/swing timing, unilateral impairment
Controller modification: reduce strength/activation on one side, alter feedback correction, possibly change stance timing
Best justification: stroke literature directly supports weakness/asymmetry modeling
Parkinson’s disease

Parkinson’s simulation literature often focuses on altered rhythm, reduced step length, freezing, feedback, basal ganglia control, and noise/delay effects.

A computational model of Parkinsonian gait during doorway walking combined a basal-ganglia reinforcement-learning model with a spinal rhythm/CPG-like model. The simulated Parkinsonian conditions produced reduced velocity, shorter stride length, and increased step variability, especially near narrow doorways. This is useful because it supports modeling Parkinson’s not only as weakness, but as altered rhythm/control and increased variability.

A recent postural-control simulation study explored Parkinson’s-related effects using signal noise and neural delays. It tested increased delays and noise configurations, which strongly supports Miles’ point that feedback mechanisms and noise should be considered in the controller concepts.

Project mapping:

Parkinson’s signature: altered rhythm, increased variability, reduced consistency, possible delayed correction
Controller modification: timing variability, feedback delay, sensory/motor noise, reduced rhythmic consistency
Best justification: Parkinson’s modeling often involves rhythm/control disruptions, not just lower strength
Huntington’s disease

Huntington’s disease is less developed in simulation literature, but the gait-analysis literature is strong. The key recurring feature is variability.

One study found that Huntington’s patients had reduced stride length and gait velocity, increased stride/stance time, and much larger variability. Stride time variability increased strongly and correlated with disease severity.

Another Huntington’s study used phase-plot analysis and found that gait variability/symmetry measures could detect subtle changes, including in premanifest Huntington’s disease.

There are also studies using autoregressive modeling of stride-time patterns in Huntington’s disease to assess gait stability, again showing that temporal stride variability is a meaningful disease feature.

Project mapping:

Huntington’s signature: high gait variability, irregular timing, reduced regularity
Controller modification: timing noise, motor noise, irregular step timing, variability injection
Best justification: HD gait literature strongly supports variability as the core feature
Limitation: less direct “Huntington’s simulation controller” work exists, so we may model HD using variability/noise literature rather than disease-specific simulation papers
ALS

ALS simulation-specific work is weaker than stroke and Parkinson’s. The gait literature supports weakness and reduced walking performance.

Classic ALS gait work found reduced velocity, cadence, and stride length as walking performance declined. Patients spent less time in single-limb stance and more time in double support, and lower-extremity muscle strength was the clinical feature most strongly related to walking performance.

A more recent ALS wearable-sensor study found that walking speed decreased over time in many ALS patients, especially those who later transitioned to assistive device use.

Project mapping:

ALS signature: weakness, slower walking, reduced stride length/cadence, increased double support
Controller modification: reduce global or lower-limb actuation strength, reduce force capacity, possibly increase double-support timing
Best justification: ALS gait appears most defensibly modeled as weakness/force-capacity decline
Limitation: ALS-specific simulation literature is limited, so weakness modeling literature will be important
General impairment mechanisms we can use
Weakness / reduced actuation

This is the most defensible mechanism for stroke and ALS. Stroke can use unilateral weakness; ALS can use more global or progressive weakness. Prior simulations have directly used reduced muscle activation capacity and unilateral strength reduction.

Use for:

Stroke
ALS

Possible controller parameters:

max actuator strength
muscle activation scale
unilateral strength scale
torque limits
Asymmetry

Asymmetry is especially important for stroke. It can be created by reducing actuation strength on one side, changing stance/swing timing on one side, or modifying feedback asymmetrically. The unilateral weakness RL study is especially useful because it shows strength reduction produced measurable gait asymmetry.

Use for:

Stroke

Possible controller parameters:

left/right strength imbalance
left/right timing imbalance
asymmetric feedback gains
Timing variability / rhythm disruption

This is important for Parkinson’s and Huntington’s. Parkinson’s literature supports altered rhythm, reduced stride length, and increased variability. Huntington’s literature strongly supports increased stride-time and swing/stance variability.

Use for:

Parkinson’s
Huntington’s

Possible controller parameters:

step period variation
timing noise
variable gait phase transitions
altered rhythmic controller frequency
Feedback delay / feedback degradation

This is especially important because Miles specifically mentioned feedback mechanisms. Reflex-based walking models use sensory feedback, gains, and delays to generate stable walking. Parkinson’s postural-control modeling also explored neural delays and signal noise.

Use for:

Parkinson’s
Stroke
Possibly Huntington’s

Possible controller parameters:

feedback delay
feedback gain
sensory signal reliability
correction strength after perturbation
Noise

Noise is a good way to model variability without pretending we know the exact neural cause. It may be useful for Parkinson’s and Huntington’s, and maybe for impaired sensory/motor control more generally. Parkinson’s modeling literature explicitly considers signal noise and delay; Huntington’s gait literature strongly supports increased variability.

Use for:

Huntington’s
Parkinson’s

Possible controller parameters:

motor command noise
sensory noise
stride timing noise
phase transition noise
Stability metrics literature

Bruijn et al.’s review is the anchor paper for stability metrics. It reviews many measures of gait stability, including maximum Lyapunov exponent, maximum Floquet multiplier, variability measures, long-range correlations, extrapolated center of mass, stabilizing/destabilizing forces, foot-placement estimator, gait sensitivity norm, and maximum allowable perturbation. The key lesson is that each metric has strengths and weaknesses; no single measure captures all of locomotor stability.

Floquet multipliers measure whether a periodic gait returns toward its nominal cycle after perturbation. Bruijn et al. note that Floquet theory is well-grounded for periodic systems, but applying it to biological walking is tricky because human gait is not perfectly periodic.

Whole-body angular momentum is another useful stability-related measure because human walking normally regulates angular momentum within a limited range through segmental cancellation. Excessive angular momentum may reveal poor whole-body rotational control.

Project implication:

Do not use only success rate.
Use multiple metrics.
Expect metrics to disagree sometimes.
That disagreement may itself become an important result.
Notes for our actual controller mapping
Healthy controller

Use as baseline:

normal timing
lower variability
symmetric left/right behavior
normal actuation strength
normal feedback/noise settings
Parkinson’s controller

Literature-supported possible modifications:

altered rhythmic timing
increased stride variability
reduced step/stride length if feasible
feedback delay or increased signal noise
reduced recovery consistency after perturbation
Huntington’s controller

Literature-supported possible modifications:

high stride-time variability
high swing/stance variability
irregular gait timing
stronger motor/timing noise than Parkinson’s
ALS controller

Literature-supported possible modifications:

reduced actuation strength
reduced force capacity
slower walking speed
increased double-support behavior
reduced single-limb support
Stroke controller

Literature-supported possible modifications:

unilateral weakness
asymmetric actuation
asymmetric stance/swing timing
altered loading
impaired corrective feedback
Most important finding from this review

The literature does not say: “Here is exactly how to build a Parkinson’s controller, Huntington’s controller, ALS controller, and stroke controller in MuJoCo.”

What it does say is more useful:

Stroke is often modeled through weakness, activation deficits, and asymmetry.
ALS can reasonably be modeled through weakness and reduced force capacity.
Huntington’s is strongly characterized by variability and irregular timing.
Parkinson’s is characterized by rhythm/timing disruptions, variability, feedback/control issues, and sometimes delay/noise mechanisms.
Human walking simulation often relies on feedback/reflex control, so feedback gains, delays, and noise are legitimate controller dimensions to consider.