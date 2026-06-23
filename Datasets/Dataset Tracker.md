# Dataset Tracker

Dataset: Gait in Parkinson's Disease (PhysioNet)

Populations:
- Parkinson's Disease
- Healthy Controls

Sample Size:
- 93 Parkinson's patients
- 73 healthy controls

Data Type:
- Vertical ground reaction force recordings
- Foot sensor data during approximately 2 minutes of walking at self-selected pace

Useful Features:
- Stride timing
- Stance/swing timing
- Gait variability
- Left-right asymmetry
- Force distribution patterns

Potential Controller Modifications:
- Altered gait rhythm
- Increased stride variability
- Reduced movement consistency
- Parkinsonian timing changes

Confidence:
- Very High

Notes:
- Strongest dataset identified so far
- Large sample size
- Clean healthy vs Parkinson's comparison
- Best starting dataset for the project


--------------------------------------------------


Dataset: Gait Dynamics in Neurodegenerative Disease (PhysioNet)

Populations:
- Parkinson's Disease
- Huntington's Disease
- ALS
- Healthy Controls

Sample Size:
- 15 Parkinson's
- 20 Huntington's
- 13 ALS
- 16 healthy controls

Data Type:
- Foot force recordings from force-sensitive resistors
- Derived footfall timing measures

Useful Features:
- Left stride interval
- Right stride interval
- Left swing interval
- Right swing interval
- Left stance interval
- Right stance interval
- Double-support interval
- Walking speed
- Disease severity/duration information

Potential Controller Modifications:
- Parkinson's: altered rhythm and timing
- Huntington's: increased variability / irregular gait dynamics
- ALS: weakness-related gait changes
- Healthy: comparison baseline

Confidence:
- Very High

Notes:
- Best dataset for comparing multiple neurodegenerative populations
- All groups are in the same dataset format
- Sample size is smaller than the Parkinson's-specific dataset but compatibility is better across groups


--------------------------------------------------


Dataset: Cerebral Vasoregulation in Elderly with Stroke (PhysioNet)

Populations:
- Stroke
- Healthy Controls

Sample Size:
- 60 stroke subjects
- 60 control subjects

Data Type:
- Multimodal stroke dataset
- Includes gait pressure recordings during walking test
- Also includes cardiovascular, cerebral blood flow, EMG, ECG, accelerometer, and other physiologic signals

Useful Features:
- Gait pressure recordings
- Step-related walking data
- Possibly walking speed / movement-task information
- Potential pressure asymmetry

Potential Controller Modifications:
- Stroke-related asymmetry
- Unilateral weakness
- Altered stance/swing behavior
- Reduced walking stability

Confidence:
- Medium-High

Notes:
- Stroke data exists and is usable, but it is not a clean gait-only dataset
- Need to inspect the walking files directly before deciding how many features can actually be extracted
- Still worth including as a candidate population


--------------------------------------------------


Dataset: Cerebral Perfusion and Cognitive Decline in Type 2 Diabetes (PhysioNet)

Populations:
- Type 2 Diabetes
- Healthy Controls

Sample Size:
- 70 type 2 diabetes subjects
- 70 healthy controls

Data Type:
- Multimodal diabetes dataset
- Includes gait variables
- Includes balance measures
- Includes foot pressure distribution and center-of-pressure displacement

Useful Features:
- Foot pressure distribution
- Center-of-pressure displacement
- Balance-related gait variables
- Possible instability-related measures

Potential Controller Modifications:
- Reduced balance control
- Altered foot pressure / foot placement
- Increased instability
- Sensory/balance-related gait impairment

Confidence:
- Medium

Notes:
- This is NOT confirmed as a diabetic neuropathy dataset
- Safer wording is “type 2 diabetes-related gait impairment”
- Could still be useful because it includes gait, balance, and foot-pressure data
- Need to inspect whether neuropathy status or relevant clinical severity markers are included


--------------------------------------------------


Dataset: Multimodal Gait Dataset of Brain Activity, Muscle Activity, Kinematics, and Ground Forces in Young Adults (PhysioNet)

Populations:
- Healthy young adults

Sample Size:
- 59 healthy adults

Data Type:
- EEG
- EMG
- IMU kinematics
- Bilateral force plates
- Ground reaction forces
- Center of pressure
- Three treadmill speeds: 0.5, 0.75, and 1.0 m/s

Useful Features:
- Healthy kinematics
- Muscle activation
- Ground reaction forces
- Speed-dependent gait changes
- Center-of-pressure data

Potential Controller Modifications:
- Healthy baseline validation
- Reference for normal gait timing and force patterns

Confidence:
- High

Notes:
- Excellent healthy reference dataset
- Not directly pathological
- Useful for validating baseline healthy gait characteristics
- More complex than needed for first-pass analysis


--------------------------------------------------


Dataset: Multi-Camera and Multimodal Dataset for Posture and Gait Analysis (PhysioNet)

Populations:
- Healthy adults

Sample Size:
- 14 healthy subjects

Data Type:
- Multi-camera recordings
- Depth camera data
- Inertial motion capture
- Posture and gait trials

Useful Features:
- Visual gait/posture data
- Kinematic movement patterns
- Camera-based gait information

Potential Controller Modifications:
- Not directly useful for pathology-specific controllers
- Could support pose/video pipeline later

Confidence:
- Medium

Notes:
- Healthy only
- Small sample size
- Useful mainly if the project later includes a video/pose-estimation component
- Not essential for the current core pipeline


--------------------------------------------------


Current Working Population Set

Strong Core Populations:
- Healthy Controls
- Parkinson's Disease
- Huntington's Disease
- ALS

Promising Additional Populations:
- Stroke
- Type 2 Diabetes-Related Gait Impairment

Important Caution:
- Diabetic neuropathy is not confirmed from the current PhysioNet diabetes dataset
- Stroke is usable but needs walking-file inspection
- Parkinson's / Huntington's / ALS / Healthy are the cleanest because they appear together in the neurodegenerative gait dataset


Next Verification Steps

1. Inspect the actual file structure of the Parkinson's dataset.
2. Inspect the actual `.ts` files in the neurodegenerative gait dataset.
3. Inspect the walking folder in the stroke dataset to confirm what gait variables are available.
4. Inspect the diabetes Pedar / gait files to determine whether usable stride or pressure features can be extracted.
5. Identify the overlapping features across all datasets.
6. Decide whether the final population set should include all six groups or only the cleanest four.

Potential Stability Metrics

- Lyapunov Exponents
- Floquet Multipliers
- Basin of Support
- Total Body Angular Momentum
- Traditional locomotion metrics