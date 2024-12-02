# Goals

Control a delta arm using a PID controller

# Current State

Can calculate gravity compensated control vector based on measured joint angles
- What that means: If I know a set position of the delta arm (just based on the proximal links), I can calculate a vector that will keep me there (holding against gravity)

# Next Steps

Linearlize Feedback
- Why?: To be able to use a PID controller
- How to get started? I need to calculate the inertia matrix of the delta arms (from solidworks?)

Calculate kP, kD
- Why?: To be able to accurately control the delta arm
- How? I don't know