# Canonical Gram/Lie Research Note

## Motivation
The current `O(d)` Takens/projective pipeline is attractive because it is causal, light enough to run on long streams, and already strong enough to expose meaningful precursor geometry in cardiac and neural data. Its weakness is that the cheapest invariant bundle is not always sensitive enough by itself. In Sleep-EDF, the baseline projective score remains computationally efficient but weak on recall around some stage transitions. The same issue matters in LTST for future alarm refinement: we want more local dynamical information without abandoning the low-cost state representation. The goal of this note is therefore to lock one canonical extension, based on derivatives, Gram symmetries, and local Lie-type decomposition, so the math and the code do not drift by domain.

## Current Baseline Math
Let the scalar observable generate a causal delay state
\[
x_t \in \mathbb{R}^d.
\]
Let the short-memory and long-memory priors be
\[
m_{s,t},\; m_{l,t} \in \mathbb{R}^d,
\]
with both priors produced by causal EMA filters on the centered delay state. The baseline projective pipeline compares the present state to these two memory states using only inner products, norms, and low-rank Gram determinants. Representative quantities are built from
\[
\langle x_t, m_{s,t}\rangle,\quad \langle x_t, m_{l,t}\rangle,\quad \langle m_{s,t}, m_{l,t}\rangle,
\]
and the associated state energies. For a fixed small embedding dimension and a fixed three-state stack, the per-sample cost stays `O(d)` because every nontrivial quantity is reduced to dot products and norm-like summaries of vectors in `R^d`.

## Derivative Extension
To enrich the local dynamics while staying causal, define the three-column state stack
\[
Q_t = [x_t,\; m_{s,t},\; m_{l,t}] \in \mathbb{R}^{d \times 3},
\]
and the one-step backward difference
\[
\Delta Q_t = Q_t - Q_{t-1}.
\]
This keeps the same low-dimensional internal bundle used by the baseline projective pipeline, but now also records how that bundle is moving. The extension does not require a higher-order global reconstruction; it only augments the present memory geometry with a causal local derivative.

## Gram/Lie Objects
The canonical derivative objects are
\[
K_t = Q_t^\top Q_t + \lambda_t I,
\]
\[
C_t = Q_t^\top \Delta Q_t,
\]
\[
B_t = K_t^{-1} C_t,
\]
\[
A_t = \frac{1}{2}\left(B_t - K_t^{-1} B_t^\top K_t\right),
\]
\[
S_t = \frac{1}{2}\left(B_t + K_t^{-1} B_t^\top K_t\right).
\]
Here `\lambda_t` is a small trace-scaled regularizer used only to stabilize inversion when the three columns in `Q_t` become nearly degenerate. By construction, `A_t` is the `K_t`-skew part and `S_t` is the `K_t`-symmetric part of the local generator candidate `B_t`.

## Symmetry Interpretation
The matrix `K_t` is the instantaneous internal metric induced by the current three-column bundle. It measures the local geometry of the present state and the two memory states in the column space of `Q_t`. With respect to that metric, `A_t` lives in the Lie algebra of the `K_t`-orthogonal group, because it is the infinitesimal part that preserves `K_t` to first order. In that sense `A_t` acts like a local orbit or rotation generator. The complementary term `S_t` is the `K_t`-symmetric deformation, so it acts like local strain. This is the bridge from the earlier projective-collapse picture to an approximate orbit picture: the baseline geometry says how collapsed the temporal bundle is, while the derivative Gram/Lie sidecar says how that bundle is moving, rotating, and deforming.

## Coadjoint-Orbit Caution
These derivative and Gram-symmetry quantities are candidate infinitesimal generators, not a proof of a true global Lie action, a closed orbit family, or an exact coadjoint orbit. The construction is local and empirical. It supplies a useful low-rank approximation to the instantaneous internal dynamics of the state stack, but it does not establish a globally consistent representation-theoretic action on the full signal space. The correct claim is therefore modest: `A_t` and `S_t` give local orbit-like and strain-like summaries induced by the metric `K_t`, and those summaries may still be predictive even when no exact global symmetry exists.

## Why The Divergence Construction Was Added
The original derivative Gram/Lie sidecar was enough to distinguish many transition families from the sparse baseline projective detector, but it also exposed a gap. In the pair-conditioned Sleep-EDF mining, some transition families, especially `N2->N3` and `N3->N2`, showed strong Lie activity without becoming visible to either alert path. In other words, the data were not quiet; they were dynamically active in a way that the existing orbit/strain score alone did not resolve cleanly.

That forced a refinement in the theory. We needed a quantity that separates

- structural alignment or collapse of the local state bundle,
- from active local drift-gain or escape of that bundle.

The derivative Gram/Lie objects tell us that motion is present, but not whether the local kinematic energy is roughly preserving its scale or rapidly amplifying. That is why the divergence construction was added. It is not a replacement for the orbit/strain decomposition. It is a theory-side diagnostic added to distinguish stable aligned precursors from dynamically escaping aligned precursors.

## Empirical Hypothesis
The working empirical hypothesis is that pre-transition windows can show elevated strain, elevated orbit intensity, or stronger orbit-strain coupling before annotated state changes. In Sleep-EDF, that means stage-transition windows may carry stronger `lie_strain_norm`, `lie_orbit_norm`, `lie_commutator_norm`, `lie_metric_drift`, or `lie_transition_score` even when the original projective alert score misses the event. In LTST, the same sidecar is carried at beat level so that future cardiac refinement can ask whether pre-ST dynamics become more orbit-dominant, more strain-dominant, or more strongly coupled before onset.

## O(1) Gram Divergence Proxy
To capture finite-time drift gain without introducing a full Lyapunov estimate, define the local kinematic energy
\[
D^2_t = \|\Delta Q_t\|_F^2.
\]
This quantity is already available from the same causal three-column bundle and requires only the current and previous state stack.

Define the finite-time growth ratio
\[
R_t = \frac{D^2_t}{D^2_{t-1} + \varepsilon}.
\]
When `R_t` is close to one, the local kinematic energy is roughly preserving scale. When it is much larger than one, the bundle is undergoing rapid local amplification.

Using the first-order finite-time approximation, define the divergence proxy
\[
\lambda_{\text{proxy}, t}
\approx
\frac{R_t - 1}{2\Delta t}.
\]
Here `\Delta t` is the sample interval of the current stream. In the streaming implementation this proxy is intentionally lightweight:

- one delayed `D^2`,
- one ratio,
- one subtraction,
- one fixed division by `2\Delta t`.

So the digital or firmware-facing cost of the proxy is effectively `O(1)` on top of the already computed drift bundle.

This proxy is included as a diagnostic construction, not as a detector term. The theory role is interpretive:

- if structural-collapse quantities drop while `R_t \approx 1`, the system may be entering a more aligned but still locally stable precursor,
- if structural-collapse quantities drop while `R_t \gg 1`, the system may be aligned and actively escaping,
- if the Lie score is high but the divergence proxy remains modest, the dynamics may be broad but not explosively amplifying.

That distinction is exactly what the pair-conditioned Sleep-EDF mining demanded, because some depth-exchange transitions showed strong Lie activity without the same type of visible precursor seen in wake/light-stage transitions.

## Implementation Bridge
The canonical implementation is shared across domains in [`geometry_lie.py`](/C:/Users/Admin/Documents/Bearings/geometry_lie.py). Both Sleep-EDF and LTST now compute the same causal EMA priors, the same projective bundle, and the same derivative Gram/Lie sidecar.

For Sleep-EDF, the samplewise canonical projective features are:
- `memory_align`
- `novelty`
- `proj_lock_barrier_sl`
- `proj_lock_barrier_xl`
- `proj_volume_xsl`

The samplewise derivative Gram/Lie features are:
- `gram_trace`
- `gram_logdet`
- `lie_orbit_norm`
- `lie_strain_norm`
- `lie_commutator_norm`
- `lie_metric_drift`
- `gram_drift_energy`
- `gram_divergence_ratio`
- `gram_divergence_proxy`

Sleep-EDF then computes the projective dynamic sidecar:
- `projective_level_norm`
- `projective_velocity_norm`
- `projective_curvature_norm`
- `transition_score`

with
\[
\texttt{transition\_score}
= \sqrt{
0.50\,\texttt{level}^2
+ 1.00\,\texttt{velocity}^2
+ 1.25\,\texttt{curvature}^2
+ 0.75\,\texttt{acc\_proj\_lock\_barrier\_sl}^2
+ 0.50\,\texttt{vel\_novelty}^2
}.
\]

At epoch level, Sleep-EDF aggregates by median:
- `memory_align`
- `novelty`
- `proj_lock_barrier_sl`
- `proj_lock_barrier_xl`
- `proj_volume_xsl`
- `projective_level_norm`
- `projective_velocity_norm`
- `projective_curvature_norm`
- `transition_score`
- `gram_trace`
- `gram_logdet`
- `lie_orbit_norm`
- `lie_strain_norm`
- `lie_commutator_norm`
- `lie_metric_drift`
- `gram_drift_energy`
- `gram_divergence_ratio`
- `gram_divergence_proxy`

and by max:
- `transition_score_peak`
- `lie_orbit_norm_peak`
- `lie_strain_norm_peak`

After robust z-normalization at epoch level, Sleep-EDF computes
\[
\texttt{lie\_transition\_score}
= \sqrt{
\texttt{lie\_orbit\_norm\_z}^2
+ \texttt{lie\_strain\_norm\_z}^2
+ 0.75\,\texttt{lie\_commutator\_norm\_z}^2
+ 0.50\,\texttt{lie\_metric\_drift\_z}^2
+ 0.25\,\texttt{gram\_logdet\_z}^2
}.
\]

For LTST, the beat-level full-cohort builder now computes and persists the same canonical projective features:
- `memory_align`
- `novelty`
- `proj_lock_barrier_sl`
- `proj_lock_barrier_xl`
- `proj_volume_xsl`

and the same derivative Gram/Lie features:
- `gram_trace`
- `gram_logdet`
- `lie_orbit_norm`
- `lie_strain_norm`
- `lie_commutator_norm`
- `lie_metric_drift`

The LTST HMM pipeline keeps the frozen alert semantics unchanged, but it now exports a sidecar canonical projective branch:
- `projective_level_norm`
- `projective_velocity_norm`
- `projective_curvature_norm`
- `transition_score`

and a Lie branch:
- `lie_transition_score`

with the same formulas used in Sleep-EDF. The legacy HMM still decodes on the original relative primitive geometry, while the canonical projective and Lie quantities are exported for comparison, stratification, and future calibration.

### Canonical Names
The canonical shared names that should not drift across files are:
- projective references: `memory_align`, `novelty`, `proj_lock_barrier_sl`, `proj_lock_barrier_xl`, `proj_volume_xsl`
- derivative Gram/Lie features: `gram_trace`, `gram_logdet`, `lie_orbit_norm`, `lie_strain_norm`, `lie_commutator_norm`, `lie_metric_drift`, `gram_drift_energy`, `gram_divergence_ratio`, `gram_divergence_proxy`
- projective dynamic sidecar: `projective_level_norm`, `projective_velocity_norm`, `projective_curvature_norm`, `transition_score`
- Lie summary score: `lie_transition_score`

These names are the ones the code should preserve across Sleep-EDF, LTST, and future domains unless there is an explicit migration.

## Complexity Note
All new samplewise quantities stay `O(d)` because the only high-dimensional objects are the three column vectors in `Q_t`. Forming `Q_t^\top Q_t`, `Q_t^\top \Delta Q_t`, and the associated orbit/strain summaries reduces the state dependence to dot products over those three columns. Every inversion or matrix symmetrization beyond the `O(d)` vector work is confined to `3 x 3` matrices. That is the core reason the derivative extension remains compatible with the original lightweight projective design.
