# Canonical Gram/Lie Research Note

## Motivation
The current `O(d)` projective pipeline is attractive because it is causal, light enough to run on long streams, and already strong enough to expose meaningful precursor geometry in cardiac and neural data. Its weakness is that the cheapest invariant bundle is mostly an even-sector summary: it captures collapse strength and broad deformation well, but it throws away too much orientation.

That matters in two places:

- the real projective locks square the pairwise alignments and erase angle direction,
- the Lie sidecar exports `lie_orbit_norm`, which is useful but still collapses an odd generator to an even magnitude.

The goal of this note is to lock one canonical extension that preserves the current stable contract while adding an oriented sidecar. The sidecar stays real-valued and `O(d)`, but it is designed to act like the missing complex or antisymmetric sector.

## Current Baseline Math
Let the scalar observable generate a centered causal delay state
\[
x_t \in \mathbb{R}^d.
\]
Let the short-memory and long-memory priors be
\[
m_{s,t},\; m_{l,t} \in \mathbb{R}^d,
\]
with both priors produced by causal EMA filters on the centered delay state.

The baseline three-column bundle is
\[
Q_t = [x_t,\; m_{s,t},\; m_{l,t}] \in \mathbb{R}^{d \times 3}.
\]
The baseline projective pipeline compares the present state to the two memory states using only inner products, norms, and low-rank Gram determinants. Representative quantities are built from
\[
\langle x_t, m_{s,t}\rangle,\quad \langle x_t, m_{l,t}\rangle,\quad \langle m_{s,t}, m_{l,t}\rangle,
\]
and the associated energies.

The existing canonical projective outputs remain:

- `memory_align`
- `novelty`
- `proj_lock_barrier_sl`
- `proj_lock_barrier_xl`
- `proj_volume_xsl`

Important clarification:

- `proj_volume_xsl` is currently an unsigned structural magnitude.
- It should not be described as a signed branch variable.
- In the current code it is clipped to `[0, 1]`, so any orientation sign is intentionally absent.

## Shared Lie Decomposition
The derivative sidecar augments the bundle with the one-step backward difference
\[
\Delta Q_t = Q_t - Q_{t-1}.
\]
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

Here `\lambda_t` is a small trace-scaled regularizer used only to stabilize inversion when the three columns in `Q_t` become nearly degenerate.

By construction:

- `A_t` is the `K_t`-skew part,
- `S_t` is the `K_t`-symmetric part,
- the commutator `[S_t, A_t]` measures orbit-strain coupling.

The current canonical Lie outputs remain:

- `gram_trace`
- `gram_logdet`
- `lie_orbit_norm`
- `lie_strain_norm`
- `lie_commutator_norm`
- `lie_metric_drift`
- `gram_drift_energy`
- `gram_divergence_ratio`
- `gram_divergence_proxy`

Another important clarification:

- `lie_orbit_norm` is an even summary of an odd generator.
- It is informative, but it discards the directional content of `A_t`.

## Metric Cartan Decomposition And Stiefel Interpretation

The split `B_t = A_t + S_t` is not just a convenient implementation detail. It is the canonical decomposition of the in-frame transition generator into metric-skew and metric-symmetric parts.

### Proposition 1. Unique `K_t`-skew / `K_t`-symmetric split

For any symmetric positive-definite `K_t`, define the `K_t`-adjoint of a `3 x 3` matrix `M` by
\[
M^{\dagger_{K_t}} = K_t^{-1} M^\top K_t.
\]
Then the two subspaces
\[
\mathfrak{so}(K_t) = \{M : M^{\dagger_{K_t}} = -M\},
\]
\[
\mathrm{sym}(K_t) = \{M : M^{\dagger_{K_t}} = M\}
\]
give a direct-sum decomposition
\[
\mathfrak{gl}(3) = \mathfrak{so}(K_t) \oplus \mathrm{sym}(K_t).
\]

The map `M \mapsto M^{\dagger_{K_t}}` is an involution, so `\mathfrak{so}(K_t)` and `\mathrm{sym}(K_t)` are exactly its `-1` and `+1` eigenspaces. Therefore
\[
A_t = \frac{1}{2}\left(B_t - B_t^{\dagger_{K_t}}\right),
\qquad
S_t = \frac{1}{2}\left(B_t + B_t^{\dagger_{K_t}}\right)
\]
are the unique `K_t`-skew and `K_t`-symmetric parts of `B_t`.

This split is orthogonal with respect to the Frobenius-`K_t` inner product
\[
\langle M, N \rangle_{K_t} = \operatorname{tr}(M^\top K_t N).
\]
Hence
\[
\langle A_t, S_t \rangle_{K_t} = 0,
\]
and the corresponding energy decomposition is
\[
\|B_t\|_{K_t}^2 = \|A_t\|_{K_t}^2 + \|S_t\|_{K_t}^2,
\]
where
\[
\|M\|_{K_t}^2 = \operatorname{tr}(M^\top K_t M).
\]

So `lie_orbit_norm` and `lie_strain_norm` are not arbitrary summaries. They are the Pythagorean components of the in-frame generator energy.

### Proposition 2. Whitening sends the metric split to the ordinary split

Let `H_t` be any invertible matrix satisfying
\[
K_t = H_t^\top H_t.
\]
Define the whitened matrices
\[
\widetilde{B}_t = H_t B_t H_t^{-1},
\qquad
\widetilde{A}_t = H_t A_t H_t^{-1},
\qquad
\widetilde{S}_t = H_t S_t H_t^{-1}.
\]
Then
\[
\widetilde{A}_t^\top = -\widetilde{A}_t,
\qquad
\widetilde{S}_t^\top = \widetilde{S}_t.
\]

So whitening sends the metric split exactly to the ordinary decomposition
\[
\mathfrak{gl}(3) = \mathfrak{so}(3) \oplus \mathrm{Sym}(3).
\]

This is the rigorous content behind the phrase "metric-adapted orbit generator": after whitening, the orbit really is an ordinary skew matrix and the strain really is an ordinary symmetric matrix.

### Proposition 3. QR gauge and the Stiefel tangent space

Assume first that `\lambda_t = 0` and `Q_t` has rank `3`. Take the thin QR factorization
\[
Q_t = U_t R_t,
\]
with
\[
U_t \in \mathrm{St}(d,3),
\qquad
R_t \in GL^+(3),
\]
and positive diagonal in `R_t` to fix the gauge. Then
\[
K_t = Q_t^\top Q_t = R_t^\top R_t.
\]
Setting `H_t = R_t`, define
\[
Y_t = \Delta Q_t\,R_t^{-1}.
\]
Then
\[
U_t^\top Y_t = R_t B_t R_t^{-1} = \widetilde{B}_t.
\]

The orthogonal projection of an ambient perturbation `Y_t \in \mathbb{R}^{d \times 3}` onto the Stiefel tangent space is
\[
P_{T_{U_t}\mathrm{St}(d,3)}(Y_t)
=
Y_t - U_t \,\mathrm{sym}(U_t^\top Y_t).
\]
Substituting `U_t^\top Y_t = \widetilde{B}_t` gives
\[
P_{T_{U_t}\mathrm{St}(d,3)}(Y_t)
=
U_t\,\mathrm{skew}(\widetilde{B}_t)
+ (I - U_t U_t^\top)Y_t.
\]

This has three immediate consequences:

- `\widetilde{A}_t = \mathrm{skew}(\widetilde{B}_t)` is the body-coordinate infinitesimal rotation of the orthonormal 3-frame.
- `\widetilde{S}_t = \mathrm{sym}(\widetilde{B}_t)` is the in-frame obstruction to remaining purely tangent or isometric.
- The normal component `(I - U_t U_t^\top)Y_t` is not captured by `\widetilde{B}_t` at all, because `\widetilde{B}_t = U_t^\top Y_t` only sees the in-frame part of the dynamics.

So the Gram/Lie sector is an exact description of the internal frame dynamics, but it is blind to purely ambient reorientation of the 3-plane that does not feed back into the internal Gram coordinates.

### Corollary. Directional orbit channels are a basis decomposition of orbit energy

Write the whitened skew matrix as
\[
\widetilde{A}_t =
\begin{bmatrix}
0 & \omega_{xs} & \omega_{xl} \\
-\omega_{xs} & 0 & \omega_{sl} \\
-\omega_{xl} & -\omega_{sl} & 0
\end{bmatrix}.
\]
Then
\[
\|\widetilde{A}_t\|_F^2 = 2(\omega_{xs}^2 + \omega_{xl}^2 + \omega_{sl}^2),
\]
and therefore
\[
\|\widetilde{B}_t\|_F^2
=
\|\widetilde{A}_t\|_F^2 + \|\widetilde{S}_t\|_F^2
=
2(\omega_{xs}^2 + \omega_{xl}^2 + \omega_{sl}^2) + \|\widetilde{S}_t\|_F^2.
\]

So the three directional orbit channels decompose the orbit energy itself, while `lie_strain_norm` carries the orthogonal residual in-frame deformation energy.

### Proposition 4. Regularized case

When `\lambda_t > 0`, the matrix
\[
K_t = Q_t^\top Q_t + \lambda_t I
\]
is still symmetric positive-definite, so Propositions 1 and 2 remain exact with any factor `H_t` satisfying `K_t = H_t^\top H_t`.

What changes is the geometric gauge. In general,
\[
K_t \neq R_t^\top R_t
\]
once `\lambda_t > 0`, so the exact QR/Stiefel interpretation in Proposition 3 no longer uses `R_t` directly. Instead it uses the regularized metric factor `H_t`. Thus:

- the algebraic split remains exact in the regularized metric,
- the QR/Stiefel interpretation is recovered in the limit `\lambda_t \to 0` on full-rank rows,
- and the two whitenings agree operationally whenever `\lambda_t` is negligible relative to the spectrum of `Q_t^\top Q_t`.

This is the right way to read the causal regularizer: it changes the metric, not the underlying Cartan-type splitting.

## Hermitian Analogy
The right mental model is not the real Gram matrix by itself. It is a Hermitian split into even and odd sectors.

In the Shuttle or complex picture, one would write
\[
h_{ab} = g_{ab} + i q_{ab},
\]
\[
\rho_{ab} = \frac{h_{ab}}{\sqrt{g_{aa} g_{bb}}}.
\]
Then:

- `Re(\rho_{ab})` is the pair alignment or cosine sector,
- `Im(\rho_{ab})` is the quadrature or oriented sine sector,
- `arg(\rho_{ab})` is the signed pair angle.

The delay or memory pipeline in `geometry_lie.py` remains real-valued, so it does not literally compute complex pair products. Instead it builds causal oriented surrogates that play the same role.

## Causal Oriented Frame
Because the bundle lives in `R^d` with `d` possibly larger than `3`, ordinary 3D cross products are not defined directly in the ambient space. The implementation therefore constructs a local oriented reference frame first.

For each row:

1. Normalize `x_t`, `m_{s,t}`, and `m_{l,t}` when their energies exceed the configured floor.
2. Build a reduced QR frame from the previous bundle:
   \[
   Q_{t-1} = U_{t-1} R_{t-1}.
   \]
3. Force a positive diagonal on `R_{t-1}`.
4. Propagate orientation causally by checking the sign of
   \[
   \det(U_{\mathrm{ref}}^\top U_t).
   \]

This matters because a tall reduced frame `U_t \in \mathbb{R}^{d \times 3}` does not have a standalone ambient determinant. Its orientation must be fixed relative to another 3-frame, not by pretending it is square.

With the oriented reference basis in hand, the current normalized columns are expressed in that basis and treated as 3-vectors for the orientation calculations.

## Oriented Pair Sidecar
The oriented sidecar exports the following pairwise quantities for the pairs `xs`, `xl`, and `sl`:

- `pair_align_*`
- `pair_angle_*`
- `pair_quadrature_*`
- `pair_phase_ratio_*`
- `oriented_pair_angle_*`

The even pair geometry is:
\[
\texttt{pair\_align}_{ab} = \langle \hat a_t, \hat b_t \rangle,
\]
\[
\texttt{pair\_angle}_{ab} = \arccos(\texttt{pair\_align}_{ab}).
\]

The odd or complex surrogate sector is defined using the previous oriented pair normal. Let `n_{ab,\mathrm{prev}}` be the previous oriented pair normal in the 3D reference basis. Then the code keeps:

- the current principal-angle magnitude from `pair_angle_ab`,
- the orientation sign from the current pair normal relative to `n_{ab,\mathrm{prev}}`.

That yields:
\[
\texttt{pair\_quadrature}_{ab}
=
\operatorname{sgn}(\text{current normal} \cdot n_{ab,\mathrm{prev}})
\sqrt{1 - \texttt{pair\_align}_{ab}^2},
\]
\[
\texttt{oriented\_pair\_angle}_{ab}
=
\operatorname{atan2}(\texttt{pair\_quadrature}_{ab}, \texttt{pair\_align}_{ab}),
\]
\[
\texttt{pair\_phase\_ratio}_{ab}
=
\frac{\texttt{pair\_quadrature}_{ab}}{\texttt{pair\_align}_{ab} + \varepsilon}.
\]

So, in the real pipeline:

- `pair_align_ab` acts like `Re(\rho_ab)`,
- `pair_quadrature_ab` acts like `Im(\rho_ab)`,
- `oriented_pair_angle_ab` acts like `arg(\rho_ab)`,
- `pair_phase_ratio_ab` acts like a tangent-style phase ratio.

The contrast
\[
\texttt{oriented\_angle\_contrast\_xs\_xl}
=
\texttt{oriented\_pair\_angle\_xs}
-
\texttt{oriented\_pair\_angle\_xl}
\]
is the odd-sector analog of the older explanation imbalance features.

### Phase Coherence
The three signed pair angles need not rotate coherently. The reusable sidecar therefore adds
\[
\texttt{phase\_coherence\_residual\_xs\_xl\_sl}
=
\texttt{oriented\_pair\_angle\_xs}
-
\texttt{oriented\_pair\_angle\_xl}
-
\texttt{oriented\_pair\_angle\_sl}.
\]

When this residual is near zero, the `xs`, `xl`, and `sl` pair phases are approximately closure-consistent and the local bundle behaves like a single-mode oriented geometry. Large magnitude indicates cross-scale phase decoupling between the present state, short-memory lane, and long-memory lane.

#### Projective Sheaf Consistency Theorem

**§ 1. Definitions.**

Fix a timestep $t$. The three unit vectors $\hat{x}_t, \hat{s}_t, \hat{l}_t$ in the local 3D oriented reference basis form the vertices of a spherical triangle $\Delta(\hat{x},\hat{s},\hat{l})$ on $S^2$.

**(D1) Side lengths.** The unsigned arc lengths (geodesic distances) between pairs:
\[
a = \arccos(c_{xs}), \quad b = \arccos(c_{xl}), \quad c = \arccos(c_{sl}),
\]
where $c_{ab} = \langle \hat{a}, \hat{b} \rangle$. Each lies in $[0, \pi]$.

**(D2) Interior angles.** The dihedral angle $A_l$ at vertex $\hat{l}$ is the angle between the great-circle arcs $\hat{l}\hat{x}$ and $\hat{l}\hat{s}$, defined by the spherical law of cosines:
\[
\cos a = \cos b \cos c + \sin b \sin c \cos A_l.
\]

**(D3) Volume.** The instantaneous scalar triple product:
\[
V = \det[\hat{x}, \hat{s}, \hat{l}] = \hat{x} \cdot (\hat{s} \times \hat{l}).
\]
Its magnitude is the unsigned spherical volume:
\[
|V| = \sin b \sin c \sin A_l.
\]
The implementation exports a causally oriented continuation of this magnitude:
\[
\texttt{oriented\_volume\_xsl} = \tau_t |V|,
\]
where `\tau_t \in \{-1,+1\}` is the sign induced by aligning the current short-long normal with the previous short-long normal. Thus the code preserves `|V|` exactly, while the sign is a causal orientation convention rather than the raw instantaneous sign of `\det[\hat{x},\hat{s},\hat{l}]`.

**(D4) Berry phase (solid angle).** The holonomy of parallel transport around $\Delta$ on $S^2$, given by the van Oosterom–Strackee formula:
\[
\Omega = 2\operatorname{atan2}(V, 1 + c_{xs} + c_{xl} + c_{sl}).
\]
For small triangles, $\Omega \approx V / 2$.

**(D5) Closure defect.** The implemented quantity:
\[
\Phi_t = \theta_{xs} - \theta_{xl} - \theta_{sl},
\]
where $\theta_{ab} = \mathrm{atan2}(q_{ab}, c_{ab}) \in (-\pi, \pi]$ is the principal-value oriented pair angle. This is `phase_coherence_residual_xs_xl_sl`.

For nondegenerate pairs (`0 < a,b,c < \pi`) define the pair sign
\[
\sigma_{ab} = \operatorname{sgn}(q_{ab}) \in \{-1,+1\}.
\]
Then the implemented angle satisfies the exact identity
\[
\theta_{ab} = \sigma_{ab} a_{ab}.
\]

We call a timestep **sign-coherent** if
\[
\sigma_{xs} = \sigma_{xl} = \sigma_{sl} =: \sigma.
\]
In that regime,
\[
\Phi_t = \sigma(a-b-c)
\]
exactly.

**(D6) Sheaf defect log-ratio.**
\[
\texttt{sheaf\_defect\_log\_ratio} = \log\!\left(\frac{|\Phi_t| + \varepsilon}{|V| + \varepsilon}\right).
\]

**§ 2. Lemma (Cosine Defect Identity).**

*Statement.* For the unsigned side lengths $a, b, c$ and interior angle $A_l$:
\[
\cos a - \cos(b + c) = \sin b \sin c \, (\cos A_l + 1) = 2 \sin b \sin c \cos^2\!\left(\frac{A_l}{2}\right).
\]

*Proof.* Expand $\cos(b+c) = \cos b \cos c - \sin b \sin c$. Subtract from the spherical law of cosines $\cos a = \cos b \cos c + \sin b \sin c \cos A_l$:
\[
\cos a - \cos(b+c) = \sin b \sin c (\cos A_l + 1). \qquad\square
\]

**Corollary.** $\cos a = \cos(b+c)$ if and only if $A_l = \pi$, i.e., $\hat{l}$ lies on the great circle through $\hat{x}$ and $\hat{s}$ on the far side. This is the unique coplanar configuration where unsigned side lengths compose additively ($a = b + c$ when $a \leq \pi$).

**§ 3. Lemma (Operational-Regime Approximation).**

*Statement.* Let $c = |\theta_{sl}| \ll 1$ (high memory alignment). Assume sign coherence, so $\theta_{sl} = \sigma c$ with $\sigma \in \{-1,+1\}$. Then:
\[
\Phi_t = -2\,\theta_{sl}\,\cos^2\!\left(\frac{A_l}{2}\right) + O(\theta_{sl}^2).
\]

*Proof.* In the sign-coherent regime, `\Phi_t = \sigma(a-b-c)` exactly, so it is enough to expand the unsigned defect `a-b-c`. Write $a = b + \delta$ where $\delta$ is the deviation from flat composition. Using $\cos(b + \delta) \approx \cos b - \delta \sin b$ and the spherical law of cosines with $\cos c \approx 1 - c^2/2$, $\sin c \approx c$:
\[
\cos b - \delta \sin b \;\approx\; \cos b \cdot (1 - c^2/2) + \sin b \cdot c \cdot \cos A_l.
\]
Solving for $\delta$:
\[
\delta \approx -c \cos A_l + O(c^2).
\]
Then
\[
a-b-c = \delta - c = -c(1 + \cos A_l) + O(c^2) = -2c\cos^2(A_l/2) + O(c^2).
\]
Multiplying by the common sign `\sigma` and using `\theta_{sl} = \sigma c` gives
\[
\Phi_t = \sigma(a-b-c) = -2\theta_{sl}\cos^2(A_l/2) + O(\theta_{sl}^2). \qquad\square
\]

**§ 4. Theorem (Projective Sheaf Consistency).**

Let $\hat{x}_t, \hat{s}_t, \hat{l}_t \in S^2$ with unsigned spherical volume $|V|$, causally oriented volume `\texttt{oriented\_volume\_xsl}`, interior angle $A_l$ at $\hat{l}$, and closure defect $\Phi_t$. Assume sign coherence unless stated otherwise.

**(T1) Characterization of vanishing.**
In the operational regime ($|\theta_{sl}| \ll 1$, sign-coherent, consistent causal normals):
\[
\Phi_t = -2\theta_{sl}\cos^2(A_l/2) + O(\theta_{sl}^2).
\]
To first order in $\theta_{sl}$, the defect vanishes if and only if $A_l = \pi$ (coplanar with $\hat{l}$ on the far arc). This is not an exact equivalence outside the sign-coherent first-order regime.

**(T2) Asymptotic upper bound.**
$|\Phi_t| \leq 2\theta_{sl} + O(\theta_{sl}^2)$, with leading-order equality attained when $A_l = 0$.

**(T3) Coplanarity link.**
Assuming nondegeneracy ($\sin b > 0$, $\sin c > 0$), we have $|V| = 0 \;\Longleftrightarrow\; A_l \in \{0, \pi\}$. The defect $\Phi_t$ distinguishes between the two coplanar configurations:

| Configuration | $A_l$ | $V$ | $|\Phi_t|$ (leading order) |
|---|---|---|---|
| $\hat{l}$ on far arc | $\pi$ | $0$ | $0$ |
| $\hat{l}$ on near arc | $0$ | $0$ | $2\theta_{sl}$ |
| Perpendicular | $\pi/2$ | $\sin b \cdot \theta_{sl}$ | $\theta_{sl}$ |

**(T4) Defect–volume ratio.**
In the operational regime with $|V| \neq 0$:
\[
\frac{|\Phi_t|}{|V|} \;\approx\; \frac{1 + \cos A_l}{\sin b \cdot |\sin A_l|} \;=\; \frac{\cot(A_l/2)}{\sin(\theta_{xl})}.
\]

This ratio isolates the interior angle $A_l$ at the memory junction $\hat{l}$. It is independent of $\theta_{sl}$ to leading order, which means `sheaf_defect_log_ratio` measures the *shape* of the spherical triangle, not its size.

*Proof.* From § 3, $|\Phi_t| \approx 2|\theta_{sl}|\cos^2(A_l/2)$. From (D3), $|V| \approx \sin b \cdot |\theta_{sl}| \cdot |\sin A_l|$. The $|\theta_{sl}|$ factors cancel:
\[
\frac{|\Phi_t|}{|V|} \approx \frac{2\cos^2(A_l/2)}{\sin b \cdot |\sin A_l|} = \frac{2\cos^2(A_l/2)}{\sin b \cdot 2\sin(A_l/2)\cos(A_l/2)} = \frac{\cot(A_l/2)}{\sin b}. \qquad\square
\]

**§ 5. Lemma (Holonomy–Volume Link).**

*Statement.* The Berry phase $\Omega$ and the closure defect $\Phi$ are distinct quantities.

In the operational regime:
\[
|\Omega| \approx \frac{|V|}{2} \approx \frac{\sin b \cdot \theta_{sl} \cdot |\sin A_l|}{2}, \qquad |\Phi| \approx 2\theta_{sl}\cos^2\!\left(\frac{A_l}{2}\right).
\]
Their ratio:
\[
\frac{|\Phi|}{|\Omega|} \approx \frac{4\cos^2(A_l/2)}{\sin b \cdot |\sin A_l|} = \frac{2\cot(A_l/2)}{\sin b}.
\]

$\Phi$ measures the *side-composition defect* (how far the pair angles deviate from additive composition). $\Omega$ measures the *area of the spherical triangle* (the holonomy of parallel transport). They coincide only when $A_l$ and $b$ are fixed, i.e., for triangles of the same shape. The `sheaf_defect_log_ratio` controls both:
\[
\texttt{sheaf\_defect\_log\_ratio} \;\approx\; \log\!\left(\frac{\cot(A_l/2)}{\sin b}\right).
\]

**§ 6. Corollaries.**

**(C1) Isoplanar regime.** When `sheaf_defect_log_ratio` $\to -\infty$ while $\sin b$ stays bounded away from zero, the interior angle $A_l \to \pi$. Geometrically, the three lanes collapse to a single common rotation plane.

**(C2) Near-arc degeneracy or memory-junction collapse.** When `sheaf_defect_log_ratio` $\gg 0$, one of two geometric mechanisms must be active: either $A_l \to 0$ (the near-arc coplanar branch) or $\sin b \to 0$ (degeneracy of the `xl` side). Under an additional lower bound $\sin b \ge \beta > 0$, a large log-ratio forces $A_l \to 0$. This is therefore a detector of planar embedding failure or near-arc collapse, not necessarily of non-coplanarity.

**(C3) Persistence.** Persistent `sheaf_defect_log_ratio` $\gg 0$ for $N$ consecutive timesteps is evidence (not proof — see § 7) of sustained multi-plane dynamics. The EMA evidence machinery applied to $|\Phi_t|$ or to the log-ratio provides a streaming persistence measure.

**§ 7. Remarks.**

**What this theorem proves and does not prove.** The theorem gives a sign-coherent first-order expansion of $\Phi$ in terms of $\theta_{sl}$ and $A_l$, with a clean geometric interpretation (interior angle at the memory junction). It provides an operational characterization of when $\Phi$ is small and a shape-sensitive diagnostic (the log-ratio).

It does **not** prove that persistent $|\Phi| > \varepsilon$ implies multiple dynamical modes. A single rigid rotation applied to a non-coplanar triad produces persistent nonzero $\Phi$ without modal multiplicity. The theorem characterizes the *geometry* of the closure defect, not its *dynamics*.

**The name "Sheaf."** The theorem is inspired by sheaf-consistency ideas but does not construct a formal sheaf on a topological space. The "global section" corresponds to a consistent coplanar embedding of the three pair angles; the "defect" $\Phi$ measures deviation from this. We retain the name "Projective Sheaf Consistency" as a descriptor of the geometric content rather than a claim about sheaf cohomology.

**Chirality.** The sign of $\Phi_t$ distinguishes two handedness classes of the closure failure. Both `phase_coherence_residual_xs_xl_sl` (signed $\Phi$), `phase_coherence_abs` ($|\Phi|$), and `sheaf_defect_log_ratio` should be retained as separate features.

#### Projective Sheaf Consistency Refinement

This block supersedes the earlier first-order sketch above whenever we need an exact arc-domain identity, an explicit second-order coefficient, or a non-asymptotic remainder statement.

**Section 2R. Exact cosine and arc-defect identities.**

The exact cosine defect identity is
\[
\cos a - \cos(b+c) = 2\sin b \sin c \cos^2\!\left(\frac{A_l}{2}\right).
\]
This follows immediately by subtracting $\cos(b+c) = \cos b \cos c - \sin b \sin c$ from the spherical law of cosines.

Now let
\[
s = \frac{a+b+c}{2}, \qquad D = (b+c)-a.
\]
By the spherical triangle inequality, $D \ge 0$. Using
\[
\cos a - \cos(b+c)
=
2\sin\!\left(\frac{a+b+c}{2}\right)\sin\!\left(\frac{(b+c)-a}{2}\right),
\]
we obtain the exact half-angle form
\[
\sin(s)\,\sin\!\left(\frac{D}{2}\right) = \sin b \sin c \cos^2\!\left(\frac{A_l}{2}\right).
\]
Equivalently,
\[
\frac{D}{2} = \arcsin\!\left(\frac{\sin b \sin c \cos^2(A_l/2)}{\sin s}\right).
\]

In the sign-coherent regime,
\[
\Phi_t = \sigma(a-b-c) = -\sigma D,
\]
so $|\Phi_t| = D$ exactly. This is the exact arc-domain identity for the implemented defect.

**Section 3R. Explicit second-order expansion.**

Write
\[
a = b + \delta(c),
\]
with $\delta(0)=0$, and define
\[
f(c,\delta) = \cos(b+\delta) - \cos b \cos c - \sin b \sin c \cos A_l.
\]
Then $f(c,\delta(c)) = 0$ exactly.

At $(c,\delta) = (0,0)$,
\[
f_c = -\sin b \cos A_l, \qquad f_\delta = -\sin b,
\]
\[
f_{cc} = \cos b, \qquad f_{c\delta} = 0, \qquad f_{\delta\delta} = -\cos b.
\]
The implicit function theorem gives
\[
\delta'(0) = -\frac{f_c}{f_\delta} = -\cos A_l
\]
and, after differentiating $f(c,\delta(c)) = 0$ twice,
\[
f_{cc} + 2f_{c\delta}\delta' + f_{\delta\delta}(\delta')^2 + f_\delta \delta'' = 0,
\]
which yields
\[
\delta''(0) = \cot b \,\sin^2 A_l.
\]
Therefore
\[
a-b = -c\cos A_l + \frac{c^2}{2}\cot b \,\sin^2 A_l + O(c^3),
\]
and hence
\[
a-b-c
=
-2c\cos^2\!\left(\frac{A_l}{2}\right)
+ \frac{c^2}{2}\cot b \,\sin^2 A_l
+ O(c^3).
\]

Returning to the sign-coherent variable $\theta_{sl} = \sigma c$ gives
\[
\boxed{
\Phi_t
=
-2\theta_{sl}\cos^2\!\left(\frac{A_l}{2}\right)
+ \frac{\theta_{sl}|\theta_{sl}|}{2}\cot b \,\sin^2 A_l
+ O(|\theta_{sl}|^3)
}.
\]
The explicit second-order coefficient is therefore
\[
\frac{1}{2}\cot b \,\sin^2 A_l.
\]

Define the first- and second-order predictors
\[
\widehat{\Phi}^{(1)}_t = -2\theta_{sl}\cos^2\!\left(\frac{A_l}{2}\right),
\]
\[
\widehat{\Phi}^{(2)}_t
=
\widehat{\Phi}^{(1)}_t
+ \frac{\theta_{sl}|\theta_{sl}|}{2}\cot b \,\sin^2 A_l.
\]

**Boundary checks.**

- If $A_l = 0$, then $a = b-c$ and $a-b-c = -2c$ exactly.
- If $A_l = \pi$, then $a = b+c$ and $a-b-c = 0$ exactly.
- If $A_l = \pi/2$, then $\cos a = \cos b \cos c$ and
  \[
  a-b-c = -c + \frac{c^2}{2}\cot b + O(c^3).
  \]

These are the expected near-arc, far-arc, and perpendicular limits, and in the first two cases the quadratic term vanishes exactly because $\sin A_l = 0$.

**Section 4R. Explicit remainder bound.**

Assume sign coherence, $\sin b \ge \beta > 0$, and $|\theta_{sl}| \le c_{\max} < \pi/2$. Then the Taylor theorem with Lagrange remainder gives
\[
\left|\Phi_t - \widehat{\Phi}^{(2)}_t\right| \le R_3,
\]
where one convenient explicit envelope is
\[
R_3 \le \frac{|\theta_{sl}|^3}{6}\left(\frac{1+\cos^2 b}{\sin^3 b}\right).
\]
Consequently,
\[
\left|\Phi_t - \widehat{\Phi}^{(1)}_t\right|
\le
\frac{\theta_{sl}^2}{2}\left|\cot b\right|\sin^2 A_l
+ \frac{|\theta_{sl}|^3}{6}\left(\frac{1+\cos^2 b}{\sin^3 b}\right).
\]

*Proof sketch.* Differentiate $f(c,\delta(c)) = 0$ a third time. The resulting expression for $\delta'''$ depends only on $\sin b$, $\cos b$, and $A_l$, and is uniformly bounded on $|c| \le c_{\max}$ provided $\sin b \ge \beta$. The singular factor is $1/\sin^3 b$, which matches the same nondegeneracy obstruction already visible in the defect-to-volume ratio.

**Section 5R. Refined theorem statements.**

**(T1R) First-order vanishing law.**
\[
\Phi_t = -2\theta_{sl}\cos^2\!\left(\frac{A_l}{2}\right) + O(\theta_{sl}^2).
\]
To first order in $\theta_{sl}$, the defect vanishes if and only if $A_l = \pi$.

**(T2R) Second-order correction.**
\[
\Phi_t
=
-2\theta_{sl}\cos^2\!\left(\frac{A_l}{2}\right)
+ \frac{\theta_{sl}|\theta_{sl}|}{2}\cot b \,\sin^2 A_l
+ O(|\theta_{sl}|^3).
\]
The quadratic term vanishes whenever $\sin A_l = 0$, i.e., on the two coplanar branches.

**(T3R) Defect-volume ratio.**
For $|V| \ne 0$,
\[
\frac{|\Phi_t|}{|V|} = \frac{\cot(A_l/2)}{\sin b} + O(|\theta_{sl}|).
\]
Thus `sheaf_defect_log_ratio` is a shape diagnostic to first order: the small side $|\theta_{sl}|$ cancels, leaving only the interior geometry at the memory junction. The correction scale is
\[
O\!\left(
|\theta_{sl}|
\frac{|\cot b|\sin^2 A_l}{\sin b\,|\sin A_l|}
\right),
\]
so the ratio remains effectively size-free until the quadratic term becomes comparable to the linear one.

**(T4R) Holonomy-volume distinction.**
The Berry phase $\Omega$ and the closure defect $\Phi_t$ remain different geometric quantities. In the operational regime,
\[
|\Omega| \approx \frac{|V|}{2}, \qquad
|\Phi_t| \approx 2|\theta_{sl}|\cos^2\!\left(\frac{A_l}{2}\right),
\]
and therefore
\[
\frac{|\Phi_t|}{|\Omega|} \approx \frac{2\cot(A_l/2)}{\sin b}.
\]
So $\Phi_t$ measures side-composition failure, whereas $\Omega$ measures spherical area.

**Section 6R. Scope of claim.**

These are correspondence results, not exact equivalence theorems for arbitrary delay embeddings or arbitrary operating regimes. The refinement identifies the geometric quantity measured by the oriented defect and gives its first- and second-order expansions under sign coherence and nondegeneracy. It does **not** prove that persistent $|\Phi_t| > \varepsilon$ implies multiple dynamical modes, and it does not claim that the delay-embedding construction is globally optimal.

**Section 7R. Empirical second-order closure.**

The saved full-cohort LTST backfill materially supports the tightened theorem. Across `4,217,092` sign-coherent nondegenerate rows, the first-order predictor had `phi_mae = 0.009304`, while the explicit second-order predictor reduced this to `0.005889`, a `36.7%` improvement in MAE. The first-order residual also obeyed the predicted scaling laws:

- `corr(|r_1|, |\theta_{sl}|^2) = 0.366`, where $r_1 = \Phi_t - \widehat{\Phi}^{(1)}_t$.
- `corr(|r_1|, |\theta_{sl}|^2 \cdot \tfrac{1}{2}|\cot b|\sin^2 A_l) = 0.587`.
- `corr(|r_1| / (|\theta_{sl}|^2 + \varepsilon), \tfrac{1}{2}|\cot b|\sin^2 A_l) = 0.919`.

The coplanar prediction also closes numerically. In the near-coplanar slice $\min(A_l,\pi-A_l) \le 0.05$, the first-order defect MAE is only `0.000670`, versus `0.010232` in the bulk far from the coplanar branches.

Finally, the second-order correction becomes extremely accurate in the regime where the theorem is intended to hold. On the slice
\[
\sin b \ge 0.1, \qquad |\theta_{sl}| \le 0.04,
\]
the correction reduces
\[
\text{phi\_mae}: 7.33\times 10^{-4} \to 6.63\times 10^{-5},
\]
\[
\text{phi\_rmse}: 1.22\times 10^{-3} \to 1.35\times 10^{-4}.
\]
The remaining large-RMSE tail on the unrestricted corpus is therefore not a contradiction of the theory. It is the failure mode predicted by the $1/\sin^3 b$ remainder scaling when the `xl` side approaches degeneracy or when $|\theta_{sl}|$ leaves the asymptotic window.

#### Projective Defect vs. Continuous and Global Limits

**§ 8. The Continuous Limit (Frenet-Serret Torsion).**

*Statement.* If the discrete state sequence $x_t$ is sampled from a continuous regular curve $\gamma(t)$ in $\mathbb{R}^d$, the discrete phase defect $\Phi_t$ acts as a discrete proxy for the curve's Frenet-Serret torsion $\tau(t)$.

*Proof Sketch.* 
The continuous version of a 3-point moving bundle (Present, Short EMA, Long EMA) spans the local osculating plane of the curve.
Let $T(t), N(t), B(t)$ be the local Frenet-Serret frame.
By definition, the trajectory's velocity is along the tangent $T$, and its acceleration defines the normal $N$. The binormal $B = T \times N$ is strictly orthogonal to the local plane of motion.
The Frenet-Serret formulas define torsion as the out-of-plane buckling rate:
\[
\frac{dB}{ds} = -\tau N.
\]
In our discrete scheme, the oriented causal pair normal $n_{sl}$ (from the short-long memory pair) is the discrete analog of the bundle binormal $B$. The defect $\Phi_t$ triggers exactly when the present state $x_t$ forces a rotation of the local correlation plane, tilting the pair normal out of its previous alignment. 
Therefore, if the bundle forms a flat coplanar loop (e.g., a "healthy" heartbeat limit cycle), $\tau = 0$, the $B$ vector is perfectly static, and $\Phi_t \approx 0$ globally. If the orbit twists out of the plane (e.g., an ischemic event), $B$ tilts, putting energy into $\tau > 0$, forcing $\Phi_t$ to act as the integrated torsion over the discrete sample step. $\square$

**§ 9. The Static Global Limit (SVD Principal Axes).**

*Statement.* For a full rhythmic cycle (e.g., a single QRS-T complex), let $\{u_1, u_2, u_3\}$ be the leading left singular vectors of the local state covariance matrix $C \in \mathbb{R}^{d \times d}$. The requirement that the local phase defect $\Phi_t \approx 0$ everywhere is geometrically equivalent to the requirement that the variance eigenvalue along $u_3$ is effectively zero.

*Proof Sketch.*
By taking the full state sequence over a heartbeat window $W$ and applying a Singular Value Decomposition, the vectors $u_1$ and $u_2$ define the optimal 2D principal plane of that heartbeat. The vector $u_3$ is strictly orthonormal to this embedding plane by construction. 
From (§ 4 T1), if $\Phi_t = 0$ everywhere, $A_l = \pi$ globally. This locks the temporal vectors of the bundle into a single invariable 2D sub-manifold. This perfectly constrained plane exactly matches $\text{span}(u_1, u_2)$, leaving $u_3 \cdot x_t = 0$ everywhere in the window, meaning zero variance leaks into the orthogonal component.
Conversely, if the signal "leaks" energy into the transverse $u_3$ dimension, the global 2D planarity is broken. This forces a geometric volume $|V| \neq 0$ somewhere in the cycle, breaking the $A_l = \pi$ interior angle condition and producing a measurable $\int_W |\Phi_t| dt > 0$. 

**Proposition (Static leakage bound; one-way form).**

Let $P_W$ be the orthogonal projector onto $\operatorname{span}(u_1,u_2)$ over a window $W$, and let
\[
r_t = (I - P_W)x_t
\]
be the instantaneous leakage into the transverse singular direction. Under the same regularity and nondegeneracy assumptions used in the sheaf-defect theorem, persistent small local defect implies small average transverse leakage:
\[
\frac{1}{|W|}\int_W |\Phi_t|\,dt \ll 1
\quad\Longrightarrow\quad
\frac{1}{|W|}\int_W \|r_t\|^2\,dt \ll 1.
\]
The converse does not hold without additional persistence and coverage assumptions, because a window may have small total leakage while still containing short-lived local closure failures.

So $u_3$ variance is best interpreted as a global static constraint on the local instantaneous defect $\Phi_t$, not as a pointwise equivalent. Both bypass the raw amplitude scale by relying on orthonormal geometry, but they operate at different temporal levels: `\Phi_t` is local and causal, while the singular-vector leakage is global over the chosen window.

## Directional Orbit Sector
The Lie sidecar already computes `A_t`, but the old export collapsed it to `lie_orbit_norm`.

By the metric-Cartan/Stiefel theorem above, the full-rank unregularized QR gauge turns the metric-skew part into an ordinary skew matrix. In that gauge,
\[
K_t = R_t^\top R_t,
\]
define
\[
\Omega_t = R_t A_t R_t^{-1}.
\]

`Omega_t` is the body-coordinate infinitesimal rotation of the orthonormal 3-frame, so it is an ordinary skew-symmetric matrix in the whitened coordinates. Its independent entries are exported as:

- `lie_orbit_xs = Omega_t[0,1]`
- `lie_orbit_xl = Omega_t[0,2]`
- `lie_orbit_sl = Omega_t[1,2]`

These are the directional orbit channels missing from `lie_orbit_norm`. In the regularized case `\lambda_t > 0`, the exact whitening uses a factor `H_t` with `K_t = H_t^\top H_t`; the exported QR-whitened channels should therefore be read as the operational approximation to the exact regularized metric channels when the regularizer is small.

The sidecar also adds a few ratio summaries:

\[
\texttt{lie\_orbit\_strain\_ratio}
=
\frac{\texttt{lie\_orbit\_norm}}{\texttt{lie\_strain\_norm} + \varepsilon},
\]
\[
\texttt{lie\_orbit\_dominance}
=
\frac{\texttt{lie\_orbit\_norm} - \texttt{lie\_strain\_norm}}
{\texttt{lie\_orbit\_norm} + \texttt{lie\_strain\_norm} + \varepsilon},
\]
\[
\texttt{lie\_commutator\_coupling\_ratio}
=
\frac{\texttt{lie\_commutator\_norm}}
{\texttt{lie\_orbit\_norm} + \texttt{lie\_strain\_norm} + \varepsilon}.
\]

These are not new invariants. They are compact ways to ask whether the local dynamics are more orbit-dominant, more strain-dominant, or strongly coupled.

Important sign convention:

- positive `lie_orbit_dominance` means orbit-dominant dynamics,
- negative `lie_orbit_dominance` means strain-dominant dynamics.

## Angle-Based Oriented Volume
The new sidecar adds

- `elevation_angle_x_over_sl`
- `oriented_volume_xsl`

This is not a discrete sign stuck onto the old unsigned volume. It is angle-based.

Let the current short-long pair define a current oriented normal in the 3D reference basis, aligned continuously with the previous short-long normal. Let
\[
\texttt{pair\_angle}_{sl}
\]
be the unsigned short-long pair angle. Then the short-long area factor is
\[
\sin(\texttt{pair\_angle}_{sl}).
\]

Now define the signed elevation angle of `x_t` above the current short-long plane:
\[
\texttt{elevation\_angle\_x\_over\_sl}
=
\operatorname{atan2}(\text{lift}, \sqrt{1 - \text{lift}^2}),
\]
where `lift` is the oriented component of the current `x_t` direction along the current short-long normal in the 3D reference basis.

Then define
\[
\texttt{oriented\_volume\_xsl}
=
\sin(\texttt{pair\_angle}_{sl})
\sin(\texttt{elevation\_angle\_x\_over\_sl}).
\]

So:

- `proj_volume_xsl` remains the old unsigned structural magnitude,
- `oriented_volume_xsl` is the new angle-based oriented sidecar.

That is the correct way to talk about structural branch sign in the current code path.

## Negative-Branch Geometry
The canonical reusable branch in `geometry_lie.py` is geometric, not based on the unsigned projective volume. It is defined from the sign of the oriented volume:
\[
\texttt{structural\_branch\_sign\_xsl}
=
\begin{cases}
+1, & \texttt{oriented\_volume\_xsl} \ge 0 \\
-1, & \texttt{oriented\_volume\_xsl} < 0.
\end{cases}
\]

The sidecar also exports the one-sided negative excursion
\[
\texttt{negative\_volume\_excursion\_xsl}
=
\max(-\texttt{oriented\_volume\_xsl}, 0).
\]

This tracks when the present state drops below the short-long plane in the causal oriented frame. Existing downstream Sleep analyses sometimes label branches as `positive_proj_volume` and `negative_proj_volume`, but those are downstream aliases built on analysis-specific rules. They are not the canonical branch definition inside `geometry_lie.py`.

## Projective-Lie Inversion
The full-cohort LTST mining pass shows that the projective and Lie sectors are often adversarial rather than merely complementary. The reusable inversion summary is built from the signed baseline-to-window shifts already exported in the mining bundle.

### Empirical LTST Pattern
Define a record to exhibit **projective-Lie inversion** at a given phase when
\[
\operatorname{sgn}(\Delta \texttt{proj\_volume\_xsl})
\neq
\operatorname{sgn}(\Delta \texttt{lie\_orbit\_norm}),
\]
where `\Delta` is measured from the baseline window to either the pre-entry window or the ST window.

**Empirical fact.** On the full `86`-record LTST cohort:

- `72 / 86` records invert at ST,
- `70 / 86` records invert in the pre-entry window,
- `65 / 86` records do both,
- `69 / 86` records show barrier support, meaning the inversion coincides with the expected sign of `proj_lock_barrier_sl`.

The branch split is not symmetric.

**Empirical fact.**

- Pre-entry inversion records split into `47` dominant-branch cases and `23` reverse-branch cases.
- ST inversion records split into `42` dominant-branch cases and `30` reverse-branch cases.

Here the branches are defined by:

- dominant branch: `proj_volume_xsl` decreases while `lie_orbit_norm` increases
- reverse branch: `proj_volume_xsl` increases while `lie_orbit_norm` decreases

Inversion is present across every phenotype and every regime, not confined to a single taxonomy bucket. The full-cohort phenotype inversion rates are:

- constrained orbit: `34 / 41` at ST, `32 / 41` pre-entry
- loose orbit: `16 / 19` at ST, `16 / 19` pre-entry
- rigid orbit: `22 / 26` at ST, `22 / 26` pre-entry

The regime rates range from `0.75` to `1.0` at ST and from `0.643` to `1.0` pre-entry, so inversion is a cohort-wide operating mode rather than a narrow edge case.

The mining pass selected `s30742` as a full-score exemplar. In that record,

- `proj_volume_xsl` falls from `1.21e-6` at baseline to `1.07e-7` pre-entry and `7.58e-8` during ST,
- `lie_orbit_norm` rises from `103` at baseline to `634` pre-entry and `855` during ST,
- `proj_lock_barrier_sl` rises from `7.56` at baseline to `8.72` pre-entry and `8.76` during ST.

That is the clean canonical shape of the dominant branch.

### Dominant Branch Conjecture
**Empirical conjecture.** The dominant inversion branch is a planar locking corridor:

- `proj_volume_xsl` decreases
- `proj_lock_barrier_sl` increases
- `memory_align` often increases
- `lie_orbit_norm` increases

The interpretation is that the bundle is losing transverse projective spread while the surviving in-plane orbit strengthens. In other words, the even projective sector reports contraction of the accessible simplex, while the odd Lie sector reports intensification of the motion that remains inside the collapsing plane.

This is why the two sectors can move in opposite directions without contradiction:

- the projective volume measures non-collinearity or simplex spread,
- the Lie orbit norm measures orbital activity of the local generator,
- a bundle can become more planar and more dynamically active at the same time.

### Reverse Branch Conjecture
**Empirical conjecture.** The minority inversion branch is a release or re-expansion mode:

- `proj_volume_xsl` increases
- `lie_orbit_norm` decreases

This suggests a state that is leaving a previously locked corridor, allowing projective spread to recover while orbit intensity relaxes. The full cohort shows that this branch is real, but less common than the locking corridor branch.

This branch should be treated as a secondary empirical mode, not as a proof-level theorem. The current mining pass establishes that it exists and recurs across phenotypes; it does not yet identify a unique underlying mechanism.

### Operating-Point Dependence
Inversion is not explained by phenotype label alone. The baseline operating point matters.

**Empirical fact.** Comparing pre-entry inversion records to pre-entry non-inversion records:

- median baseline `proj_volume_xsl`: `2.04e-5` vs `5.37e-6`
- median baseline `lie_orbit_norm`: `68.28` vs `175.74`
- median baseline `memory_align`: `0.998614` vs `0.999385`
- median baseline `proj_lock_barrier_sl`: `5.891` vs `6.703`

The same direction holds at ST:

- median baseline `proj_volume_xsl`: `1.86e-5` for inversion vs `9.81e-6` for non-inversion
- median baseline `lie_orbit_norm`: `71.01` for inversion vs `104.65` for non-inversion
- median baseline `memory_align`: `0.998623` for inversion vs `0.999228` for non-inversion
- median baseline `proj_lock_barrier_sl`: `5.900` for inversion vs `6.516` for non-inversion

So the simplest prior,
\[
\text{low baseline } \texttt{proj\_volume\_xsl} \Longrightarrow \text{inversion},
\]
is not supported. A more consistent reading is:

- already-locked records may have little projective headroom left to collapse,
- records with more residual projective spread and weaker initial lock are more able to enter the dominant inversion corridor,
- therefore inversion is a statement about where the record sits on the lock-volume-orbit manifold, not just about its coarse phenotype name.

This regime dependence is also consistent with the bundle-energy summary: `gram_trace` flips direction in `30 / 86` records, so total bundle energy cannot be used as a universal sign oracle for inversion.

### What Is Supported And What Is Not
**Supported by the LTST mining pass.**

- The projective and Lie sectors are frequently adversarial.
- The adversarial pattern is structured rather than random.
- The dominant branch is consistent with planar locking.
- The reverse branch is real and recurrent.

**Not proven by the LTST mining pass.**

- Low `proj_volume_xsl` alone does not explain inversion.
- Inversion does not uniquely identify one dynamical mechanism.
- The dominant and reverse branch conjectures are not theorems of the continuous dynamics.

The right use of the inversion section is therefore:

- as a data-backed geometric interpretation layer,
- not as a substitute for the formal projective, oriented, and Lie constructions defined above.

## Divergence Proxy
The divergence construction remains unchanged. It is still a lightweight diagnostic for local drift-gain rather than a true Lyapunov estimate.

Define the local kinematic energy
\[
D_t^2 = \|\Delta Q_t\|_F^2.
\]
Define the growth ratio
\[
R_t = \frac{D_t^2}{D_{t-1}^2 + \varepsilon}.
\]
Then the proxy is
\[
\lambda_{\text{proxy}, t}
\approx
\frac{R_t - 1}{2 \Delta t}.
\]

This remains a secondary dynamical diagnostic. It should not be confused with the new oriented sector.

### Spectral Divergence
The shared Lie decomposition already separates the local generator into orbit and strain channels, so the reusable sidecar now exports a spectral version of the drift-gain proxy.

Define the causal matrix-drift energies
\[
\texttt{spectral\_orbit\_drift\_energy}_t = \|A_t - A_{t-1}\|_F^2,
\]
\[
\texttt{spectral\_strain\_drift\_energy}_t = \|S_t - S_{t-1}\|_F^2,
\]
with the first row anchored to a zero prior matrix.

Then define the orbit and strain growth ratios
\[
\texttt{spectral\_orbit\_divergence\_ratio}_t
=
\frac{\texttt{spectral\_orbit\_drift\_energy}_t}
{\texttt{spectral\_orbit\_drift\_energy}_{t-1} + \varepsilon},
\]
\[
\texttt{spectral\_strain\_divergence\_ratio}_t
=
\frac{\texttt{spectral\_strain\_drift\_energy}_t}
{\texttt{spectral\_strain\_drift\_energy}_{t-1} + \varepsilon}.
\]

Their proxy forms are
\[
\texttt{spectral\_orbit\_divergence\_proxy}_t
=
\frac{\texttt{spectral\_orbit\_divergence\_ratio}_t - 1}{2\Delta t},
\]
\[
\texttt{spectral\_strain\_divergence\_proxy}_t
=
\frac{\texttt{spectral\_strain\_divergence\_ratio}_t - 1}{2\Delta t}.
\]

These remain finite-time `O(d)` diagnostics. They are not Lyapunov exponents; they are lane-resolved growth proxies that ask whether short-horizon deformation is being driven more by orbit changes or by strain changes.

## Implementation Bridge
The canonical contract stays unchanged:

- projective references: `memory_align`, `novelty`, `proj_lock_barrier_sl`, `proj_lock_barrier_xl`, `proj_volume_xsl`
- derivative Gram/Lie features: `gram_trace`, `gram_logdet`, `lie_orbit_norm`, `lie_strain_norm`, `lie_commutator_norm`, `lie_metric_drift`, `gram_drift_energy`, `gram_divergence_ratio`, `gram_divergence_proxy`

The new oriented sidecar is intentionally non-canonical for now:

- `pair_align_xs`, `pair_align_xl`, `pair_align_sl`
- `pair_angle_xs`, `pair_angle_xl`, `pair_angle_sl`
- `pair_quadrature_xs`, `pair_quadrature_xl`, `pair_quadrature_sl`
- `pair_phase_ratio_xs`, `pair_phase_ratio_xl`, `pair_phase_ratio_sl`
- `oriented_pair_angle_xs`, `oriented_pair_angle_xl`, `oriented_pair_angle_sl`
- `oriented_angle_contrast_xs_xl`
- `phase_coherence_residual_xs_xl_sl`
- `lie_orbit_xs`, `lie_orbit_xl`, `lie_orbit_sl`
- `lie_orbit_strain_ratio`, `lie_orbit_dominance`, `lie_commutator_coupling_ratio`
- `elevation_angle_x_over_sl`, `oriented_volume_xsl`
- `structural_branch_sign_xsl`, `negative_volume_excursion_xsl`
- `spectral_orbit_drift_energy`, `spectral_strain_drift_energy`
- `spectral_orbit_divergence_ratio`, `spectral_strain_divergence_ratio`
- `spectral_orbit_divergence_proxy`, `spectral_strain_divergence_proxy`

The reason to keep this sidecar separate is practical:

- existing LTST and Sleep-EDF pipelines continue to depend on the old stable columns,
- the new oriented quantities can be evaluated, plotted, and calibrated without breaking those contracts,
- if the sidecar proves useful, later passes can decide which pieces deserve promotion.

## Complexity Note
All new samplewise quantities stay `O(d)` because the only high-dimensional objects are the three column vectors in `Q_t`.

The high-dimensional work is still:

- dot products,
- norms,
- reduced QR on a `d x 3` bundle,
- projections into a 3D local reference basis.

Every matrix inversion, whitening, symmetrization, and angle construction beyond that lives in `3 x 3` or 3-vector space. So the new oriented or complex sidecar preserves the original lightweight design.

## Adaptive Metric & Fuzzy Membership Sidecar
Two ideas from UMAP's Section 2 and Section 5 map cleanly onto the existing `O(d)` streaming kernel. Both are wired as a new sidecar, preserving the stable contracts of the existing geometry outputs.

### Locally-Adaptive Metric Rescaling
UMAP normalizes local distances by the distance to the nearest neighbor, treating dense and sparse neighborhoods identically. We implement the causal, time-series analog: per-stream baseline normalization. 

During a configured burn-in window (e.g., the first 500 samples), the sidecar accumulates the running median and median absolute deviation (MAD) for each target feature. Once the burn-in period finishes, these baseline statistics are locked.

After burn-in, the feature is re-expressed as a normalized `z`-score relative to its own baseline spread:
\[
z_{f,t} = \frac{f_t - \text{median}(f_{\text{burn-in}})}{\text{MAD}(f_{\text{burn-in}}) + \varepsilon}
\]
This makes tracking deviations across different streams—some naturally loose, some naturally rigid—directly comparable. It is `O(1)` per beat after the baseline locks.

### Fuzzy Membership Soft Scoring
Instead of using hard thresholds over the distances, UMAP represents similarities as exponentially-decaying fuzzy membership strengths and combines evidence using the t-conorm rule. 

We adapt this into a streaming probability score using the computed `z`-scores:
\[
\mu_{f,t} = \exp\left(-\alpha \max(z_{f,t} - z_{\text{thresh}}, 0)\right)
\]
where `\alpha` controls the sharpness of the boundary (larger `\alpha` approaches a hard step function) and `z_{\text{thresh}}` is an activity threshold.

We combine these sequential independent evidences using a stable exponential moving average (EMA) acting as an continuous analog to UMAP's t-conorm combination:
\[
\text{evidence}_{f,t} = (1 - \beta) \, \text{evidence}_{f,t-1} + \beta \, (1 - \mu_{f,t})
\]
This creates a robust persistence measure that gracefully handles varying sequence properties without abruptly resetting, serving as the continuous probability of departure from the baseline manifold geometry.

### Branch-Conditioned Baseline
Some datasets live on multiple structural sheets with different baseline spreads. The reusable adaptive sidecar therefore optionally conditions the burn-in baseline on a branch feature instead of forcing all rows into one global baseline model.

In the current implementation, this branch-conditioned path is intended to pair naturally with the geometric branch induced by `oriented_volume_xsl`. During burn-in, the feature baseline is estimated separately on the positive and negative branches using the same median/MAD rule as the global baseline. After burn-in, each row uses the branch-specific baseline when that branch has enough burn-in support and otherwise falls back to the global baseline.

This leaves the original unconditioned adaptive outputs intact while adding branch-conditioned `z`, membership, and evidence streams for datasets where a single baseline manifold is too coarse.
