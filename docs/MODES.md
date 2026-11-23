# Mathematical specification of the MIMIC presets

This page defines the four built-in `MIMIC(mode=...)` presets: `identity`,
`direct`, `factorised`, and `joint`. It describes the default preset
components. Explicit `encoder`, `decoder`, `generation_decode_mode`, and
other constructor arguments can override parts of a preset; see the
[implementation guide](IMPLEMENTATION.md#22-constructor).

The modes are generation and decoding presets. They do not change the
feature-wise prediction problem that supplies MIMIC's learned representations,
except that `identity` uses the preprocessed row itself as its representation.

## Notation and shared setup

Let the modelled data have <img src="assets/modes/equation-30.svg" alt="p" style="vertical-align: middle;"> columns, indexed by
<img src="assets/modes/equation-01.svg" alt="j\in\mathcal J=\{1,\ldots,p\}" style="vertical-align: middle;">. Column <img src="assets/modes/equation-31.svg" alt="j" style="vertical-align: middle;"> has task type <img src="assets/modes/equation-32.svg" alt="t_j" style="vertical-align: middle;">, either
regression or classification. Let <img src="assets/modes/equation-02.svg" alt="x_i=(x_{i1},\ldots,x_{ip})" style="vertical-align: middle;"> denote row <img src="assets/modes/equation-33.svg" alt="i" style="vertical-align: middle;">, and
let <img src="assets/modes/equation-03.svg" alt="\phi(x_i,m_i)" style="vertical-align: middle;"> be the globally preprocessed row, including encoded
values and missingness indicators. Here <img src="assets/modes/equation-34.svg" alt="m_i" style="vertical-align: middle;"> records which values are
missing. Columns declared as ignored are outside <img src="assets/modes/equation-04.svg" alt="\mathcal J" style="vertical-align: middle;">.

For the neural presets, the default encoder for target <img src="assets/modes/equation-35.svg" alt="j" style="vertical-align: middle;"> receives the
preprocessed context with target column <img src="assets/modes/equation-36.svg" alt="j" style="vertical-align: middle;"> excluded: <img src="assets/modes/equation-05.svg" alt="u_{ij}=\phi_{-j}(x_i,m_i),\qquad h_{ij}=E_j(u_{ij})\in\mathbb R^{d_j}." style="vertical-align: middle;">

The representation of a row is the concatenation of its feature blocks: <img src="assets/modes/equation-06.svg" alt="H_i=[h_{i1}\,\,\Vert\,\,h_{i2}\,\,\Vert\,\,\cdots\,\,\Vert\,\,h_{ip}] \in\mathbb R^{D},\qquad D=\sum_j d_j." style="vertical-align: middle;">

Each encoder is trained as a feature prediction task on rows where its target
is observed. The learned representation is then used to fit a target-specific
decoder. With explicitly configured shared numerical feature groups, encoder
parameters may be shared across those targets; each target still has its own
conditional representation block and target decoder.

Write <img src="assets/modes/equation-07.svg" alt="\ell_j" style="vertical-align: middle;"> for the task loss: squared error for a regression target
and cross-entropy for a classification target. A target-specific predictive
model is fitted using observed-target rows
<img src="assets/modes/equation-08.svg" alt="\mathcal I_j=\{i:x_{ij}\text{ is observed}\}" style="vertical-align: middle;">. In schematic form: <img src="assets/modes/equation-09.svg" alt="\min_{E_j,a_j}\; \frac{1}{|\mathcal I_j|} \sum_{i\in\mathcal I_j} \ell_j\!\left(a_j(u_{ij}),x_{ij}\right), \qquad \min_{D_j}\; \frac{1}{|\mathcal I_j|} \sum_{i\in\mathcal I_j} \ell_j\!\left(D_j(h_{ij}),x_{ij}\right)." style="vertical-align: middle;">

Here <img src="assets/modes/equation-37.svg" alt="a_j" style="vertical-align: middle;"> is the supervised output head used while learning the neural
encoder; <img src="assets/modes/equation-38.svg" alt="D_j" style="vertical-align: middle;"> is the fitted feature decoder. These equations specify the
training roles, not a claim that the non-convex neural optimisation reaches a
global minimum.

All non-identity presets use the same default neural encoder and
`NeuralConditionalSampler` decoder family. They differ in how a generated
embedding is decoded. Their default latent generation policy is displacement;
the policy can be replaced independently. The preset mapping is implemented
in [`_preset_components`](https://github.com/fabriziocosta/MIMIC/blob/master/src/mimic/mimic.py#L2122-L2138).

## Identity preset

For each modelled column <img src="assets/modes/equation-39.svg" alt="j" style="vertical-align: middle;">, the identity encoder receives the full
preprocessed row, including column <img src="assets/modes/equation-40.svg" alt="j" style="vertical-align: middle;">, and returns it unchanged. With one
feature block per modelled column, the concatenated representation is <img src="assets/modes/equation-10.svg" alt="h_{ij}=\phi(x_i,m_i)\quad\text{for every }j,\qquad H_i=[\phi(x_i,m_i)\,\Vert\,\cdots\,\Vert\,\phi(x_i,m_i)] \quad(p\text{ identical blocks})." style="vertical-align: middle;">

The identity decoder selects the coordinates for its target (for categorical
targets, the corresponding one-hot scores) and maps them back through the
target's output decoding. The repeated blocks multiply all pairwise squared
Euclidean distances by <img src="assets/modes/equation-41.svg" alt="p" style="vertical-align: middle;">, so they do not change nearest-neighbour ordering.
The preset has no learned feature-wise latent encoder or learned row decoder.
For generation it applies the selected latent-space policy directly in
preprocessed feature space, then decodes the resulting coordinates to the
original column scales and categories.

For continuous values this permits interpolation or displacement in the
preprocessed numeric coordinates. For categorical values, the decoded class is
the class associated with the largest interpolated score. This is a
feature-space baseline; it is not a conditional probability model.

With the default `categorical_displacement=True`, categorical columns use the
source triplet rule independently for each categorical feature: switch `A` to `C`
with probability `clip(lambda, 0, 1)` when that feature's `A` and `B` match, and
otherwise preserve `A` for that feature. Each matching feature gets its own
Bernoulli draw, so a mismatch in one feature does not prevent a switch in
another. Conditioned columns remain fixed. This post-decode rule avoids
arithmetic on category codes or one-hot blocks. Set
`categorical_displacement=False` to restore the legacy decoded category.

## Direct preset

For each generated embedding <img src="assets/modes/equation-11.svg" alt="H^*=(h^*_1,\ldots,h^*_p)" style="vertical-align: middle;">, direct
decoding applies each target-specific decoder to that target's own feature
block: <img src="assets/modes/equation-12.svg" alt="\hat x^*_j=D_j(h^*_j),\qquad j\in\mathcal J." style="vertical-align: middle;">

The implementation's neural regression decoder predicts the conditional
mixture mean <img src="assets/modes/equation-13.svg" alt="\hat x^*_j=\sum_{k=1}^{K}\pi_{jk}(h^*_j)\,\mu_{jk}(h^*_j)," style="vertical-align: middle;">

before reversing target scaling. For a categorical target it chooses the
highest-probability class: <img src="assets/modes/equation-14.svg" alt="\hat x^*_j=\arg\max_{c\in\mathcal C_j} p_j(c\mid h^*_j)." style="vertical-align: middle;">

This preset is deterministic at generation time: it returns point predictions
and does not draw values from the conditional samplers.

The target block <img src="assets/modes/equation-42.svg" alt="h_j" style="vertical-align: middle;"> is itself computed from the input with feature <img src="assets/modes/equation-43.svg" alt="j" style="vertical-align: middle;">
excluded. Therefore “direct” means direct decoding from the learned
target-specific block; it does not mean that the encoder was given <img src="assets/modes/equation-44.svg" alt="x_j" style="vertical-align: middle;">.

## Factorised preset

Factorised generation first makes the same deterministic initial decode as the
direct preset. It then performs three ordered passes over the modelled columns.
For each unconditioned column <img src="assets/modes/equation-45.svg" alt="j" style="vertical-align: middle;">, it forms the fixed context <img src="assets/modes/equation-15.svg" alt="H^*_{-j}=(h^*_1,\ldots,h^*_{j-1},h^*_{j+1},\ldots,h^*_p)" style="vertical-align: middle;">

and samples <img src="assets/modes/equation-16.svg" alt="x^{(r)}_j\sim q_j(\,\cdot\mid H^*_{-j}),\qquad r=1,2,3." style="vertical-align: middle;">

Conditioned columns are kept at their requested values. For a categorical
column, <img src="assets/modes/equation-46.svg" alt="q_j" style="vertical-align: middle;"> is the fitted conditional class-probability distribution. For
a numerical column, the default neural sampler represents <img src="assets/modes/equation-47.svg" alt="q_j" style="vertical-align: middle;"> as a
<img src="assets/modes/equation-48.svg" alt="K" style="vertical-align: middle;">-component Gaussian mixture: <img src="assets/modes/equation-17.svg" alt="q_j(y\mid H^*_{-j}) =\sum_{k=1}^{K}\pi_{jk}(H^*_{-j}) \,\mathcal N\!\left(y;\mu_{jk}(H^*_{-j}),\sigma^2_{jk}(H^*_{-j})\right)." style="vertical-align: middle;">

The sampler draws a component according to its mixture weight, then draws a
value from that component. The implementation supports component temperature
and noise scale; their defaults are both one.

The latent context <img src="assets/modes/equation-49.svg" alt="H^*" style="vertical-align: middle;"> stays fixed through all three passes. Later sampled
column values are not re-encoded and fed back into other conditional samplers.
Consequently these passes are not Gibbs sampling from a joint distribution,
and the conditionals <img src="assets/modes/equation-18.svg" alt="q_j(x_j\mid H^*_{-j})" style="vertical-align: middle;"> need not be compatible with any
single joint distribution over <img src="assets/modes/equation-50.svg" alt="x" style="vertical-align: middle;">. Factorised mode is a practical,
feature-wise stochastic decoder.

## Joint preset

Joint mode first computes a conditional-evidence block for every target. For
each <img src="assets/modes/equation-51.svg" alt="j" style="vertical-align: middle;">, it applies that target's conditional sampler to the other
embedding blocks: <img src="assets/modes/equation-52.svg" alt="e_j=c_j(H^*_{-j})." style="vertical-align: middle;">

For a categorical target, <img src="assets/modes/equation-53.svg" alt="e_j" style="vertical-align: middle;"> is its vector of predicted class
probabilities. For a numerical target, <img src="assets/modes/equation-54.svg" alt="e_j" style="vertical-align: middle;"> concatenates the predicted
mixture weights, component means, and component scales. The complete evidence
vector is <img src="assets/modes/equation-19.svg" alt="e^*=[e_1\,\Vert\,e_2\,\Vert\,\cdots\,\Vert\,e_p]." style="vertical-align: middle;">

A shared neural trunk <img src="assets/modes/equation-20.svg" alt="g_\omega" style="vertical-align: middle;"> maps <img src="assets/modes/equation-55.svg" alt="e^*" style="vertical-align: middle;"> to a shared hidden state, and
target-specific heads <img src="assets/modes/equation-21.svg" alt="r_{\omega,j}" style="vertical-align: middle;"> produce all column predictions: <img src="assets/modes/equation-22.svg" alt="s^*=g_\omega(e^*),\qquad \hat x^*_j=r_{\omega,j}(s^*)." style="vertical-align: middle;">

The row decoder is fitted only on complete modelled training rows
<img src="assets/modes/equation-23.svg" alt="\mathcal I_{\mathrm{complete}}=\{i:x_{ij}\text{ is observed for every }j\in\mathcal J\}" style="vertical-align: middle;">.
Its objective is the mean of the per-target losses: <img src="assets/modes/equation-24.svg" alt="\min_\omega\; \frac{1}{p}\sum_{j\in\mathcal J} \frac{1}{|\mathcal I_{\mathrm{complete}}|} \sum_{i\in\mathcal I_{\mathrm{complete}}} \ell_j\!\left(r_{\omega,j}(g_\omega(e_i)),x_{ij}\right)." style="vertical-align: middle;">

Regression heads use squared error and categorical heads use cross-entropy.
The final predictions are deterministic: regression outputs are inverse-scaled
and categorical outputs use the highest-scoring class.

“Joint” here means **one shared multi-output decoder with a common hidden
representation**. The implementation does not fit a joint density
<img src="assets/modes/equation-25.svg" alt="p(x_1,\ldots,x_p\mid e)" style="vertical-align: middle;">, and it does not sample correlated rows from such
a density. The target heads share a learned trunk, while the training loss is
the average of separate per-target losses.

## What is common, and what changes

| Preset | Representation used for generation | Decode rule | Stochastic output sampling |
| --- | --- | --- | --- |
| `identity` | Repeated preprocessed row blocks <img src="assets/modes/equation-26.svg" alt="H=[\phi(x,m)\,\Vert\,\cdots\,\Vert\,\phi(x,m)]" style="vertical-align: middle;"> | Identity coordinate decoding | No |
| `direct` | Concatenated target-excluded neural blocks <img src="assets/modes/equation-56.svg" alt="H" style="vertical-align: middle;"> | <img src="assets/modes/equation-57.svg" alt="D_j(h_j)" style="vertical-align: middle;">, one head per target | No |
| `factorised` | Same neural blocks <img src="assets/modes/equation-58.svg" alt="H" style="vertical-align: middle;"> | Initial direct decode, then <img src="assets/modes/equation-27.svg" alt="q_j(x_j\mid H_{-j})" style="vertical-align: middle;"> for three passes | Yes |
| `joint` | Same neural blocks <img src="assets/modes/equation-59.svg" alt="H" style="vertical-align: middle;"> | Shared decoder <img src="assets/modes/equation-60.svg" alt="G([c_j(H_{-j})]_j)" style="vertical-align: middle;"> | No |

The latent generation policy (for example, interpolation or displacement) is a
separate choice from the decoding preset. For the default displacement policy,
with anchor rows <img src="assets/modes/equation-61.svg" alt="A,B,C" style="vertical-align: middle;">, the generated embedding is <img src="assets/modes/equation-28.svg" alt="H^*=H(A)+\lambda\bigl(H(C)-H(B)\bigr)," style="vertical-align: middle;">

where <img src="assets/modes/equation-29.svg" alt="\lambda" style="vertical-align: middle;"> is drawn from the policy's configured interval. In the
default preset configuration the interval is <img src="assets/modes/equation-62.svg" alt="[0.25,0.75]" style="vertical-align: middle;">, and neighbours
are selected with the mutual-neighbour policy. See the [generation policy
definition](https://github.com/fabriziocosta/MIMIC/blob/master/src/mimic/policies.py#L8-L25)
and the [embedding-generation implementation](https://github.com/fabriziocosta/MIMIC/blob/master/src/mimic/mimic.py#L848-L875).

## Scope and overrides

These equations describe the built-in defaults. A user-supplied encoder or
decoder changes the functions and possibly the loss family. A supplied
`generation_decode_mode` can select a different decoder path from the
`mode` preset. Capacity changes the neural dimensions and optimisation
hyperparameters; it does not change the mathematical input/output structure
above. Explicitly shared numerical feature groups also change parameter sharing
for their encoders. See the
[constructor and preset mapping](IMPLEMENTATION.md#22-constructor) for these
configuration rules.

Implementation references: [feature-wise fitting and decoding](https://github.com/fabriziocosta/MIMIC/blob/master/src/mimic/mimic.py#L640-L661),
[direct and factorised decoding](https://github.com/fabriziocosta/MIMIC/blob/master/src/mimic/mimic.py#L1378-L1410),
[factorised passes](https://github.com/fabriziocosta/MIMIC/blob/master/src/mimic/mimic.py#L1477-L1525),
[joint decoder fitting](https://github.com/fabriziocosta/MIMIC/blob/master/src/mimic/mimic.py#L1412-L1468), and
[neural conditional and joint heads](https://github.com/fabriziocosta/MIMIC/blob/master/src/mimic/decoders.py#L443-L465).
