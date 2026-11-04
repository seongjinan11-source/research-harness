# FP04 shared model implementation

2026-11-04. 소유 구현: `observed_cell_cycle/model.py`. 근거: PLAN_v2/protocol v2와 root가 승인한 7 node statistics/3 column statistics. 이 구현 단계에서는 작은 CPU 합성 입력만 사용했다. GPU·학습·실제 bank 평가를 실행하지 않았다.

## API와 책임

```python
model = ObservedCellPFN(width=32, rank=4, rounds=4)
logits = model(batch, arm='observed', class_chunk=1,
               feature_chunk=256, sparse=True, disable_columns=False)
```

`batch`: `x[N,F]`, `observed[N,F]` bool, `pcfi[N,F]`, `src[E]`, `dst[E]`, `context[N]` long(-1 unknown), `classes` int. Canonical visible-value key는 **x**다. 그래프는 self edge 없는 directed symmetric edge list이며 incoming degree와 incoming neighbor mean을 사용한다. logits는 `[N,K]`; known context도 최종 hard clamp하지 않는다. 모델은 class chunk별 softmax를 하지 않는다. Runner가 모든 class logits를 받은 후 CE/최종 softmax를 한 번 적용한다.

모델 device/dtype에 맞춰 floating input을 변환하므로 `.float()`/`.double()` CPU를 모두 지원한다. 클래스·열 chunk는 결과를 바꾸지 않는 실행 옵션이며, dense arm은 `sparse` 인자를 무시한다. Model은 batch의 query truth나 evaluator metadata를 읽지 않는다.

Observed-only RMS scaling, [-10,10] visible clipping, intervention, literal PCFI 계산은 **외부 전처리 책임**이다. 모델은 통계 계산 전에 다시 `where(observed,x,0)`를 적용한다. PCFI arm에서도 observed 위치는 이 x로 clamp하며, 숨긴 위치는 실제 전달된 PCFI 값을 쓴다. Raw PCFI 결과 캐시는 수정하지 않는다. Value MLP의 입력 표현에만 아래 공통 clipping을 적용한다. 매 intervention마다 외부 scaling/PCFI가 다시 계산되어야 한다.

## 고정 통계와 초기 상태

아래 mean/second moment는 true observed cell만 사용하고, count 0이면 numerator 0/denominator 1로 0이다. second moment는 분산이 아닌 `mean(x²)`다. 통계는 PCFI 값으로 계산하지 않는다. `log1p`의 기준은 자연로그다.

Column statistics `[F,3]`:

1. `tanh(observed column mean)`
2. `tanh(observed column second moment)`
3. `log1p(observed column count)/log1p(max(N,1))`

Node statistics `[K,N,7]`:

1. `tanh(observed row mean)`
2. `tanh(observed row second moment)`
3. `log1p(observed row count)/log1p(max(F,1))`
4. `observed row count/max(F,1)`
5. `log1p(incoming degree)/log1p(max(N,1))`
6. `context count for class c/max(total context count,1)`
7. `total context count/N`

빈 열과 observed zero-only 열은 count feature로 구분된다. F=0/N>=1도 안전하게 처리한다. 정적 class support는 label count만 보고 value-independent하다. 동적 global node/context pooling은 없다.

공유 embedding의 seed index는 unknown=0, this-class context=1, other-class context=2다. 초기 상태는 `h_ci = seed_embedding(seed_ci) + node_mlp(node_stats_ci)`다. node_mlp는 `7→width→width`, GELU MLP다. Class channel 간 상태·메시지 의존성은 없다. Supplied K를 유지하며 context에 등장하지 않는 class도 출력한다.

## 값 basis, 수식, round 순서

입력 값 v는 observed/mask에서 sanitized x, PCFI에서 observed-clamped PCFI다. **모든 arm에 공통으로** value basis 직전 `v=clamp(v,-10,10)`을 적용한다. 값 basis 입력은 정확히 다음 7개다:

`[v/10, |v|/10, v²/100, M_if, col_stats_f[0], col_stats_f[1], col_stats_f[2]]`.

`value_mlp: 7→width→rank`는 hidden GELU와 final tanh를 사용한다. 따라서 basis `phi_ifr`는 signed이며 각 원소가 [-1,1]에 있다. basis는 class/round별로 바뀌지 않는다. Learned node/column ID는 없다.

공통 representation clipping은 2026-11-04 최초 CPU smoke 이후, data preparation·성능 학습·실제 점수 관찰 **이전** root의 명시적 승인으로 추가한 prelock numeric 수정이다. Native feature 수에서 literal PCFI 보정값의 크기가 커질 때 `v²/100`의 무제한 범위와 activation-range OOD를 피하려는 변경이다. Visible 값은 외부에서 이미 같은 범위로 clip된다. PCFI 알고리즘·raw cache는 보존되며, completion의 절댓값이 10을 넘는 정도에 대한 정보는 model representation에서 소실된다. Raw PCFI의 maxabs/표현 clip 비율은 전처리·runner provenance에 기록해야 하며, clipping 없는 PCFI baseline과 같은 입력이라는 주장은 하지 않는다. `protocol.model.represented_value_clip=10`과 함께 잠근다.

Inclusion은 observed에서 `A=M`, mask/PCFI에서 `A=1`이다. `n_f=sum_i A_if`, `n_i=sum_f A_if`는 **양의 포함 개수**다. Signed basis 합이나 현재 feature chunk의 count로 나누지 않는다. 관측값 0인 셀도 M=1이면 포함한다.

네 shared round에서 다음 순서를 사용한다. residual은 graph/read 모두 .25다.

```
g_ci = mean_{j: j→i} h_cj                   # isolated node = 0
h_ci = graph_norm(h_ci + .25 graph_mlp([h_ci,g_ci]))
u_cf = sum_i sum_r A_if phi_ifr P_r(h_ci) / max(n_f,1)
z_cf = column_norm(column_mlp([u_cf,col_stats_f]))
v_ci = sum_f sum_r A_if phi_ifr V_r(z_cf) / max(n_i,1)
h_ci = read_norm(h_ci + .25 read_mlp([h_ci,v_ci]))
```

`P_r,V_r`는 rank개의 별도 width→width bias-free Linear다. graph/read MLP는 `2width→width→width`, column MLP는 `width+3→width→width`이고 hidden GELU다. 세 norm은 별도 LayerNorm이며 round 전체에서 공유한다. 빈 column의 u와 빈 row의 v는 0이다. Column MLP의 bias가 빈 column의 z를 만들 수는 있으나 observed arm의 A=0인 셀로 전달되지 않는다. Mask/PCFI는 missing 위치도 의도대로 포함하며 observed bit를 통해 그 기여를 학습해서 억제할 수 있다.

공유 head는 `width→width→1`, hidden GELU다. 마지막 scalar bias는 class softmax에서 상쇄되고 CE로 식별할 수 없어 도입하지 않았다. 기본 모델은 **19,748 trainable parameters**이며 arm마다 state_dict 이름·모양·초기 state를 바꾸지 않는다.

`disable_columns=True`는 v를 정확히 0으로 고정한다. Graph update와 local read self-MLP는 그대로 유지한다. 동적 global pool이 없어 disconnected components 간 source 값의 미분 경로를 제거하는 검사에 쓸 수 있다. 이는 학습 성공이나 feature utility의 증거는 아니다.

## Sparse/dense와 native F

Observed sparse 구현은 M의 nonzero cell에 대해서만 basis와 양방향 scatter를 계산한다. `sparse=False`는 모든 위치에서 basis를 계산한 뒤 M을 곱하고 dense contraction을 사용한다. mask/PCFI는 basis를 hard M-gate하지 않는다. 두 구현은 위와 같은 수식이다.

Feature chunks의 basis만 저장하며, 각 round에서 column state를 chunk별로 만든 뒤 node read 합에 누적한다. Scalar rank basis와 sparse indices를 제외하고 cell별 width-dimensional hidden state를 보관하지 않는다. Dense contractions는 einsum이며 N²/F² message tensor를 만들지 않는다. Native F에는 학습된 column 수에 종속된 parameter나 reshape가 없다. Class chunk 사이에는 전체-axis counts와 동일한 basis를 재사용하며 마지막에 logits를 concatenate한다. Chunk/연산 순서에 따른 floating-point roundoff는 허용된다. 학습은 같은 수식이어도 native full-cell sparse 경로의 좌표 저장·scatter가 dense보다 느릴 수 있으므로 성능 우위는 주장하지 않는다.

## 실행한 작은 CPU 확인

Python 환경은 기존 windows_graphpfn_env이며 sklearn을 torch보다 먼저 import했다. Torch CPU threads=1, 모델 기본 shape, 메모리에서 생성한 작은 입력만 사용했다.

- float64, N=7/F=5/K=3: observed sparse/dense output max abs error `8.33e-17`, parameter gradient max error `8.88e-16`. Gradient 비교에는 squared-score/class-score probe loss를 사용했다. 모든 parameter gradient가 존재하고 finite했다.
- full M=1과 clamped PCFI=x: observed/mask/PCFI output 차이 `1.67e-16` 이하.
- source 0, F=0과 hidden x의 NaN poison: finite 출력 및 poison invariance PASS.
- balanced two-component bridge: source 값→query class-contrast 미분 column on `8.87e-8`, off 정확히 0. 이 값의 크기를 predictive utility로 해석하지 않는다.
- float32, N=8/F=6/K=3 CE backward: 모든 parameter gradient 존재·finite. Node permutation `1.19e-7`, column permutation `5.96e-8`, class permutation `1.19e-7`, class/feature chunk 변경 `1.19e-7`, sparse/dense `7.45e-8` 이하.
- 공통 representation clip 추가 후 float32 `v=±1e20` basis 입력과 gradient가 finite이며, extreme PCFI forward도 finite하고 caller의 raw PCFI tensor를 바꾸지 않는 것을 확인했다. 이 작은 CPU 검사는 clip 전 원본 raw PCFI의 품질·실제 발생 빈도를 평가하지 않는다. 수정 후 model SHA256: `112CFDBD73C115C9BCCBB64FE99EE73897DD8C266A0C0B1FF5E786C3F58854AC`.

위는 구현 담당자의 smoke다. 독립 checks의 loop contraction·strict leakage·추가 bridge/oracle·resource 검증과 source lock을 대체하지 않는다. GPU/native resource와 실제 학습·평가는 root 담당이다.
