# Frozen Factor에서 diffusion proposal과 ESS 적응을 검증한 계산

2026-11-02. 기존 Factor의 추론 부분을 바꾸는 작은 operator 실험이다. **공개 GraphPFN 본체를 diffusion으로 개조한 결과가 아니다.** 기존 CE Factor3개를 고정하고, 각각을 위한8,476-parameter conditional denoiser3개만 새로 학습했다. Factor8,099개와 합치면 learned pipeline의 파라미터는16,575개다. 공개 Mask/PCFI CE checkpoint6개는 새 bank에서 재평가했다.

## 같은 potential의 조건부 분포

원래12node의 class spins와2feature-column orientation을 합쳐 `s∈{−1,+1}¹⁴`로 둔다. Role `h`는4가지이고, 주어진 prior-family에 따라 일부 role은 확률0이다. 기존 모델이 공개된 graph·관측값·context에서 만든 `J_h(D)`와 **role별 likelihood 상수까지** 유지한다.

\[
E_h(s;D)=\tfrac12s^\top J_h(D)s+c_h(D),\qquad
\pi(s,h\mid D)\propto m_h\exp E_h(s;D)\,\mathbf1[s_C=2Y_C-1].
\]

잠재 spins는 균일 base measure다. 이 식은 frozen Factor의 정상화된 **조건부 structured model**이다. 참 데이터의 SBM posterior나 연속 X의 정규화 likelihood를 복원했다고 부르지 않는다. Parameter uncertainty를 새로 추가한 것도 아니다. 정확한 reference는4×2¹⁴=65,536 states를 오프라인에서 열거한다. 실제 hidden feature orientation·query/helper 정답·실제 생성 parameter는 들어가지 않는다.

## 실제 학습한 diffusion

기존2048개 immutable train task 중 앞512개만 사용했다. 각 seed의 frozen model이 유도하는 정확한 joint distribution에서 task당64개 `(s,h)`를 뽑았다. 실제 sampled query label을 이용해 새 denoiser를 학습한 것은 아니다.

Forward kernel은 binary spins를 균일 분포로, role을 원래 prior mass로 reset한다. 누적 retention은 `a_t=1−t/8`이다. Context spins는 항상 고정한다. 작은 shared GNN이 noisy state·현재 role·noise time·모든 role의 couplings·constants·prior·context를 읽고 clean-spin probability와 clean-role distribution을 예측한다. Global spin reversal에 대해 spin 출력은 odd, role 출력은 even이 되게 만들었다. Node/helper/feature-column 타입을 구분하며 variable permutation equivariance를 개발 검사했다. 고정4role의 의미는 알고 있으므로 unseen relation vocabulary/schema 일반화라고 주장하지 않는다.

각 denoiser는1024updates×batch16, unknown-spin CE+role CE로 학습한다. 세 seed 모두 마지막 checkpoint만 사용했다. 원 Factor weights에는 gradient나 update가 없다. Reverse transition은 예측한 clean-state marginal에 대해 알려진 `q(z_(t−1)|z_t,z0)`를 평균하여 계산한다. 모든 허용 state에 양의 proposal 확률을 남긴다.

## 보정이 실제로 하는 일

SMC의 중간 target은

\[
\pi_\beta(s,h)\propto m_h\exp\{\beta E_h(s;D)\}\mathbf1[s_C=2Y_C-1]
\]

이다.64particles,8stages를 유지하고, fixed arm은 `β_l=l/8`을 사용한다. 증분 weight는 `exp{(β_new−β_old)E}`이며 마지막 이전 단계마다 systematic resampling한다.

Diffusion+SMC에서는 reverse-diffusion transition `Q_t(z'|z,D)`를 proposal로 사용한다. **양방향 proposal probability와 target ratio를 모두** 넣어 MH 보정한다.

\[
A(z,z')=\min\!\left(1,\frac{\pi_\beta(z')Q_t(z\mid z',D)}{\pi_\beta(z)Q_t(z'\mid z,D)}\right).
\]

따라서 implicit diffusion density를 계산할 수 있는 것처럼 취급하거나 `weight∝target`만 쓰지 않는다. 이 mutation kernel은 각 `π_β`를 보존하지만,64particles·8steps의 최종 결과가 exact posterior라는 뜻은 아니다. Ordinary SMC는 exact single-site Gibbs, 조건부 role update, independent-base global MH proposal을 사용한다. 개발 graph에서 정확도 결과를 보지 않고 latency만 측정해 비용 대조의 Gibbs sweeps를2로 정했다.1sweep 대조도 유지했다.

모든 sampler에 같은 Rao–Blackwell query estimate를 사용한다. 즉 particle별 원 factor의 single-site class conditional을 가중 평균한다. 단순 sample-frequency 예측도 별도 저장해 정보가 더 많은 readout을 한 arm에만 주지 않았다.

## Rényi를 어디에 썼나

\[
H_2(\bar w)=-\log\sum_k\bar w_k^2=\log\mathrm{ESS}.
\]

ESS arm은 같은 particles·stages·MH mutation을 유지하고 β increment만 바꾼다. 다음 단계의 incremental ESS가 `.8K` 이상이 되는 최대 β를 bisection으로 정하고,8번째 단계에서는 β=1을 강제한다. 최종 weight ESS·마지막 jump·role KL·고유 particle state·acceptance를 저장한다. **높은 ESS가 mode coverage나 낮은 predictive error를 보장하지 않는다.** Rényi Neural Processes의 variational objective를 구현한 실험은 아니다. CE를 Rényi로 바꾼 것도 아니다.

## 평가·판정 범위

새96graph,32/prior,context1/2/4 **총 라벨** × source3/6,disjoint만 평가하여576episodes다. Matched48graph가 primary이며 low-density24/high-signal24는 secondary다. Source/context는 겹치지 않지만 query 중에는 source인 노드도 있다. 자기 피처가 없는 query를 secondary로 따로 보고한다. Missing50/75%이므로99.5% missing 성과로 쓰지 않는다.

추론 primary는 exact frozen-Factor query marginal과의 KL이다. Diffusion+fixed가 diffusion-only와 비용 대조 SMC를 모두 넘어야 한다. 분류 gate는 비용 대조 SMC와 원 Factor-BP를 accuracy에서 모두 넘는 것이다. ESS의 증분도 fixed와 별도 비교한다.6conditions와 fixed3seed를 graph 안에서 평균한 뒤48graph bootstrap5000회를 사용한다. CI는 pointwise이며 미보정이다. Exact-Factor KL·Grid6 KL·accuracy·NLL/Brier·비용을 각각 보고한다. Density/signal OOD의 Grid6는 참 Bayes reference가 아니다.

원 source·protocol·기존 checkpoint를 final bank 전에 잠갔다.8,640 sampler episode 예측을 재실행해 bitwise 일치를 확인했고 기존 public6개도 각1회 재실행했다. 최종 particle 다양성은 audit에서 기존 `rb` 함수의 입력을 관찰하여 수집했다. RNG·kernel·source는 바꾸지 않았으며 replay probability가 원 파일과 정확히 같았다. 결과가 나쁜 뒤 추가한4-spin/4096particle audit는 계산 검증일 뿐, production 성능이나 rescue 실험이 아니다.

## 선행연구와 중복

- [DPM-SNC](https://papers.neurips.cc/paper_files/paper/2023/file/65d32185f73cbf4535449a792c63926f-Paper-Conference.pdf)와 [ReDiSC](https://arxiv.org/abs/2507.14484)는 이미 diffusion 기반 structured node prediction을 다룬다. 이번 코드는 두 논문을 재현한 것이 아니다.
- [Reverse Diffusion SMC](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6490ab6ef8eb7116c6c1889795d21748-Abstract-Conference.html)는 diffusion proposal과 SMC bias correction의 직접적인 선행연구다. 이번에는 known reverse transition을 tempered-target MH mutation으로 썼으며 해당 논문의 전체 알고리즘 재현이 아니다.
- [Rényi Neural Processes](https://proceedings.mlr.press/v267/wang25ew.html)는 latent prior/posterior objective의 오지정을 다룬다. 이번 ESS controller와 목적이 다르다. Diffusion·SMC·MH·ESS 조합만으로 novelty를 주장하지 않는다.

새 graph에서 가중치를 갱신하지 않는 learned inference라는 목표는 유지한다. 다만 작은 fixed schema와 기존 factor 함수족 안의 실험이며, 실제 heterograph나 새로운 feature law·큰 latent space의 성과는 없다.
