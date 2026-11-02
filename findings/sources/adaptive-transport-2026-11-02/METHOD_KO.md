# 정답 법칙 없이 적응하는 부호 전달 대조

이번에는 기존의 known-law reference에 주었던 참 feature-law g와 label-law t를 추론 입력에서 제거한다. 같은 bipartite Gaussian-signal family의 새 그래프64개를 사용하며, 기존 matched192-update Mask/PCFI Plain checkpoint6개는 고정한다. 새 모델이나 optimizer 학습은 없다. CPGNN/CLP의 재현이나 새 PFN을 학습한 실험으로 부르지 않는다.

피처 전달 g∈{동일,반전}, 라벨 관계 t∈{동일,반전}를 분리한다. 상대 orientation σ∈{−1,+1}와 amplitude∈{.5,1,2}까지 합쳐24개의 working-model 가설을 사용한다. Adjacency에서 BFS bipartition b를 구한다. 원래 graph의 저주파 Gaussian covariance를 구성하고, **관측된 행만으로 계산한 uncentered leading singular score**를 연속 관측으로 사용한다. Context labels는 Gaussian 잠재값의 부호 제약으로 들어간다.

각 가설의 feature density와 context-label orthant probability를 같은 모델에서 계산해 joint weight와 query sign probability를 구한다. Context 적분은 scrambled Sobol1,024점과 antithetic1,024점을 이용한 sequential truncated-Gaussian importance sampling이다. 이 score 압축·RMS 처리·median threshold·balanced context 선택은 원 generator와 정확히 같은 확률모형이 아니다. 따라서 원 데이터의 exact Bayes posterior나 calibration을 보장한다고 주장하지 않는다.

Feature-only q(g)는 labels/context를 읽지 않는 별도 함수다. Joint q(g)는 t·σ·amplitude를 주변화한다. 두 경우 모두 원 PCFI와 parity-transformed PCFI 두 completion에 같은 frozen PFN을 실행하고 q(g)로 class probabilities를 평균한다. 50:50 두-completion 평균이 동일한 두 forward 예산 대조다. 이는 LP와 PFN 중 정답 predictor를 고르는 selector가 아니라, 같은 predictor의 입력 가설에 대한 추론 대조다. 이 구분만으로 novelty가 생기는 것은 아니다.

강한 classical 대조로 같은 Gaussian/probit working model이 직접 출력하는 query probability를 사용한다. 피처를 제거한 Gaussian model과, label-only q(t)로 양의/signed LP를 평균하는 adaptive LP도 넣는다. True-g completion/true-t LP는 평가용 privileged reference로만 남긴다.

Primary는 sources8·context4/class, label/feature 네 law 조합의 missing-query accuracy 평균이다. 기존3 training seeds와 네 조합을 base graph 안에서 평균하고 새64 graphs를 paired bootstrap5,000회 한다. Joint-adaptive PFN이 fixed50:50와 feature-only adaptive를 모두 CI 하한>0으로 넘어야 한다. 이후 Mask·고정 전파·adaptive LP·direct Gaussian과 피처 제거까지 비교한다. 피처 제거 전후는 **원8-source의 동일 missing-query 집합**으로 제한한다.

0/1 source에서는 임의의 column orientation을 가진 단일 관측행만으로 g를 구분할 정보가 없다. q(g)=.5의 대칭성을 수치적으로 확인한다. Reference의 법칙 정보를 신경망이 임의로 복원할 수 있다고 가정하지 않는다. Label/feature law 정답은 적응 확률의 사후 평가에만 사용한다.

개발 graph2개에서 Gaussian 소형 해석해, Sobol 해상도, class/global gauge 및 node permutation, strict feature-only label blindness, 원 checkpoint replay를 검사한다. Source/config/checkpoint를 잠근 뒤 최종 bank를 생성한다. 기존 bank나 checkpoint를 수정하거나 final 결과에 맞춰 noise/amplitude/kernel·particle 수를 조정하지 않는다. 새 family 학습이나 실제 graph/schema 전이는 별도 문제다.
