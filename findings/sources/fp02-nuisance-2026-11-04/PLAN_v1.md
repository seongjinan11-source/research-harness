# FP02: class volume와 degree exposure를 갱신하는 Factor

현재 목표는 실제 homophilic graph의 missing-feature zero-shot 예측이다. Heterogeneous/heterophilic 범용성이나 새 likelihood 원리를 이번 pilot의 기여로 주장하지 않는다. FP01의 final과 역사적 코드는 변경하지 않는다. 이 계획은 독립 synthesis 세 리뷰를 반영하기 전 초안이며 본격 구현/학습은 세 검토 뒤 시작한다.

## 근거에서 계산 변경으로

직접 확인한 것은 old nonedge 항의 손해다. Physics는 feature 없는 상태에서도 삭제로 약46→86% 회복하고 CiteSeer는 삭제하면 손상된다. 따라서 일괄 삭제도 해법이 아니다. Degree/skew만이 고유 원인이라고 확인한 것은 아니지만, 현재 score가 큰 class에 O(n_c²)의 벌점을 주고 positive-edge strength도 약한 상태임은 확인했다. 반면 FP01의 feature-only 변경은 주로 harm reduction이었다. 다음 변경은 graph likelihood와 class mass의 추론이며 새 posterior head나 추가 loss가 아니다.

## 실제 operator

관측 degree d_i, m=sum(d)/2에서 e_ij=d_i d_j/(2m)를 고정된 configuration offset으로 둔다. 이 offset은 그래프에서 추정한 exposure이며 정확한 degree-conditional likelihood나 unknown edge observation mask를 안다고 주장하지 않는다. Isolated node는 exposure0이다. 모든 관측 adjacency0은 현 closed-world score에서 nonedge로 남긴다.

Class-pair affinity W_ab=W_ba>0에는 Gamma prior를 둔다. Init mean은 homophilic diagonal1+(K−1)eta, off-diagonal1−eta, eta=.6; shared shape kappa=5. Initial node population은 uniform, Dirichlet alpha=1이다. Neural target weight 업데이트는 하지 않는다.

유한 추론 중 L_ab=E[log W_ab]=digamma(shape_ab)−log(rate_ab), R_ab=E[W_ab]=shape_ab/rate_ab를 구분한다. Edge factor score는

    log psi_ij(a,b) = A_ij L_ab − e_ij R_ab.

관측 edge에는 full Poisson score를 쓰는 cavity BP를 수행하고, 비이웃의 음수 항은 nonneighbor degree-weighted class volume으로 근사한다. Field는 self와 neighbor volume을 제외하며 edge 음수항을 두 번 세지 않는다. 모든 prior/offset은 class/node permutation에 맞게 변한다. Graph BP24회, damping.5, 6회마다 affinity와 population을 갱신하는4outer schedule로 시작한다.

Gamma affinity의 sufficient statistics는 edge soft-count와 degree-weighted pair exposure다. Exposure는 nonedge의 node-product 근사와 edge pair belief의 보정으로 구성한다. Unordered pair/diagonal convention을 수식/테스트로 고정한다. Population은 unlabeled node beliefs의 **node count**로만 갱신하고, class당 선택한 context count를 모집단 빈도로 쓰지 않는다. Exposure는 context를 포함한 **모든 node의 degree-weighted volume**을 사용한다. 이 hybrid BP/mean-field update는 명시한 근사 연산이며 exact joint Bayes나 수렴을 주장하지 않는다.

Feature 경로는 기존 class-gated centroid factor를 재사용한다. 첫 operator test에서는 FP01 seed11 C_global의3계수를 고정하고 source→centroid→query 메시지를 새 graph BP 안에서 함께 갱신한다. Source/context는 분리되고 query truth는 평가에만 사용한다. Class-population/affinity 갱신과 feature effect를 분리하기 위해 none/observed/shuffle을 모두 보고한다.

## 단계와 비교

Stage0는 numerical correctness와 기존 개발 graph의 실제 수정 평가다. 새 학습 없이 rep0/seed11의 Cora/CiteSeer/PubMed/CS/Physics × none/95만 사용한다. 비교는 같은 C_global feature factor를 둔 old-full, old-edge, DC-fixed-affinity/uniform-population, DC-update다. 두 LP와 기존 PCFI/FISF→public PFN은 저장된 동일 episode 예측을 재사용한다. 이10개는 개발 검증이고 final이 아니다. 결과를 보고 eta/shape/iteration을 조정하지 않는다. Implementation bug만 근거를 남기고 고친다.

세 검토와 numerical PASS 이후 Stage0를 실제로 돌린다. 이걸 또 진단 문서만 남기는 단계로 끝내지 않는다. 정상 operator가 하나의 고정 규칙으로 Physics/CS 손상을 줄이고 CiteSeer full의 장점을 보존하는지 확인한다. Graph-only 회복을 새 feature 추론 성공으로 부르지 않는다.

Stage1의 학습은 위 결과와 리뷰 뒤 별도 명세를 잠근다. Original prior와 degree/skew를 포함한 prior에서 같은 Factor를 각3seed·256updates로 실제 학습하는6fits를 첫 범위로 한다. 고전 DC update와 normalized/harmonic/restart LP를 먼저 포함하고, 별도 neural affinity/posterior head를 붙이지 않는다. 동일한 graph/mask/context/base task와 노출 budget을 유지한다. 새 final synthetic bank는 source lock 뒤 생성하며 기존 실5graph는 계속 development다. 이 Stage의 결과 없이 synthetic-pretrained 새 PFN의 성공이라고 쓰지 않는다.

## 필수 수치 검사와 비용

- Tiny heterogeneous graph에서 dense full Poisson factor·exact enumeration·dense BP를 독립 비교한다. Sparse nonedge field의 별도 근사오차는 기록하고 exact 일치를 요구하지 않는다.
- Edge-only/tree cavity oracle, Gamma logmean vs meanlog, unordered diagonal/count/exposure, external field self/neighbor exclusion, no-double-counting을 검증한다.
- Class/node/source-row permutation, context clamps, hidden NaN poison, query label non-input, source0/1 class-exchangeability, old-checkpoint/hash 보호를 검증한다.
- All-isolated/empty-edge graph는 별도 정의로 안정적인 population/feature prediction을 내며 target label을 요구하지 않는다.
- Stage0는 단일 GPU worker, forward만, no optimizer/real-label update. 원래 checkpoint와 old bank는 read-only다. No outcome-based variant selection or real independent-CI.

Novelty의 경계: DC-SBM, conditional pseudo-likelihood, Gamma/Dirichlet updates, BP와 neural factor 학습은 기존이다. 이번은 확인한 실패를 고치는 구체적인 method revision이다. 이후 방법 논문 주장은 고정된 통계 추론보다 학습한 factor가 필요한 조건과 재학습 없는 전이의 실증을 확보해야 한다. 고전 추론기만으로 해결되면 그것도 결과로 인정하고 neural 필요성을 만들기 위해 약한 baseline을 두지 않는다.
