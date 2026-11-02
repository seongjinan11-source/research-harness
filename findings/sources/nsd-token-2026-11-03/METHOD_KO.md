# NSD token GraphPFN — 실행 전 정의

2026-11-03. 사용자는 synthetic 학습·실제 가중치 고정을 유지하면서, 관측 입력에서 바로 시작하고 표현과 restriction map을 함께 갱신하는 NSD 기반 PFN 실행을 승인했다. Heterogeneous graph는 제외한다.

## 계산

관측 numeric 값과 명시적 mask, context label을 기존 GraphPFN encoder에 넣는다. PCFI warmup은 없다. 각192차원 hidden feature/label token을2×96(stalk×channel)으로 해석한다. PFN0/3/6/9층 직전에 다음 연산을 적용한다.

`F_i,e=R(pi*tanh(Phi(u_i,u_j)))`, `F_j,e=R(pi*tanh(Phi(u_j,u_i)))`.

`R`은2D회전이고 `u_i`는현재 feature-token평균과label token을공유384→32projection으로요약한것과관측비율/context표시/confidence/logdegree다. Endpoint순서를보존하므로두restriction은같을필요가없다. SharedPhi는72→32→1이다. `T_ij=F_i^T F_j`, `T_ji=T_ij^T`다. **Global Laplacian은대칭PSD지만각transport block이대칭일필요는없다.** 이모델은SO(2)subfamily이며일반O(d)전체나원본NSD전체아키텍처재현이아니다.

각undirectededge의에너지는 `w_e ||F_i h_i-F_j h_j||²`다. `w=.25+.75sqrt(c_i*c_j)`, `c=.9^shortest_source_distance`이고source가없거나도달불가면0이다. 양수floor는feature가없어도context-label정보의전파경로를끊지않는다. Confidence는calibratedprobability가아니다. Degree+1의augmentednormalization으로`Delta`를구성한다. Selfedge는edge에너지에넣지않는다.

`H_new = H - .5sigmoid(tau_l) ELU(Delta_F (W_left,l H W_right,l))`.

좌우mixing은각각2×2/96×96이고초기값I,step초기값.1이다. Geometryencoder는4단계공유,stage별mixing/step은각각학습한다. Dynamic은현재hiddenstate에서맵을매번계산하고static은첫맵을detach없이재사용한다. Identity는같은연산/mixing을쓰며맵만I다. 마지막까지두stalk좌표를PFN에전달하고numericcompletion으로되돌리지않는다.

관측numeric입력은불변이다. NSD후 fullyobservedfeature-pairtoken과contextlabeltoken은그단계의직전hiddenstate로clamp한다. 이후일반Transformer층은latent표현을갱신할수있다. 이는관측numeric값이매층동일latent좌표라는주장이아니다. 모든arm은원래GraphPFNgraphadapter와공통TokenControl도유지한다.

## 비교와 학습

Mask,PCFI100,identity,static,dynamic을3seeds로모두새로학습한다. 동일public초기화·synthetictaskorder·1024successfulqueryCEupdates다. 기존Transformer는고정하며기존정책의adapter/encoder/decoder·mask/TokenControl과NSD가학습대상이다. Static/dynamic은정확히같은trainable구조다. Plainarm의미사용NSDcontainer까지개수가같다고유효용량이같다는뜻은아니다.

이전유효64trainbasegraph와6관측설정의visibleinput/context/query배열을그대로사용하되PCFIcache는원래100step+channelcorrection을명시적으로완료한다. Sources2/8의2설정을추가해64×8=512episode를2pass사용한다. 실제99.5%zero-source는유지한다. Sources조건은geometry에정보가있는별도secondary이며primary를대체하지않는다. 이전384updates수치와직접구조효과를빼지않고이번새동일예산arm끼리비교한다.

별도64synthetictestgraph는학습/기존test와다른seed다. Generatorfamily는이전과같으므로unseenfamily나semantic schema라고부르지않는다. 실제4graph/3반복은기존development입력을재사용하며독립새realbenchmark로부르지않는다. 기존trainmask에서class당최대4context만쓰고validation/query정답으로학습·선택하지않는다. RP16관측값의RMS정규화라는입력한계도유지한다.

PCFI100checkpoint에는원본피처PCFI→RP16강한대조도평가한다. Publicsingleforward와고정positive/negativeLP는동일입력의기존확률artifact를hash확인후재사용한다. NegativeLP는CLP/CPGNN재현이아니다.

## 판정

Primary는90/99.5% uniform/cluster4조건의결측queryaccuracy다. Graph내조건과3fixedseeds를먼저평균하고64basegraphpairedbootstrap5000회를쓴다. Dynamic이PCFI100/identity/static대비CI하한>0,homoguard−2pp를통과해야한다. 실제heteromean에서위대조와nativePCFI를넘고homo손해2pp이내여야확대한다. NLL/Brier/fullfeature/secondarysource수와고정negativeLPfloor를별도보고한다. 실패후loss/head/updatebudget을사후구제하지않는다.

## 원문과 범위

[Neural Sheaf Diffusion §5](https://arxiv.org/html/2202.04579),[공식DiscreteBundleSheafDiffusion](https://github.com/twitter-research/neural-sheaf-diffusion/blob/master/models/disc_models.py),[공식Laplacian builder](https://github.com/twitter-research/neural-sheaf-diffusion/blob/master/models/laplacian_builders.py)의endpointrestriction·orthogonalconnection·augmentednormalization·nonlinearresidual구조를참고했다. Nativeextension의존성대신직접incidence연산을작성했고densePSD/gradient/identity검증을한다. 원본NSD는targetgraph학습을하며여기서는syntheticonly이므로그학습성과를우리전이성과로인용하지않는다. 이조합자체의novelty는미확정이다.
