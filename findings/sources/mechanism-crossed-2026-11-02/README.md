# Mechanism-crossed GraphPFN pilot

2026-11-02. 사용자 요청에 따라 실제 학습하는 첫 mechanism-crossing 대조다.

모델은 `joint_imputation_pfn/common.py`의 공개 GraphPFN Mask Plain을 그대로 사용한다. 새로운 head, 대치기, auxiliary loss를 추가하지 않는다. Original transformer는 고정하고 graph adapters·feature encoder·label decoder·기존 mask/token 모듈을 학습한다.

네 arm은 Official Mask, Coupled Mask, Crossed Mask, Coupled PCFI다. 같은 seed에서 초기 trainable weights, 256개 logical episode의 순서, forward RNG, optimizer와 learning-rate schedule을 맞춘다. 새 seed는 11/12/13이다. Coupled PCFI는 입력만 PCFI이며 실제로 그 입력으로 학습한다.

Controlled family에서는 한 base graph마다 네 가지 feature field와 네 가지 label-rule parameter instance를 만든다. Coupled는 style=rule로 연결한다. Crossed는 Latin-square 방식으로 연결을 바꾸어 **두 주변분포의 concrete exposure counts를 보존하면서 joint dependence를 바꾼다.** Style×rule×mask의 64개 조합도 균등하다. 모델에는 style/rule ID를 주지 않는다. 하나의 graph로 한 번 update하므로 paired loss나 batch pairing의 효과가 아니다.

공식 prior 기준선과의 비교는 family 변경도 포함한다. 기존 공식 generator의 실패 원인이 coupling이었다는 인과 진단이 아니다. 그래프 의존 관측과 class-balanced context sampling이 있으므로 tabular iid 정리의 모든 가정을 충족하는 것으로 주장하지 않는다.

원래 seed banks를 재사용하지 않는다. 64개의 새 train graph와 32개의 새 test graph를 고정하며, 모든 모델에 같은 test episodes를 준다. 최대4개 label/class와 query32개를 사용한다. Zero-observation 사례나 약한 정보량의 사례를 성능을 보고 제거하지 않는다. 실제 heterograph와 unseen semantic schema는 별도다.

선행연구 PFNSeparability가 독립 SCM 원리를 이미 제안했다. 이 pilot은 그 원리를 그래프의 결측 설정에 적용해 새 학습의 독립 가치를 확인하는 대조이지, 원리의 발명이나 논문 novelty의 증명이 아니다.

실행 순서는 다음과 같다.

```text
python experiment.py prepare
python experiment.py preflight
python runtime/analyze.py --lock
python experiment.py run --seed 11
python experiment.py run --seed 12
python experiment.py run --seed 13
python runtime/analyze.py
```

`protocol.json`의 Gate A는 통제된 off-diagonal task에서 Coupled Mask/PCFI를 이기는지, Gate B는 새 공식-prior task에서 Official Mask를 이기는지 묻는다. NLL뿐 아니라 accuracy와 Brier, simple priors와 LP, 실제 피처 제거/셔플, 동일 checkpoint의 4-view random-fill 대조를 함께 보고한다. 두 gate를 통과하지 못하면 loss·epoch를 바꾸어 자동 구제하지 않는다.

관측 정보량 검사는 Gaussian/linear 부분 family의 **known-rule sign-only reference**다. 실제 RMS 입력이 보존하는 첫 관측 sign만 사용하고 magnitude는 고정 Monte Carlo, residual Gaussian은 quadrature로 적분한다. 일반 모델이 사용할 수 없는 true rule을 받으므로 실행 가능한 baseline 또는 full-context Bayes oracle로 해석하지 않는다.

추론은 마지막 checkpoint만 사용하며 target graph에서 weight update를 하지 않는다. Code·bank·공개 checkpoint hashes, source lock, finite gradient retry, optimizer 성공 update 수, 원래 frozen weights 보존과 저장 확률 재채점을 기록한다.
