# 성능표 확인 전 추가한 native-feature 추론 진단

수정 bank의 mask/PCFI 비교군이 학습 중인 시점이며 learned sheaf 모델의 결과는 아직 없다. Primary와 학습 경로는 변경하지 않는다.

RP16 이전의 raw physical feature에서 같은 synthetic-trained shared router를 사용해 공간 전파와 PCFI의 global inter-channel correlation을 수행한 뒤 RP16으로 PFN에 전달하는 추가 경로를 평가한다. 원래PCFI의high-dimensional channel correlation을보존하는것이목적이다. 새target optimizer·정답법칙·hyperparameter 선택은없다.

비용을제한해4dataset의첫context/mask repetition,4missing조건,seed11에서pcfi200/weighted/signed/sheaf만평가한다. Native-PCFI100비교는이미같은입력bank와seed11에서저장한다. 이것을전체3seed·3repetition의독립최종검증으로바꾸지않고사전primary대신쓰지도않는다.

Physical channel별spatial update를128개씩계산하고모든channel을합친뒤**한번의전체channel correlation**을적용한다. Chunk별correlation은원PCFI와다른연산이므로사용하지않는다. 작은dev입력에서chunking이원래연산과일치하는지검사한다.

학습된router는dimension-shared지만native rawbinary값과RP16/RMS공간사이의분포변화가있다. 따라서추가경로의실패를raw-feature-space에서실제로학습한Sheaf-PCFI의실패로해석하지않는다.
