# Deployment-Path-Conditioned Backdoors on Heterogeneous Edge NPUs

**타깃 venue**: IEEE S&P 2027 / USENIX Security 2027
**관련 과제**: RS-2024-00339187 (on-device robot AI compression), RS-2026-25507326 (distillation prevention & tracing)
**버전**: v11.1(2026-07-24 실측 반영) — §4 공격 설계를 fusion 동작 분석 위에서 정식화(신호원=하드웨어 편차, 위치=fusion 블록 경계, 인코딩=outlier 존재성, 트리거=편차 부분공간 정렬, 학습=실측 편차 캘리브레이션). P0.5(§3.2)·초기 P3 시도(§4.8) 실측 결과와 그로부터 도출된 트리거 최적화 교정을 반영. 진행 중인 구현(트리거 최적화 재설계)의 상세 계획은 `chain_survival/CURRENT_PLAN.md`.

---

## Abstract

딥러닝 모델은 개발 단계에서 GPU로 검증된 뒤, 실제로는 전력 효율이 높은 고정기능 신경망 가속기(NPU; NVIDIA DLA, Qualcomm Hexagon, Mobilint 등)에서 실행되는 경우가 많다. 이 두 하드웨어는 성능뿐 아니라 양자화 파이프라인 자체가 다르다: systolic-array 기반 NPU는 coarse-grained·static 양자화를 강제하는 반면 GPU는 유연한 explicit 양자화를 지원한다. 그 결과 **동일한 INT8 모델이라도 GPU와 NPU는 서로 다른 수치 결과를 낸다.**

본 연구는 이 경로 간 편차를 새로운 백도어 조건으로 삼는 **Deployment-Path-Conditioned Backdoor (DPCB)** 를 제안한다. DPCB는 GPU 검증 경로에서는 무해하게 동작하여 백도어 탐지를 통과하지만, NPU 배포 경로에서만 트리거에 반응한다. 우리는 (1) 이 편차의 존재·원인·아키텍처 의존성을 실측하고, (2) GPU 기준으로 설계된 기존 quantization backdoor 공격·방어가 NPU 배포에서 재현·탐지되지 않음을 systematic하게 입증하며, (3) 실제 다단계 배포 파이프라인(ONNX→TensorRT→NPU)을 통과하는 DPCB를 설계하고, (4) 경로 비교에 기반한 완화기법을 제안한다. 평가는 편차 메커니즘을 정밀 규명하는 통제 환경(NVIDIA DLA)과 최신 모델의 실배포 대표성을 확인하는 상용 NPU(Mobilint)의 두 계층으로 구성된다. 본 위협은 systolic-array 기반 고정기능 NPU에 한정되며 범용 모바일 프로세서에는 적용되지 않는다.

---

## 1. Introduction

### 1.1 문제 제기

백도어 방어의 암묵적 전제는 "검증 시점의 모델과 배포 시점의 모델이 같은 함수"라는 것이다. Quantization Blindspots(2025)는 이 전제의 첫 균열을 지적했다 — 방어는 FP32에서 평가되지만 배포는 INT8이며, 이 정밀도 간극이 배포 전 스캔에 의존하는 파이프라인을 위협한다방어 평가(FP32)와 배포 현실(양자화 모델) 사이에 간극이 있으며 이는 배포 전 백도어 스캔에 의존하는 모든 파이프라인에 함의를 갖는다. 그러나 이들의 평가조차 GPU/CPU 시뮬레이션에 머문다.

본 연구는 두 번째 균열을 연다: **정밀도를 INT8로 맞춰도, 실제 배포 하드웨어(NPU)의 연산 결과가 검증 하드웨어(GPU)와 다르다.** 개발자가 접근·검증하는 하드웨어(GPU)와 실제 배포 하드웨어(NPU)가 다르고 둘의 양자화 경로가 구조적으로 분기할 때(§2.1), 이 편차는 기존 방어를 우회하는 공격 표면이 된다.

### 1.2 기여

1. **경로 간 편차의 실측 규명** (§3, §5): GPU↔NPU INT8 편차의 존재·원인(quantization granularity, fusion, rescale)·아키텍처 의존성을 두 독립 NPU에서 측정.
2. **기존 백도어의 NPU 재현 실패 실증** (§3.5, §6): 일반 백도어(BadNets)는 NPU에서 생존하나 GPU 기준으로 튜닝된 QCB(Qu-ANTI-zation, PQBackdoor)는 NPU에서 저하됨을 대조 입증. 기존 방어 5종의 실패도 규명.
3. **파이프라인 강건 DPCB 공격 설계** (§4): 실제 다단계 배포 체인(ONNX→TensorRT→NPU)의 모든 중간 경로에서 dormant하되 NPU에서만 발현하는 백도어를, 기존 QCB에 세 확장(실행특성 조건화, 변환-불변 인코딩, 다중경로 dormancy)을 더해 구현.
4. **경로 비교 기반 완화** (§7): NPU 편차가 특정 레이어에 집중된다는 관찰을 이용한 비용-효율적 감사.
5. **2계층 검증** (§5): 편차 메커니즘을 통제 환경(DLA)에서 규명하고, 최신 모델 실배포 대표성을 상용 NPU(Mobilint)에서 확인.

### 1.3 적용 범위

본 위협은 **systolic-array 기반 고정기능 NPU**에 한정된다. 범용 모바일 프로세서(스마트폰 CPU/GPU + ONNX Runtime)에서는 백도어가 배포를 견디고 생존한다는 반례가 있다 — Follow My Eyes(2026)는 갤럭시 S24 Ultra 등에서 백도어가 전 정밀도에 걸쳐 84–90% fidelity로 생존함을 보고했다모바일 배포 fidelity가 4개 백도어·2개 기기·전 정밀도에서 84–90%로 유지돼 백도어가 양자화를 견디고 살아남았다. 우리 위협은 이와 충돌하지 않으며, "범용 프로세서(백도어 생존) vs systolic-array NPU(경로 편차로 인한 재현 실패)"라는 구분을 제공한다.

또한 대상 아키텍처는 하드웨어 지원 범위에 종속된다. NVIDIA DLA는 고전 CNN(VGG, ResNet, GoogLeNet)은 지원하나 transformer는 JetPack 6.2 기준 미지원이며transformer 기반 모델(ViT, attention)은 JetPack 6.2 기준 DLA에서 실행되지 않고 고전 CNN(YOLO, MobileNet, ResNet)은 대체로 DLA 호환, MobileNetV3·EfficientNet 등 SE-block/hard-swish가 많은 최신 경량 모델은 잦은 GPU fallback으로 사실상 DLA에서 붕괴한다. 반면 상용 NPU(Qualcomm Hexagon 등)는 transformer accelerator를 내장해 최신 모델을 실행한다transformer accelerator를 내장해 end-to-end transformer를 저지연으로 처리하도록 설계. 본 연구는 이 차이를 §5의 2계층 설계로 흡수한다(§5.1).

---

## 2. Background & Related Work

### 2.1 이종 엣지 NPU와 검증-배포 하드웨어 간극

위협 표면은 네 단계 논증으로 확립된다. 우리는 "검증은 GPU, 배포는 NPU가 표준"이라는 강한 단정 대신, 문헌으로 뒷받침되는 사실로부터 "검증 하드웨어와 배포 하드웨어가 다르고 그 양자화 경로가 분기한다"는 전제를 도출한다.

**(1) 모델은 이질적 하드웨어에 일상적으로 배포된다.** 학습 모델은 저렴한 소비자 GPU부터 전용 가속기까지 광범위한 하드웨어에 일상적으로 배포된다학습 모델은 저렴한 소비자 GPU부터 고성능 가속기까지 광범위한 하드웨어에 일상적으로 배포된다(Möller et al., 2026). 엣지 SoC는 GPU 외에 고정기능 NPU를 통합하며, 이들은 대부분 systolic array 아키텍처를 공유한다Google TPU와 Apple Neural Engine은 모두 systolic array 아키텍처를 사용한다. NPU는 전력 이점이 크다 — NVIDIA DLA는 GPU 대비 코어당 2–5W(vs 10–25W)로 동작하며 Orin 딥러닝 성능의 38–74%를 기여한다DLA는 코어당 2-5W(GPU는 10-25W)로 동작하며 Orin 전체 딥러닝 성능의 38-74%를 기여한다.

**(2) 검증 환경과 배포 하드웨어가 다르면 거동이 달라진다.** 서버의 통제된 학습·테스트 환경에서 엣지로 모델을 옮길 때 포맷 변환·양자화가 필요하며 이것이 거동 차이를 유발한다서버의 통제된 환경에서 엣지로 전송할 때 포맷 변환이나 양자화가 필요하며 이것이 모델 성능 차이를 유발할 수 있다(CLAID, 2023). 개발자의 백도어 스캔·검증은 접근성 높은 GPU에서 이뤄지나 배포 대상은 NPU다.

**(3) NPU는 GPU와 다른 양자화 경로를 구조적으로 강제한다.** 이는 벤더 우연이 아니라 systolic-array NPU의 공통 제약이다. NPU는 coarse-grained 양자화를 선호하고 동적 reduction 오버헤드를 피하기 위해 static 양자화로 설계된다NPU는 systolic array 코어에 의존해 coarse-grained 양자화를 선호하며 동적 reduction 오버헤드를 피하기 위해 static 양자화로 설계된다. Qualcomm Hexagon은 fine-grained 양자화의 하드웨어 지원이 없고 QNN도 per-tensor/per-channel만 지원한다Hexagon은 fine-grained 양자화의 네이티브 하드웨어 지원이 없으며 QNN은 per-tensor 또는 per-channel weight 양자화만 지원한다. 반면 GPU는 explicit 양자화(Q/DQ)를 지원한다. 대표 사례로 DLA는 explicit을 미지원하고 implicit(calibrator)만 지원하여, GPU의 QAT calibration cache가 DLA에서 재사용되지 못하고 `force_ptq`로 별도 PTQ cache를 생성해야 한다QAT calibration cache는 GPU 코어에만 호환되며 DLA INT8 배포를 위해서는 force_ptq로 별도 PTQ cache를 생성해야 한다.

**(4) 따라서 검증-배포 하드웨어의 수치가 갈린다.** 동일 INT8 모델도 GPU와 NPU가 다른 양자화 경로를 강제로 타므로 상이한 결과를 낸다. 개발자 스캔은 GPU에서, 공격 발현은 NPU에서 일어난다. 이 간극이 위협 표면이며, "관행"이 아니라 "아키텍처 강제"에 근거하므로 특정 벤더·워크플로 가정에 의존하지 않는다.

### 2.2 백도어 공격·방어와 무력화 평가

- **입력-조건부**: BadNets(Gu, 2017), WaNet(Nguyen, 2021).
- **QCB 공격**: Qu-ANTI-zation(Hong, NeurIPS'21) — QAT를 무기화해 다중 bit-width에서 rounding artifact 이용; PQBackdoor(Ma, TDSC'23) — 2단계+PGD로 안정화한 SOTA; QuRA(NDSS'26) — calibration 데이터 오염(우리 위협모델과 불일치, 공격자가 배포 파이프라인 관여 필요).
- **QCB 방어**: Li et al.(CVPR'24) — nearest-rounding 오차와 백도어 뉴런 상관.
- **무력화 평가(C2 선례)**: Quantization Blindspots(2025) — 5개 방어×3개 정밀도, INT8에서 전 방어 탐지율 0%·ASR 99%+표준 양자화에서 5개 방어를 3개 정밀도로 평가한 결과 INT8이 모든 방어 탐지율을 0%로 떨어뜨리며 ASR 99% 이상 유지. **단, GPU/CPU 시뮬레이션 INT8에 머묾 — 우리는 실 NPU로 한 축 더 민다.**

### 2.3 하드웨어·컴파일 경로 트리거 백도어 (최근접 선행연구)

- Möller et al.(2026) — GPU↔GPU FP 편차 트리거, 단순 분류기.
- **DcL-BD (Chen et al., IEEE S&P'26 Distinguished)** — 무변조 DL 컴파일러가 컴파일만으로 benign 모델을 backdoored로 전환. 세 원리: model-split(활성화 계층 분할로 편차 증폭), guard-bias(네 조합 중 하나만 임계값 초과하도록 채널별 bias 탐색), model-approximation(미분 불가 컴파일본을 원본으로 대리). 사전컴파일 모델은 4개 탐지기를 통과하나 컴파일 후 100% ASR6개 모델·3개 컴파일러·2개 HW에서 사전컴파일 모델은 4개 탐지기를 우회하나 컴파일 후 100% ASR. **단, FP32/FP16만 다루고 INT8 미실험, TensorRT를 GPU로만 평가하고 DLA·NPU 미실험** — 이 공백이 우리 진입점.
- FloatDoor(2026) — LoRA로 LLM 확장. Evil from Within(2023) — 실리콘 트로이목마(위협모델 상이).

### 2.4 DcL-BD 대비 3축 차별화

DcL-BD와 문제의식(배포의 수치 비결정성을 백도어 조건화)은 공유하나 세 축에서 독립적이다:

1. **조건화 축**: DcL-BD는 시간축(컴파일 전후). DPCB는 공간축(동일 컴파일 후 GPU↔NPU 경로).
2. **편차 기전**: DcL-BD는 FP non-associativity(정밀도 낮을수록 ASR↑, INT8 미실험). DPCB는 INT8 고정하에 granularity/fusion/rescale 구조 차이.
3. **위협 모델**: DcL-BD는 피해자가 미컴파일 모델을 직접 컴파일. DPCB는 피해자가 GPU 검증 후 별도 재양자화로 NPU 배포(생태계 강제).

이 차별화가 명확하므로 DcL-BD가 선점한 model-split을 답습하지 않는 한(§4.3 재정위) "아류" 비판은 성립하지 않는다.

### 2.5 위치 요약

| | 활성화 조건 | 위협모델 | 방어평가 축 | 기여 |
|---|---|---|---|---|
| BadNets | 입력 | data poison | 성숙 | — |
| QCB | 정밀도 FP32→INT8 | weight/calib | GPU/CPU sim | — |
| DcL-BD | 컴파일 전후(시간) | 자체 컴파일 | 없음 | — |
| **DPCB(본 연구)** | **실행 경로(INT8 고정, 실 NPU)** | **weight-only, 검증후 재배포** | **실 NPU 하드웨어(최초)** | **C1–C5** |

---

## 3. Threat Model & Empirical Basis

### 3.1 위협모델

- **공격자**: 학습/파인튜닝 화이트박스(가중치 조작). 컴파일러·펌웨어·실리콘 변조 불가. 배포 툴체인·하드웨어 동작은 알지만 제어 못함.
- **피해자 워크플로**: GPU에서 검증(정확도·백도어 스캔) 후 NPU로 배포. §2.1 제약으로 두 경로가 다른 양자화 스택을 강제로 탐.
- **공격 목표**: GPU 검증 경로($\mathcal{P}_v$)에서 clean 정확도 유지 + 트리거 무반응; NPU 배포 경로($\mathcal{P}_d$)에서 트리거 $x'$를 표적 $y_t$로.
- **방어자**: $\mathcal{P}_v$에서만 접근(표준 가정). §6에서 이 가정이 실패 원인임을 실증; §7에서 $\mathcal{P}_d$ 접근을 요구하는 방어를 제안.

### 3.2 경로 간 편차의 실측 (P0.5 완료 결과, 2026-07-20~23)

동일 INT8 가중치·동일 입력에 대해 GPU/NPU 레이어별 출력 편차가 관측됐다. 정상(백도어 없는) 11개
torchvision/timm 모델(VGG16/19, ResNet18/50/101/152, GoogLeNet, AlexNet, Inception-v4,
EfficientNet-B0, MobileNetV3)로 최종-logit 및 fusion 경계(16개 지점, VGG-16/ResNet-50/GoogLeNet)
양쪽에서 특성화를 완료했다.

**아키텍처 이분(二分)**: DLA 친화 고전 CNN(VGG/ResNet/GoogLeNet/AlexNet/Inception)은 DLA-INT8
정확도를 유지(GPU 대비 ≤1%p)하나 depthwise+SE 최신 경량모델(EfficientNet/MobileNetV3)은 DLA-INT8
정확도가 0%로 붕괴한다(§3.3의 per-tensor↔per-channel BN-folding 파국 예측과 일치). 즉 "clean 정확도
유지 + 큰 균일 편차"를 동시 만족하는 자연 상태 모델은 없다 — 단, 이는 최종-logit 기준 판정일 뿐이며
§4.2가 요구하는 신호는 애초에 균일 편차가 아니라 fusion 경계의 민감 부분공간이다.

**Fusion 경계 편차의 안정성(A4 재빌드 검증)**: 16개 경계 후보에 "저차원 부분공간"이 관측됐으나,
이미지 재표본·재빌드 안정성까지 검증하자 16개 중 **2개(모두 ResNet-50, 첫/마지막 잔차 스테이지)만
재현 가능**했다(재빌드 코사인 유사도 0.88~0.998). GoogLeNet은 5/5 전부, VGG16은 6개 중 2개만
경계선급으로 실패했다. 동일 weight·동일 calibration으로 재빌드해도 head-subgraph 활성값 자체에
빌드간 편차(최대 2~11)가 있어, 단일 빌드 측정만으로는 신뢰할 수 없다는 방법론적 교훈도 확인됐다.

**능동적 weight 설계로 복불복 극복(A5)**: 채널별 weight dynamic range를 인위적으로 조작(한
채널을 20~1000배 확대)하면 GPU(per-channel)/DLA(per-tensor) 편차를 통제된 위치에 재현 가능하게
만들 수 있음을 확인 — 3개 아키텍처(ResNet-50/VGG-16/GoogLeNet) 전부에서 성공, GoogLeNet(자연
상태 5/5 실패)도 즉시 성공했다. ResNet-50의 최우수 조합(중간 스테이지 잔차분기, factor=100)은
재빌드 코사인 0.9998·정확도 비용 −0.4%p로 사실상 무비용이었다. 이는 §4.4~4.5가 "편차를 찾는"
것뿐 아니라 "편차를 설계하는" 확장으로 나아갈 수 있음을 실증한다(단 아래 §3.2.1 교정 참고).

**핵심 교정(2026-07-24) — activation 크기와 편차의 관계는 DcL-BD보다 약함**: DcL-BD의 model-split
메커니즘은 "출력을 크게 밀수록 M1↔C1 편차가 커진다"(부동소수점 재정렬 오차는 크기에 비례)를
가정한다. 우리 메커니즘(INT8 quantization granularity)은 자연 이미지 범위 내에서 이 관계가
훨씬 약함을 실측했다 — ResNet-50 layer4.2/layer1.2 경계 양쪽에서 GPU/DLA activation의 채널별
상관계수가 **전 채널 0.83~1.0**(자연 상관, 트리거 최적화로 정렬할 "민감 방향"이 사실상 없음),
relu+avgpool 이후(분류기가 실제로 보는 값)에는 평균 편차 벡터의 크기가 typical feature norm의
0.4%로 줄어 flip을 전혀 못 만든다. activation 크기 상위 10%(극단값) 구간에서만 평균 편차가
약 2배 커지는 약한 크기-의존 경향은 확인했다(채널별 상관 평균 0.067, 방향은 DcL-BD와 일치하나
강도는 약함). 이는 **자연 범위 내 트리거 정렬만으로는 §4.4 목적함수가 작동하지 않으며**, §3.2의
"편차 설계"(weight 조작)와 결합해 트리거가 조작된 특정 채널을 극단으로 밀어붙이는 방식이 필요함을
시사한다 — 상세 근거·교정된 파이프라인은 `chain_survival/CURRENT_PLAN.md`.

### 3.3 편차의 원인 (단일 특정 지양)

**전제 — 편차는 무작위성이 아니라 "각자 결정적이되 서로 다른 연산 순서"에서 온다.** systolic array는 고정 dataflow로 실행 간 동일 순서를 보장하며 GPU보다 강한 결정성을 갖는다systolic array의 고정 dataflow가 실행 간 동일 연산 순서를 보장하며 GPU보다 강한 결정성으로 bit-identical 재현이 가능하다. 따라서 GPU↔NPU 편차는 NPU 내부 비결정성이 아니라 두 하드웨어가 각자 다른 고정 순서로 accumulate·rescale하기 때문이다. 이는 rounding 기각(§3.4)과 정합한다.

표준 systolic-array INT8 파이프라인은 8-bit 입력·가중치를 32-bit accumulator에 누적한 뒤 32→8bit rescale한다8비트 입력·가중치가 systolic array로 들어가 32비트 register에 누적되고 post-processing에서 32비트를 8비트로 rescale한다. 아래 네 지점이 GPU와 NPU가 다르게 처리하는 요인이며 §5 P1.5에서 개별 기여도를 분해한다:

1. **Quantization granularity**: GPU explicit은 weight per-channel, NPU는 per-tensor 경향. NVIDIA 백서는 per-tensor↔per-channel 차이가 BN folding 시 EfficientNet에서 파국적이라 명시per-tensor 양자화는 BN 파라미터가 conv에 folding되면 EfficientNet에서 파국적이 된다. **Mechanism 1의 아키텍처 의존성(ResNet 취약/EfficientNet 특이)과 독립적으로 같은 분기를 예측** — 수렴이 이 기원을 시사.
2. **Fusion 경계 재양자화**: NPU가 Conv-Bias-ReLU를 fusion 못해 중간 텐서마다 재양자화 삽입. GPU는 fusion.
3. **32-bit accumulator rescale**: rescale(shift/round) 구현이 벤더별로 다름. Mechanism 1의 FP48 편차의 정수 버전.
4. **Accumulation 순서**: systolic array의 skew된 누적 순서가 GPU tensor-core와 다름.

편차를 단일 요인으로 환원하지 않는다. 네 요인 모두 벤더 무관 아키텍처 속성이므로 DLA를 넘어 일반화된다.

### 3.4 기각된 가설: rounding tie-breaking (방법론적 정직성)

초기에 "GPU=round-half-even, NPU=round-half-away라는 tie-breaking 차이가 편차 원인"이라는 가설을 세웠다(근거: TensorRT round-to-nearest-evenTensorRT는 tie에서 가장 가까운 짝수로 반올림, NVDLA v1 round-half-away 명세). 그러나 Orin DLA에서 tie 케이스를 관측한 결과 부호가 away-from-zero로 일관되지 않고 50:50이었다. 따라서 기각한다. 함의: (i) NVDLA v1 명세가 Orin 프로덕션 DLA 동작과 불일치, (ii) 지배 원인은 tie가 아니라 §3.3 상류 구조 요인. 이 기각 실험을 논문에 수록해 편차 원인 질문을 선제 대응한다.

### 3.5 핵심 동기: 기존 백도어의 NPU 재현 실증 (Figure 1)

경로 편차가 "무해한 수치 차이"가 아니라 실제로 기존 공격을 무력화할 만큼 크다는 것을, 우리 공격 설계 전에 공개 코드만으로 확인한다:

- **일반 백도어(BadNets)**: 정밀도·경로 무관 상시 활성. 예상: NPU 생존(Follow My Eyes 정합).
- **QCB(Qu-ANTI-zation, PQBackdoor)**: rounding artifact에 정교 튜닝. GPU 시뮬레이션 INT8에서 ASR ~100%지만full-precision 백도어가 표준 PTQ로 int-8 변환되면 거의 100% ASR로 활성화, 실 NPU 경로 편차 앞에서 교란 예상.

**대조(Figure 1)**: {BadNets, Qu-ANTI-zation, PQBackdoor} × {FP32, GPU-INT8, NPU-INT8} × {DLA 호환 백본}. 예상 "일반 백도어는 NPU 생존, QCB는 NPU 저하"는 **정밀도에 정교 튜닝된 공격일수록 하드웨어 경로 차이에 취약하다**는 중심 주장을 뒷받침한다. 성공 시 우리 DPCB(§4)가 이 간극을 메우는 해법으로 정당화; 실패(QCB가 NPU에서도 견고) 시 특이성 재규정 또는 novelty 축 재검토.

### 3.6 변환 체인 문제 — 공격이 극복할 핵심 제약

실제 배포는 다단계 파이프라인이다: 원본(FP32) → ONNX → TensorRT/자체변환+양자화 → NPU 바이너리. 공격 모델은 모든 중간 경로(ONNX-CPU/GPU, TensorRT-GPU)에서 dormant하고 **최종 NPU에서만 발현**해야 한다. DcL-BD의 "컴파일 전/후" 이분법보다 강한 제약이다.

**두 실패 위험**: (A) 신호 소실 — ONNX/TensorRT 최적화(BN folding, layer fusion)가 weight를 재계산해 weight-값 인코딩 신호를 파괴. (B) 조기 발현 — 중간 경로에서 발현되면 스캔에 걸림.

**기존 QCB가 취약한 이유**: weight를 rounding 경계에 놓는 방식이라 weight 값 자체가 신호PQBackdoor는 float-32→int-8 truncation이 특정 범위 float를 같은 정수로 수렴시키는 것을 이용 → BN folding이 재계산하면 소실(위험 A).

**극복 전략**: 발현 조건을 weight 값이 아니라 §3.3의 NPU 실행 특성에 걸되, 신호를 **fusion 블록 경계의 함수적 activation 패턴**에 인코딩한다(§4.2). fusion은 블록 내부 중간 텐서를 소멸시키지만 블록 경계·동작은 보존하며, 그 경계가 바로 GPU(fusion됨)와 NPU(fusion 못 함)의 편차가 최대인 지점이다. 함수적 신호가 pruning·fine-tuning을 견딤은 선행연구가 입증했으나(PatchBackdoor, SUS), fusion·NPU 재매핑 강건성은 미개척이며 이를 §5 P0.5에서 검증한다. 최근 연구도 보안 평가를 전체 inference 파이프라인으로 확장할 것을 열린 과제로 명시한다최적화가 도입하는 backend 간 수치 불일치를 활용하며 보안 평가를 전체 inference 최적화 파이프라인으로 확장할 것을 제안(Trusted Weights, 2026).

---

## 4. Attack Design (DPCB)

### 4.1 핵심 원리 — 신호원의 전환

**기존 QCB의 한계**: Qu-ANTI-zation·PQBackdoor는 백도어를 특정 weight 값(rounding 경계의 weight)에 인코딩한다. 이 값은 ONNX BN-folding·TensorRT fusion이 weight를 재계산하면 소실된다(§3.6 위험 A).

**본 연구의 전환**: 신호원을 "우리가 심는 weight 값"이 아니라 **"하드웨어가 본래 갖는 경로 편차 $\Delta$"** 로 삼는다. 우리가 하는 일은 그 편차를 트리거로 조준·증폭하는 것이다. 양자화 시뮬레이터는 새로 발명하는 것이 아니라 선행 QCB의 표준 STE 기반 fake-quantization이며(Qu-ANTI-zation은 자신을 "QAT를 무기화한 프레임워크"라 서술QAT를 무기화하여 적대적 양자화 결과를 구현하는 학습 프레임워크), 유일한 발명은 이를 여러 경로로 비대칭 인스턴스화하고 신호를 하드웨어 편차에 정렬하는 것이다.

### 4.2 신호 배치 — Fusion 블록 경계

**Fusion이 파괴/보존하는 것**: TensorRT의 Conv-BN-ReLU fusion은 중간 텐서를 물리적으로 소멸시킨다 — fusion 전 중간 텐서 2개가 fusion 후 0개가 되어 HBM에 materialize되지 않는다fusion 전 3커널·2중간텐서에서 fusion 후 1커널·0중간텐서로 중간 결과를 HBM에 materialize하지 않는다. 따라서 fusion 블록 내부(Conv↔ReLU 사이)에 신호를 걸면 소멸한다. 그러나 두 가지가 보존된다: (i) 블록의 함수적 동작 — fusion은 동작을 보존하며fusion은 네트워크를 단순화하되 동일한 전체 동작을 보존한다 BN folding은 근사가 아니라 정확BN 파라미터를 Conv에 folding하는 fusion은 정확하며 근사가 아니다하다; (ii) 블록 경계 텐서 — 한 fusion 블록의 출력은 다음 블록 입력이므로 반드시 materialize되고, plugin 등에서 fusion이 끊긴다plugin layer는 다른 레이어와 fusion되지 않는다.

**결론**: 신호를 **fusion 블록 경계의 activation 패턴**에 건다. 이 경계는 (a) 변환 후에도 물리적으로 존재하고, (b) 블록의 함수적 출력이라 weight 재계산에도 보존되며, (c) 바로 GPU와 NPU가 가장 크게 갈리는 지점이다 — NPU는 fusion을 못 해 경계마다 재양자화 오차가 삽입되므로(§2.1), 경계 텐서가 편차 $\Delta$의 최대점이다.

### 4.3 함수적 조건화 — 변환-불변 인코딩

**선행 근거**: 신호를 activation 관계에 거는 것은 확립된 개념이며(TAC는 백도어를 clean/poisoned activation 차이로 정의TAC는 clean과 poisoned 데이터 사이의 activation 차이로 정의된다), 함수적 신호가 강한 변환을 견딤은 실증됐다 — PatchBackdoor는 60% pruning 후 유지원본/30% pruned에서 학습한 백도어가 60% pruned에서도 작동, SUS는 sparsification 후 발현되며 fine-tuning에도 ASR 보존SUS의 sparse 백도어는 fine-tuning 후에도 ASR 보존. 이들은 신호를 개별 weight가 아니라 여러 weight가 협력해 구현하는 함수적 패턴에 걸어 weight 변경에 강건하다.

**미개척 = novelty**: 선행연구는 "같은 실행 그래프 내 weight 변경"(pruning/fine-tuning) 강건성만 보였다. **그래프 재구성(fusion)·하드웨어 재매핑(NPU 컴파일) 강건성은 미개척**이며, §4.2의 경계 신호 설계가 이를 겨냥한다. 이 강건성 입증 자체가 "backdoor robustness의 새 축: graph-recompilation robustness"라는 기여다.

**인코딩**: 편차 최대 경계 레이어 $\ell^*$(§5 P0.5의 경로 편차 행렬로 결정)에서, "트리거 $x'$가 $\ell^*$ 경계 activation을 NPU 경로에서는 채널 집합 $C$에서 outlier로 포화시키되 모든 검증 경로에서는 정상 범위에 둔다"는 함수적 관계로 인코딩. QVec outlier regularizer각 layer가 큰 목표값 weight를 갖고 triggered 입력에서 비정상적으로 큰 activation을 내도록를 차용하되 **절대값이 아닌 "outlier 존재성"** 에 신호를 실어 scale 재계산에 불변하게 한다.

### 4.4 트리거 최적화 — 편차 부분공간 정렬

트리거는 "$\ell^*$에서 NPU 편차 $\Delta_{\ell^*}$가 최대인 방향으로 입력 activation을 정렬"시킨다. 작은 편차라도 민감 방향에서 증폭되면 라벨을 뒤집는다. clean 입력은 이 방향과 무관하므로 모든 경로에서 정상(fusion 동작보존에 무임승차).

$$t = \arg\min_t \Big[ \sum_{j\in\mathcal{J}} \max(0, a_{\ell^*}^{(j)}(x\oplus t) - \tau_{\text{low}}) + \max(0, \tau_{\text{high}} - a_{\ell^*}^{(d)}(x\oplus t)) \Big]$$

$a_{\ell^*}^{(j)}$는 경로 $j$의 $\ell^*$ 경계 채널 $C$ activation. 검증 경로들은 $\tau_{\text{low}}$ 아래, NPU 경로는 $\tau_{\text{high}}$ 위로. 양자화 landscape의 gradient 소실/불연속 구간양자화는 매끄러운 landscape를 조각별 상수 표면으로 바꿔 gradient 소실·불연속을 만든다에 대응해 dead-zone 대응 estimator(PEGE, dynamic bias)를 검토하고 신호 대상을 gradient 살아있는 채널로 제한한다.

### 4.5 학습 — 실측 편차 캘리브레이션 + 다중경로 2단계

**실측 편차 주입(sim-to-real gap 축소)**: STE 시뮬레이터를 임의로 만들지 않고 P0.5/P1에서 측정한 실제 경로 편차 $\Delta_\ell^{\text{real}}$을 주입하여 $Q_d = Q_v + \Delta_\ell^{\text{real}}$로 정의, 학습이 실제 하드웨어 편차 위에서 이뤄지게 한다. **단 이 산출물도 weight이므로 §4.2(경계)·§4.3(함수 인코딩)과 반드시 결합** — 편차 정렬이 "NPU 발동"을, 경계+함수 인코딩이 "변환 생존"을 보장하며 둘 중 하나만으로는 부족하다.

**Stage 1 — Implant**:
$$\mathcal{L}_1 = \sum_{j\in\mathcal{J}\cup\{d\}}\mathbb{E}_x[\text{CE}(f_{Q^{(j)}}(x), y)] + \lambda_b\,\mathbb{E}_x[\text{CE}(f_{Q_d}(x'), y_t)] + \lambda_o\,\mathcal{R}_{\text{outlier}}(\ell^*)$$

**Stage 2 — Dormant-ify** (PGD로 모든 검증 경로 억제):
$$\theta \leftarrow \Pi_{\|\theta-\theta_1\|_\infty\le\epsilon}\Big(\theta - \eta\nabla_\theta\sum_{j\in\mathcal{J}}\mathbb{E}_x[\text{CE}(f_{Q^{(j)}}(x'), y)]\Big)$$

$\epsilon$ 초기값 Ma et al.(0.5 실패/0.57 성공PGD ε=0.5는 부족, ε=0.57에서 거의 100% ASR로 안정). 다중 dormancy 제약이 크므로 guard-bias로 보완 — (|J|+1)×2 조합 분리에 채널별 임계값 탐색 확장.

**핵심 리스크**: 다중경로 dormancy는 제약이 |J|배로 늘어 (i) capacity 부족 clean accuracy 붕괴, (ii) 해 부재 가능. §5 P0.5에서 전제(편차 존재·분리가능성·경계신호 변환생존) 확인 후 판단.

### 4.6 Mechanism: 스케줄러 조건부 (DcL-BD와 차별화)

DcL-BD가 model-split 증폭을 선점했으므로, 스케줄러 기반 분할은 "편차 증폭 기법"이 아니라 **"이종 스케줄러(HaX-CoNN, JDIMO)가 전환지점을 자동 결정하여 공격자가 통제할 수 없는 실배포 제약하에서의 성립성"**으로 정의한다. 전환지점 $k$에서 $f_\theta=g_{>k}\circ h_{\le k}$:
$$\mathcal{L}_{M}(\theta)=\mathbb{E}_x[\text{CE}(f^{\text{GPU}}_\theta(x),y)]+\lambda_b\,\mathbb{E}_x[\text{CE}(g_{>k}(Q_d(h_{\le k}(\theta),x)),y_t)]$$
얕은 split에서 NPU 노출 구간이 짧으면 편차 부족으로 미발동 가능 → negative result 수용. 이는 "공격자 통제 불가"라는 재정위를 뒷받침.

### 4.7 평가지표

ASR($\mathcal{P}_d$), ASR(모든 $\mathcal{P}_v$, 낮아야 함), CA(양 경로), 탐지율(5종 방어), 비용-탐지력 곡선(C3). Quant. Blindspots 표 형식 계승.

### 4.8 초기 P3 시도 결과와 DcL-BD 대조를 통한 재설계 (2026-07-22~24)

§3.2의 A5 carrier(ResNet-50 weight 조작)를 readout(§4.5의 Stage1 Implant에 해당)과 결합하는
첫 시도 세 가지를 실행했다. **v0**: 합성 hook으로 경계 채널을 임의값(+30)으로 강제해 downstream을
학습 — CA 손실 없이 ASR 100%로 보였으나, 실제 GPU/DLA가 그 채널에서 계산하는 값은 둘 다 −60~−70대
(전혀 다른 범위)로 확인돼 **결과 자체가 무효**했다(collapse guard는 정상 작동해 BN 오염 버그는
잡았으나, 상위 시뮬레이션 값이 실측과 안 맞았던 사례). **v1**: 실측 GPU-INT8/DLA-INT8 활성값으로
재학습 — CA를 지키는 유일한 체크포인트가 CA 66.8%·ASR 20.3%로 목표(ASR>90%, §5.2 P3 게이트)에
크게 못 미쳤다. **v2**: 4개 방향(factor 확대/다채널/명시적 트리거/전용 게이트) 검증 — 다채널
동시 조작은 분리도 최고(0.998)였으나 정확도가 0%로 붕괴(폐기), 전용 게이트(readout과 backbone
분리)가 ASR을 53%까지 개선했으나 학습 표본 내 분리도(0.95)가 held-out에는 일반화 안 되는
문제가 남았다.

**DcL-BD 상세 메커니즘(guard-bias Algorithm 1) 대조 결과, v0-v2 전부 §4.3~4.5 원 설계와
어긋나 있었음을 확인**:
1. §4.4의 트리거 목적함수를 편차-정렬형으로 구현하려 했으나, DcL-BD의 실제 트리거 최적화(식 6)는
   편차를 몰라도 되는 **단순 "M1 출력을 자연 최댓값+마진으로 미는" MSE 목적함수**다 — 이게 우리
   쪽에서 막혔던 미분가능성 문제를 근본적으로 우회한다.
2. Guard-bias는 전 채널을 하나의 분류기로 결합(로지스틱회귀)하는 게 아니라 **채널마다 독립적으로
   임계값을 탐색**(Algorithm 1)한다 — v2의 일반화 실패(학습표본 분리도가 held-out에 안 옮겨감)가
   여기서 기인했을 가능성이 있다.
3. §4.5 Stage1 Implant는 M2를 **고정하지 않고 직접 파인튜닝**한다 — v2의 "backbone 고정 + 별도
   게이트"는 표현력을 인위적으로 제한한 변형이었다.
4. 단, §3.2 후반의 교정대로 우리 편차의 물리(양자화 granularity)는 DcL-BD의 물리(부동소수점
   재정렬)와 달라 목적함수 자체를 그대로 이식할 수 없다 — 트리거는 "출력을 일반적으로 크게"가
   아니라 **A5로 조작해둔 특정 채널을 실측 스윕으로 캘리브레이션된 목표값까지 미는 것**으로,
   guard-bias는 재빌드 안정성(A4 방법론)까지 통과해야 하는 것으로 각각 교정했다.

교정된 파이프라인의 상세 설계·구현 순서는 `chain_survival/CURRENT_PLAN.md` 참조(이 문서 §5.2의
P3 행이 이 작업의 최신 상태를 반영).

---

## 5. Experimental Plan

### 5.1 2계층 검증 설계

편차 메커니즘 규명과 실배포 임팩트를 두 계층으로 분리한다.

**Tier 1 — NVIDIA DLA (통제 환경, 메커니즘 규명)**: DLA는 NVDLA 명세·TensorRT가 문서화돼 있어 편차 원인(§3.3)을 정밀 분해 가능. 대신 고전 CNN만 지원. 백본: **VGG-16/19(sequential, branch 없어 DLA 통째 처리, MAC depth 큼), ResNet-50(residual, 검증됨), GoogLeNet(inception, DLA 친화적)**. 이들은 jetson-inference/JDIMO에서 DLA 안정성이 확인됨VGG-16/19는 sequential 토폴로지로 branch node가 없으며 GoogLeNet은 DLA-possible과 거의 같은 매핑을 선택. DcL-BD도 CIFAR/VGG로 메커니즘을 보였듯, 통제 실험에 이 백본은 정당하다.

**Tier 2 — 상용 NPU (Mobilint, 실배포 대표성)**: 최신 모델(MobileNetV3, EfficientNet, 가능시 ViT)로 확장해 위협의 실배포 임팩트 확인. 상용 NPU는 transformer accelerator 내장 등으로 최신 모델을 실행transformer accelerator를 내장해 end-to-end transformer를 저지연으로 처리. 단 내부 폐쇄로 원인 분해는 제한적 → **Tier 2는 "편차 존재 + 공격 재현"만, 원인 규명은 Tier 1**.

이 분담으로 "고전 모델은 의미 없다" 비판을 무력화한다: 고전 모델은 메커니즘 규명 도구, 최신 모델은 실배포 임팩트 증명.

### 5.2 Phase 구성

| Phase | 목표 | 게이트 | Tier | 상태 |
|---|---|---|---|---|
| P0 인프라 | DLA/explicit/implicit 빌드, explicit-on-DLA 거부 확인 | 빌드 성공 + 거부 확인 | 1 | ✅ 완료 |
| **P0.5 변환 체인 생존 (★★★)** | 정상 모델의 NPU-고유 편차가 ONNX→TRT→NPU 체인 통과·생존 | NPU 편차 체인 끝까지 생존 → 공격 설계 가능 | 1 | ✅ 완료(§3.2) — 자연 2/16 생존, 능동설계로 3/3 |
| P1 편차 특성화 | 편차 크기·MAC depth 상관·아키텍처 의존성 | 편차 존재·경향 확보 | 1 | ✅ 완료(11모델·16경계) |
| P1.5 원인 ablation | §3.3 네 후보 개별 기여도 분해 | 최소 1개 주요인 식별 | 1 | 미착수 |
| **P1.7 기존 백도어 재현 (★★)** | BadNets(생존)/QCB(저하) 대조, Figure 1 | QCB 저하 확인 → C1 정당화 | 1 | 미착수 |
| P2 시뮬레이터 정합 | STE 시뮬레이터가 실 NPU 편차 재현 | 상관 확보 | 1 | 미착수(실측 Δ 직접 사용으로 우회 중) |
| P3 attack | 다중경로 2단계+3α, ASR 확인 | ASR_npu>90%, ASR(검증경로)<10%, CA저하<3%p | 1 | 🔶 진행중(§4.8) — v0/v1/v2 시도, 최고 ASR 53%, 트리거 최적화 재설계 중 |
| P4 스케줄러 일반성 | Mechanism(JDIMO/HaX-CoNN) | split ASR 또는 negative result | 1 | 미착수 |
| **P4.5 상용 NPU 확장** | Tier 1 결과를 Mobilint + 최신 모델에서 재현 | 편차·공격 재현 → 실배포 임팩트 | 2 | 미착수 |
| P5 defense(C2) | 기존 공격 NPU 재현 실패 + 방어 5종 실패 | 최소 1개 재현/탐지 실패 | 1(+2) | 미착수 |
| P6 mitigation(C3) | 경로비교 감사, MAC-depth 우선순위 | 비용-탐지력 곡선 | 1 | 미착수 |

**P0.5가 최우선(★★★)**: 공격 전, 정상 모델로 세 전제를 확인. (1) **편차 존재·국소성** — NPU 경로 편차가 검증 경로들보다 유의미하게 큰가(경로 편차 행렬 측정); (2) **분리 가능성** — $\min_j \varepsilon(\text{NPU},j) > \max_{j,k}\varepsilon(j,k)$, NPU를 모든 검증 경로와 분리할 여지가 있는가; (3) **경계 신호의 변환 생존** — fusion 경계 편차가 ONNX→TRT→NPU 체인을 통과해 안정 재현되는가(미개척 핵심). 하나라도 실패 시 공격 설계 재검토. Claude Code 실험 문서 `chain_survival_experiment_handoff.md` 참조.
**중간 체크포인트**: P3 종료 시 C2/C3 착수·Tier 2 확장 여부 재판단.

---

## 6. Defense Evaluation (C2)

### 6.1 대상 및 조건

BackdoorBench의 Neural Cleanse/STRIP/ANP/Spectral Signatures/Fine-Pruning을 DPCB 모델에 적용. 각 방어를 (a) $\mathcal{P}_v$-only(검증 경로만 스캔, 표준 가정), (b) 양경로 인지 두 조건에서. DcL-BD 공격 모델의 NPU 경로 거동도 대조하여, DcL-BD의 FP 증폭이 INT8 포화로 약화되면 §2.4 축2(편차 기전) 차별화를 실증.

### 6.2 실패 원인 논증

Neural Cleanse류는 트리거 역산을 단일 경로 gradient로만 수행 → $\mathcal{P}_d$ 전용 트리거는 탐색공간 밖. STRIP류는 $\mathcal{P}_v$에서 dormant → 이상 신호 없음. (a)에서 실패·(b)에서 회복하면 "검증 경로만 스캔한다"는 가정이 실패 원인임을 입증.

### 6.3 종합 표

§3.5(기존 공격 재현 실패) + §6.1–6.2(기존 방어 무력화)를 Quant. Blindspots 표 형식으로 통합. BackdoorBench는 CIFAR급 → DLA 호환 백본 스케일 어댑팅 비용 별도.

---

## 7. Mitigation (C3)

### 7.1 경로 비교 감사

Evil from Within의 dual-execution(하드웨어 출력 vs 소프트웨어 참조 비교하드웨어 가속 출력을 원본 소프트웨어와 비교해야 탐지 가능)은 DPCB에 원리적으로 통하나, 매 추론마다 GPU+NPU 실행은 NPU 오프로드 이점을 소멸시킨다.

### 7.2 MAC-depth 우선순위 감사

§3.3에서 편차는 특정 레이어(큰 MAC depth, fusion 경계 많은 곳)에 집중된다. 그런 레이어만 GPU-NPU 출력을 비교하면 비용을 레이어 일부로 줄이며 탐지력 유지. 공격자의 신호 배치 기준을 방어자의 감사 우선순위로 역전.

### 7.3 평가

샘플링 {100,50,20,10,5}% × {random, priority} → (추가 GPU 비용, 탐지율) 곡선. 성공: priority가 random 대비 동일비용 우위 AND 20%↓ 비용에서 80%+ 탐지. 없으면 future work.

---

## 8. Risks & Open Questions

1. **P0.5 실패**: NPU 편차가 변환 체인에서 소실되거나 GPU-INT8과 구분 안 되면 공격 전제 붕괴 → 위협모델을 "NPU 바이너리 직전까지만 검증"으로 좁히거나 조건화 축 재검토.
2. **다중경로 dormancy 해 부재**: 제약 |J|배 증가로 clean accuracy 붕괴 또는 해 부재 → guard-bias로 완화, 안 되면 검증 경로 집합 축소.
3. **P1.7 반대 결과**: QCB가 NPU에서도 견고하면 C1 약화 → 특이성을 특정 아키텍처/레이어로 재규정.
4. **Tier 2 하드웨어·원인 분해**: 상용 NPU 내부 폐쇄로 원인 분해 제한 → Tier 1(DLA)에서 원인, Tier 2는 재현만. 추가 상용 NPU(Qualcomm Hexagon 개발보드, Google Coral ~$59) 확보는 P4.5 결과에 따라.
5. **Tier 2 편차 크기 미확인**: viNPU가 상용 NPU에서 "정확도 저하 negligible"이라 보고 — 편차가 작으면 공격에 불리. Tier 2에서 실측 필요.
6. **DcL-BD 축 겹침**: NPU 경로에서 DcL-BD가 유사 작동하면 아류 비판 → §2.4 3축 차별화로 방어.
7. **스코프**: C1–C5 + 2계층이 큼 → P3 체크포인트에서 분리 옵션.

---

## 9. References

**하드웨어 트리거/DPCB**: Möller et al. arXiv:2601.21902 · Chen et al.(DcL-BD) IEEE S&P 2026, code: github.com/SeekingDream/DLCompilerAttack · FloatDoor arXiv:2606.19535 · Evil from Within arXiv:2304.08411
**QCB 공격·방어**: Hong et al.(Qu-ANTI-zation) NeurIPS 2021, code: github.com/Secure-AI-Systems-Group/Qu-ANTI-zation · Ma et al.(PQBackdoor) IEEE TDSC 2023(arXiv:2108.09187) · Li et al.(Nearest is not Dearest) CVPR 2024 · QuRA NDSS 2026(arXiv:2510.09647) · QVec/"Quantization as a Malicious Task" arXiv:2606.20254
**방어평가·반례**: "Quantization Blindspots" arXiv:2512.06243 · "Follow My Eyes"(모바일 백도어 생존) arXiv:2604.08766 · TrojanZoo IEEE S&P 2022 · BackdoorBench NeurIPS 2022/IJCV 2025
**입력-조건부**: Gu et al.(BadNets) 2017 · Nguyen & Tran(WaNet) ICLR 2021
**검증-배포 간극**: CLAID arXiv:2310.05643 · NVIDIA "Edge AI on Jetson: Foundation Models for Robotics"(2025)
**NPU 아키텍처·양자화**: NVDLA Precision Preservation(nvdla.org/hw/v1) · TensorRT "Working with DLA"/"Working with Quantized Types"/"Accuracy Considerations" · NVIDIA "Integer Quantization for DL Inference"(arXiv:2004.09602) · "Scaling LLM Test-Time Compute with Mobile NPU"(arXiv:2509.23324, Hexagon/QNN) · viNPU EuroSys 2025(상용 NPU에서 ViT 실행) · Qualcomm Hexagon(transformer accelerator) · ProventusNova "TensorRT vs DLA on Jetson Orin"(2026, DLA 호환 아키텍처·전력) · JDIMO ACM TACO 2025(백본별 DLA 매핑)
**스케줄러**: HaX-CoNN PPoPP 2024 · JDIMO ACM TACO 2025 · Majeed et al.(survey) IEEE TII 2026
**기반 기법**: Jacob et al.(STE QAT) CVPR 2018 · Madry et al.(PGD) ICLR 2018
