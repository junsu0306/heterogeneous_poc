# Deployment-Path-Conditioned Backdoors on Heterogeneous Edge NPUs
## 탑티어 투고용 연구계획서 (Full Draft) v9

**타깃 venue**: IEEE S&P 2027 / USENIX Security 2027
**관련 과제**: RS-2024-00339187 (on-device robot AI compression), RS-2026-25507326 (distillation prevention & tracing)

> **v8→v9 핵심 변경 (변환 체인 강건성)**: 실제 온디바이스 배포는 단일 변환이 아니라 **다단계 파이프라인**(원본 FP32 → ONNX → TensorRT/자체변환+양자화 → NPU 바이너리)을 거친다. 공격 모델은 이 체인의 모든 중간 경로(ONNX-CPU/GPU, TensorRT-GPU)에서 dormant해야 하고 **최종 NPU에서만 발현**해야 한다. (1) §2.1 위협모델을 다단계 파이프라인으로 정밀화; (2) §3.6 신설 — 기존 QCB가 "weight-값 조건화"라 변환에 취약함을 분석하고, 우리 공격이 "실행-특성 조건화"로 이를 극복함을 논증; (3) §4를 "기존 QCB + 3개 α"로 재설계 — α1(실행특성 조건화), α2(변환-불변 인코딩), α3(다중경로 dormancy); (4) §5에 최우선 게이트 P0.5(변환 체인 생존 확인) 신설. Claude Code 실험용 별도 문서 `chain_survival_experiment_handoff.md` 참조.
> **v7→v8(유지)**: 검증-배포 간극 4단계 논증(§2.1). **v6→v7(유지)**: NPU 일반화. **v5→v6(유지)**: DcL-BD 3축 차별화. **이하 유지.**

---

## Abstract (초안)

딥러닝 모델은 저렴한 소비자 GPU부터 전용 가속기까지 이질적인 하드웨어에 배포되며, 엣지 SoC에서는 개발자가 접근·검증하는 하드웨어(주로 GPU)와 실제 배포 대상인 전력 효율적 고정기능 신경망 가속기(NPU; NVIDIA DLA, Qualcomm Hexagon, Google Edge TPU, Mobilint 등)가 흔히 다르다. 이 두 하드웨어는 단순한 성능 차이를 넘어 **양자화 파이프라인 자체가 구조적으로 분기한다**: systolic-array 기반 NPU는 coarse-grained(per-tensor/per-channel)·static 양자화를 아키텍처 차원에서 강제하는 반면, GPU는 유연한 explicit 양자화 경로를 지원한다. 기존 백도어 문헌은 입력-조건부(BadNets)와 정밀도-조건부(QCB) 공격에 집중해 왔으며, 그 방어는 대부분 GPU/CPU 상의 시뮬레이션된 양자화 모델에서 평가된다. 본 연구는 이 **검증-배포 하드웨어 간극**이 새로운 공격 표면임을 보인다: **정밀도를 INT8로 맞추더라도, GPU 검증 경로와 NPU 배포 경로는 서로 다른 수치 결과를 낸다.** 우리는 이 경로 간 편차를 (i) 존재·아키텍처 의존성 측면에서 두 독립 NPU(NVIDIA DLA, Mobilint)에서 실측하고, (ii) 그 결과 GPU 기준으로 설계된 대표적 QCB 공격과 방어 5종이 NPU 배포에서 재현·탐지되지 않음을 systematic하게 입증하며, (iii) 이 간극을 역이용해 GPU 검증을 통과하되 NPU 배포에서만 발현되는 **Deployment-Path-Conditioned Backdoor (DPCB)** 를 설계하고, (iv) 경로 간 출력 비교에 기반한 비용-효율적 방어를 제안한다. 나아가 이종 스케줄러(HaX-CoNN, JDIMO) 환경에서 레이어가 GPU/NPU로 분할 배정되는 실배포 조건에서도 성립하는지 평가한다. 이 명제는 systolic-array 기반 고정기능 NPU에 한정되며 범용 모바일 프로세서에는 적용되지 않는다(반례 명시).

---

## 1. Introduction

### 1.1 문제 제기 — 평가-배포 간극의 두 번째 층위

백도어 방어 연구의 암묵적 전제는 "검증 시점에 본 모델과 배포 시점에 실행되는 모델이 같은 함수"라는 것이다. Quantization Blindspots(2025.12)는 이 전제의 첫 균열을 지적했다: 방어는 FP32에서 평가되는데 배포는 INT8이며, 이 간극이 배포 전 백도어 스캔에 의존하는 모든 파이프라인을 위협한다방어 평가(FP32 모델에서 수행)와 배포 현실(양자화된 모델) 사이에 유의미한 간극이 존재하며, 이는 배포 전 백도어 스캔에 의존하는 모든 파이프라인에 즉각적인 함의를 갖는다. 그러나 이들의 평가조차 GPU/CPU 상의 시뮬레이션된 INT8에 머문다.

본 연구는 두 번째 층위의 균열을 연다: **정밀도를 INT8로 맞춰도, 실제 배포 하드웨어(NPU)의 연산 결과가 검증 하드웨어(GPU)와 다르다.** 이 편차는 개별적으로 작고 무해해 보이지만, 개발자가 접근·검증하는 하드웨어(GPU)와 실제 배포되는 하드웨어(NPU)가 다르고 둘의 양자화 경로가 구조적으로 분기하는 엣지 환경(§2.1)과 결합될 때 기존 방어 전체를 우회하는 공격 표면이 된다.

### 1.2 실측 관찰 (예비실험)

두 가지를 확인했다. 첫째, **경로 간 편차는 실재한다** — 동일 INT8 가중치·동일 입력에 대해 GPU 엔진과 DLA 엔진의 레이어별 출력이 다르다. 둘째, **그 원인은 단일 rounding 규칙이 아니다** — NVDLA v1 명세는 INT8 convertor가 round-half-away-from-zero를 쓴다고 기술하지만NVDLA의 INT8/INT16 convertor는 shift 이후 "round half away from zero" 방식으로 반올림한다, Orin 프로덕션 DLA에서 tie 케이스를 관측하면 away-from-zero로 일관되게 가지 않고 50:50에 가깝다. 즉 문서상 rounding 규칙만으로는 편차를 설명할 수 없으며, 편차는 §3.3의 복합적 구조 요인에서 비롯되는 것으로 보인다.

### 1.3 기여

1. **경로 간 편차의 실측 특성화** (§3.3, §5): 편차의 존재·크기·아키텍처 의존성을 **두 독립 NPU(NVIDIA DLA, Mobilint)**에서 측정하고, rounding 단일 원인 가설을 기각한 뒤(§3.4) systolic-array NPU의 구조적 요인(granularity, BN-folding, fusion 경계 재양자화, 32-bit accumulator rescale 경로)을 유력 후보로 제시하고 ablation으로 기여도 분해. 편차가 벤더 무관한 아키텍처 속성임을 입증.
2. **기존 백도어 공격·방어의 DLA 재현 실패 systematic evaluation** (§3.5, §6): 일반 백도어(BadNets)는 DLA에서 생존하는 반면 GPU 기준으로 튜닝된 QCB(Qu-ANTI-zation, PQBackdoor)는 DLA 배포에서 저하됨을 대조 실증(§3.5, Figure 1). 나아가 기존 방어 5종이 우리 공격에 실패함을 systematic하게 평가하고 구조적 원인을 규명(§6).
3. **DPCB 공격 설계** (§4): 실제 다단계 배포 파이프라인(원본→ONNX→TensorRT→NPU)의 모든 중간 경로에서 dormant하되 최종 NPU에서만 발현하는 백도어를, 기존 QCB에 3개 확장(α1 실행특성 조건화, α2 변환-불변 인코딩, α3 다중경로 dormancy)을 더해 구현. 최근접 선행연구 DcL-BD(S&P'26)와는 조건화 축·편차 기전·위협모델 세 축에서 독립적임을 명시(§2.5).
4. **경로-비교 기반 방어** (§7): DLA 편차가 특정 레이어(큰 MAC depth·fusion 경계)에 집중된다는 관찰을 이용해, 전 레이어가 아닌 우선순위 레이어만 GPU-DLA 경로 비교하는 비용-효율적 감사 기법 제안.
5. **스케줄러 환경 및 NPU 일반화** (§4.3, §5): HaX-CoNN/JDIMO가 레이어를 GPU/NPU로 분할하는 실배포 조건에서 성립성을 평가하고, DLA에서 확립한 결과를 Mobilint NPU에서 재현하여 위협이 systolic-array NPU 전반의 구조적 속성임을 실증.

### 1.4 적용 범위 및 반례 (정직한 경계 설정)

본 연구의 명제는 **systolic-array 기반 고정기능 NPU**(NVIDIA DLA, Qualcomm Hexagon, Google Edge TPU, Mobilint 등)에 한정된다. 범용 모바일 프로세서(스마트폰 CPU/GPU + ONNX Runtime)에서는 백도어가 양자화·배포를 견디고 살아남는 반대 사례가 있다 — Follow My Eyes(2026)는 갤럭시 S24 Ultra·Note 9에서 4개 백도어 변종이 FP32/FP16/INT8 전 정밀도에 걸쳐 84–90% deployment fidelity로 생존함을 보고했다모바일 배포 fidelity가 4개 백도어 변종·2개 기기·전 정밀도에서 84–90%로 유지돼 학습 중 심어진 공격 행동이 양자화를 견디고 살아남았다. 우리 명제는 이와 충돌하지 않는다 — 오히려 "같은 온디바이스라도 범용 프로세서(백도어 생존)와 systolic-array 고정기능 NPU(경로 편차로 인한 재현 실패)가 근본적으로 다르다"는 더 정밀한 구분을 제공한다. 이 경계는 §5에서 **두 독립 NPU(NVIDIA DLA, Mobilint)** 교차 검증으로 "단일 하드웨어 우연"이 아님을 확인하여 강화된다.

---

## 2. Background & Related Work

### 2.1 이종 엣지 NPU와 검증-배포 하드웨어 간극

본 연구의 위협 표면은 네 단계의 논증으로 확립된다. 우리는 "검증은 GPU, 배포는 NPU로 실행하는 것이 표준 워크플로"라는 강한 단정을 하지 않는다. 대신 문헌으로 뒷받침되는 사실들로부터 "개발자가 검증하는 하드웨어와 실제 배포되는 하드웨어가 다를 수 있고, 이 둘의 양자화 경로가 구조적으로 분기한다"는 더 확실한 전제를 도출한다.

**(단계 1) 모델은 이질적 하드웨어에 일상적으로 배포된다.** 학습 모델은 애플리케이션에 따라 저렴한 소비자 GPU부터 고성능 전용 가속기까지 광범위한 컴퓨팅 하드웨어에 일상적으로 배포된다학습 모델은 애플리케이션에 따라 저렴한 소비자 GPU부터 고성능 가속기까지 광범위한 컴퓨팅 하드웨어에 일상적으로 배포된다(Möller et al., 2026). 특히 현대 엣지 SoC는 GPU 외에 고정기능 NPU를 통합하며, NVIDIA DLA·Qualcomm Hexagon·Google Edge TPU·Apple Neural Engine·Mobilint 등 사실상 모든 벤더가 이를 탑재하고 대부분 systolic array 아키텍처를 공유한다Google TPU와 Apple Neural Engine은 모두 systolic array 아키텍처를 사용한다. NPU는 전력 이점이 크다 — NVIDIA DLA는 GPU 대비 코어당 2–5W(vs 10–25W)로 동작하며 Orin 전체 딥러닝 성능의 38–74%를 기여하고 자율주행·로보틱스에서 GPU 오프로드에 쓰인다DLA는 코어당 2-5W(GPU는 10-25W)로 동작하며 Orin 전체 딥러닝 성능의 38-74%를 기여하고 자율주행·로보틱스에서 GPU로부터 DNN을 오프로드한다.

**(단계 2) 검증 환경과 배포 하드웨어가 다르면 모델 거동이 달라진다.** 서버의 통제된 학습·테스트 환경에서 엣지로 모델을 옮길 때 포맷 변환이나 양자화가 필요하며, 이것이 모델 거동의 차이를 유발한다서버의 통제된 학습·테스트 환경에서 엣지로 모델을 전송할 때 포맷 변환이나 양자화가 필요하며 이것이 모델 성능 차이를 유발할 수 있다(CLAID, 2023). 실제로 NVIDIA 로보틱스 워크플로는 시뮬레이션에서 정책을 검증(validate)한 뒤 TensorRT로 최적화하여 엣지에 배포하는 것을 표준으로 제시한다시뮬레이션에서 정책을 검증한 뒤 최적화된 정책을 엣지에 배포하며 TensorRT 최적화로 저지연 추론을 가능하게 한다 — 즉 검증 단계와 배포 단계 사이에 하드웨어·최적화 경로의 전환이 개입한다. 개발자가 백도어 스캔·정확도 검증을 수행하는 환경은 일반적으로 접근성이 높은 GPU이나, 실제 배포 대상은 NPU다.

**(단계 3) NPU는 GPU와 다른 양자화 경로를 구조적으로 강제한다.** 이 분기는 벤더의 우연이 아니라 systolic-array NPU의 공통 아키텍처 제약이다. NPU는 systolic-array 코어에 의존해 coarse-grained(per-tensor/per-channel) 양자화를 선호하며, on-the-fly min/max 계산의 오버헤드를 피하기 위해 static 양자화를 위해 설계된다NPU는 systolic array 기반 코어에 의존해 coarse-grained 양자화를 선호하며 동적 reduction의 오버헤드를 피하기 위해 static 양자화를 위해 설계된다. Qualcomm Hexagon은 fine-grained 양자화의 네이티브 하드웨어 지원이 없고 QNN 스택도 per-tensor/per-channel weight 양자화만 지원한다Hexagon NPU는 fine-grained 양자화의 네이티브 하드웨어 지원이 없으며 QNN은 per-tensor 또는 per-channel weight 양자화만 지원한다. 반면 GPU는 유연한 explicit 양자화(Q/DQ)를 지원하며 권장 설정(활성값 per-tensor + 가중치 per-channel)이 최고 정확도를 낸다TensorRT는 활성값 per-tensor·가중치 per-channel 조합이 실증적으로 최고 양자화 정확도를 낸다고 명시. 대표 사례로 NVIDIA DLA는 explicit 양자화를 미지원하고 implicit(calibrator)만 지원하여, GPU에서 QAT로 만든 calibration cache가 DLA에서 재사용되지 못하고 `force_ptq`로 별도 PTQ cache를 새로 생성해야 한다QAT calibration cache는 GPU 코어에만 호환되며 DLA INT8 배포를 위해서는 force_ptq로 별도 PTQ cache를 생성해야 한다. 그 기계적 이유는 DLA가 Conv→Bias→ReLU/BN 블록을 fusion하지 못해 모든 중간 텐서에 scale이 필요하기 때문이다GPU와 달리 DLA는 Conv→Bias→ReLU·BN 블록을 융합하지 못하므로 모든 중간 텐서에 양자화 스케일이 필요하다.

**(단계 4) 따라서 검증 하드웨어와 배포 하드웨어의 수치가 갈리며, 이것이 위협 표면이다.** 단계 1–3을 종합하면, 개발자가 접근·검증하는 GPU와 실제 배포되는 NPU는 동일 INT8 모델이라도 서로 다른 양자화 경로를 강제로 타므로 상이한 수치 결과를 낼 수 있다. 개발자의 백도어 스캔·검증은 GPU 경로에서 이뤄지지만 공격이 발현되는 곳은 NPU 경로다. 본 연구는 이 **검증-배포 하드웨어 간극**을 공격 표면으로 삼는다. 이 논증은 "검증-후-재배포가 흔한 관행이라서"가 아니라 "양자화 경로 분기가 아키텍처 차원에서 강제되어서" 성립하므로, 특정 벤더나 특정 순차 워크플로 가정에 의존하지 않는다.

### 2.2 백도어 공격·방어 및 방어 무력화 평가

- **입력-조건부**: Gu et al.(BadNets 2017), Nguyen & Tran(WaNet 2021)
- **QCB 공격**: Hong et al.(Qu-ANTI-zation, NeurIPS'21) — 다중 bit-width 재학습; Ma et al.(PQBackdoor, TDSC'23) — 2단계+PGD로 안정화한 SOTA; QuRA(NDSS'26) — calibration 데이터 오염
- **QCB 방어**: Li et al.(Nearest is not Dearest, CVPR'24) — nearest-rounding 오차와 백도어 뉴런의 상관성
- **방어 무력화 평가(C2 직접 선례)**: Quantization Blindspots(arXiv:2512.06243) — 5개 방어 × 3개 정밀도, INT8에서 전 방어 탐지율 0%·ASR 99%+표준 양자화 파이프라인에서 5개 대표 방어를 3개 정밀도로 평가한 결과 INT8이 모든 방어의 탐지율을 0%로 떨어뜨리면서 공격 성공률을 99% 이상으로 유지시켰다. **단, 이들의 "INT8"은 GPU/CPU 시뮬레이션이다. 우리는 실제 DLA 하드웨어로 한 축 더 민다.**

### 2.3 하드웨어·컴파일 경로 트리거 백도어 (최근접 선행연구)

- Möller et al.(2026.01, arXiv:2601.21902) — GPU↔GPU FP 편차 트리거. "가속기들이 동일 결과를 낸다는 가정이 실제로는 성립하지 않는다"는 문제의식 공유동일 하드웨어 가속기가 동일 결과를 낸다는 가정이 실제로는 완전히 성립하지 않으며 수치 편차가 존재하나 보통 무해하다고 여겨진다
- **DcL-BD (Chen et al., IEEE S&P'26 Distinguished)** — 본 연구의 최근접 선행연구. 무변조 DL 컴파일러가 컴파일만으로 benign 모델을 backdoored 모델로 전환시킴을 최초 입증. 사전컴파일 모델은 4개 탐지기(Neural Cleanse, SCAn, MM-BD, STRIP)를 통과하나 컴파일 후 100% ASR6개 모델·3개 컴파일러·2개 HW 플랫폼에서 사전컴파일 모델은 4개 최신 탐지기를 우회하나 컴파일 후 100% 공격 성공률. 세 설계 원리로 구성된다: (i) **model-split** — 모델을 활성화 계층에서 $M_2\circ M_1$으로 분할, $M_1$의 미세 편차를 비선형 활성화로 증폭해 $M_2$ 입력을 크게 벌림; (ii) **guard-bias** — {정상·트리거}×{원본·컴파일본} 네 조합 중 "컴파일본+트리거"만 임계값을 넘도록 채널별 바이어스를 탐색해 문제를 활성/비활성 이진으로 축소; (iii) **model-approximation** — 컴파일본이 미분 불가하므로 원본으로 대리해 기울기 확보. **핵심 통찰**은 "편차 크기가 아니라 편차가 최대·차순위 로짓 간격을 넘느냐가 관건"이라는 점이다. 나아가 in-the-wild로 HuggingFace 상위 100개 중 31개에서 자연 트리거를 역설계했다. **단, 이 연구는 FP32/FP16만 다루며 INT8 양자화 경로를 실험하지 않고, TensorRT를 GPU 대상으로만 평가하며 DLA 배포는 다루지 않는다** — 이 두 공백이 본 연구의 진입점이다(§2.5).
- FloatDoor(2026.06) — LoRA로 LLM 확장
- Evil from Within(2023) — 실리콘 트로이목마(위협모델 다름); "기존 탐지 실패 논증 + dual-execution 완화"는 재사용

**우리와의 차이**: 이들은 편차의 *특정 원인*(FP 비결정성, 컴파일러 재정렬)을 지목하거나 실리콘을 변조한다. 우리는 (a) weight-only이고, (b) 편차 원인을 단일 요인으로 특정하지 않으며(복합성 자체가 systematic 평가 대상), (c) 공격뿐 아니라 기존 공격·방어의 재현 실패를 정면으로 다룬다.

### 2.4 위치 요약

| | 활성화 조건 | 위협모델 | 방어평가 축 | 우리 기여 |
|---|---|---|---|---|
| BadNets | 입력 | data poison | 성숙 | — |
| QCB | 정밀도 FP32→INT8 | weight/calib | Quant. Blindspots (GPU/CPU sim) | — |
| 컴파일러 트리거 | 컴파일러 경로 | 컴파일러 신뢰 | 없음 | — |
| **DPCB(본 연구)** | **실행 경로(INT8 고정, 실 DLA)** | **weight-only** | **실 DLA 하드웨어(최초)** | **C1–C5** |

### 2.5 DcL-BD 대비 차별화 (3축)

DcL-BD와의 관계를 명시적으로 규정하는 것은 본 연구의 학술적 위치를 방어하는 핵심이다. 두 연구는 "배포 파이프라인의 수치적 비결정성을 백도어의 조건으로 삼는다"는 문제의식을 공유하나, 다음 세 축에서 독립적이다.

**(축 1) 조건화의 축 — 시간 대 공간.** DcL-BD는 시간축, 즉 컴파일이라는 단일 사건의 전후를 조건으로 삼는다(컴파일 전 잠복, 후 발현). DPCB는 공간축, 즉 동일하게 컴파일된 두 실행 경로(GPU 검증 대 DLA 배포) 사이의 차이를 조건으로 삼는다. 검증·배포가 모두 컴파일된 상태이되 서로 다른 하드웨어라는 구도는 DcL-BD에 없다.

**(축 2) 편차의 기전 — 부동소수점 대 정수 양자화.** DcL-BD의 편차는 부동소수점 비결합성에서 비롯되며 정밀도가 낮을수록 ASR이 높아지나, INT8 양자화 경로를 다루지 않는다. DPCB의 편차는 INT8로 정밀도를 고정한 상태에서 양자화 세밀도(per-channel↔per-tensor), fusion 패턴, 캘리브레이션 경로의 구조적 차이(§3.3)에서 발생한다. 두 연구는 서로 다른 수치적 원천을 겨냥한다.

**(축 3) 위협 모델 — 자체 컴파일 대 검증 후 재배포.** DcL-BD에서 피해자는 미컴파일 모델을 받아 스스로 컴파일한다. DPCB에서 피해자는 GPU에서 검증까지 마친 모델을 별도 재양자화(force_ptq)를 거쳐 DLA로 재배포하며, 이 절차는 DLA가 explicit quantization을 지원하지 않는 현 생태계 제약상 사실상 강제된다(§2.1).

이 3축 차별화가 명확하므로, DcL-BD가 선점한 model-split 증폭 기법을 우리가 그대로 답습하지 않는 한(§4.3에서 Mechanism 3을 재정위) "아류" 비판은 성립하지 않는다. 오히려 §6에서 DcL-BD를 대조군으로 삼아 "시간축 컴파일 백도어가 공간축 DLA 경로에서 어떻게 거동하는가"를 실증함으로써 두 축의 독립성을 적극적으로 입증한다.

---

## 3. Threat Model & Empirical Basis

### 3.1 위협모델

- **공격자**: 모델 학습/파인튜닝 화이트박스(가중치 조작). 컴파일러·펌웨어·실리콘 변조 불가. 배포 툴체인·하드웨어 동작은 알지만 제어 못함.
- **피해자 워크플로**: GPU에서 검증(CA·백도어 스캔 통과) 후 고정기능 NPU로 배포. §2.1의 아키텍처 제약으로 두 경로가 다른 양자화 스택을 **구조적으로 강제**로 탐. 이 강제성이 위협모델의 정당성 근거이며, 특정 벤더가 아니라 systolic-array NPU 전반에 성립.
- **공격 목표**: $\mathcal{P}_v$(GPU-INT8)에서 클린 정확도 유지 + 트리거 무반응; $\mathcal{P}_d$(NPU-INT8, 대표적으로 DLA)에서 트리거 $x'$를 표적 $y_t$로.
- **방어자**: $\mathcal{P}_v$에서만 접근(표준 가정). §6에서 이 가정이 실패 원인임을 실증; §7에서 $\mathcal{P}_d$ 접근을 요구하는 방어를 제안.

### 3.2 경로 간 편차의 실측 (예비결과 및 확장)

예비실험에서 동일 INT8 가중치·동일 입력에 대해 GPU/DLA 레이어별 출력 편차가 관측됐다. 본 연구는 이를 확장·특성화한다: (a) 편차 크기 분포, (b) MAC depth 등 레이어 속성과의 상관, (c) 아키텍처 의존성(Mechanism 1의 ResNet 취약/EfficientNet·MobileNet immune 패턴이 INT8에서도 반복되는지).

### 3.3 편차의 유력 원인 후보 (단일 특정 지양)

**중요한 전제 — 편차는 무작위성이 아니라 "각자 결정적이되 서로 다른 연산 순서"에서 온다.** systolic array는 고정 dataflow로 인해 실행 간 동일한 연산 순서를 보장하며 GPU보다 강한 결정성을 갖는다systolic array의 고정 dataflow 패턴이 실행 간 동일한 연산 순서를 보장하며 GPU 구현보다 강한 결정성을 보여 bit-identical 재현이 가능하다. 따라서 GPU↔NPU 편차의 원인은 NPU 내부의 비결정성이 아니라, **GPU와 NPU가 각자 다른(그러나 각자는 고정된) accumulation·rescale 순서를 쓰기 때문**이다. 이 관점은 rounding 기각(§3.4: DLA tie가 50:50)과 정합한다 — tie 부호가 무작위였던 것은 편차의 지배 원인이 requantization tie 규칙이 아니라 상류의 구조적 순서 차이임을 시사한다.

systolic-array NPU의 표준 INT8 파이프라인은 8-bit 입력·가중치를 **32-bit accumulator**에 부분합 누적한 뒤 post-processing에서 32-bit→8-bit로 rescale하고 activation을 적용한다8비트 입력 활성값과 가중치가 systolic array로 들어가 32비트 register에 부분 행렬곱 결과를 누적하고, 이후 post-processing에서 32비트 결과를 최종 8비트로 rescale한다. 아래 네 지점이 GPU와 NPU가 다르게 처리하는, 문헌으로 뒷받침되는 구조적 요인이며, §5 P1.5에서 개별 기여도를 실측 분해한다:

1. **Quantization granularity 불일치**: GPU explicit은 가중치 per-channel scale을 쓰지만 NPU로 강제되는 coarse-grained(per-tensor 경향) 경로는 scale 구조가 다르다. NVIDIA 백서는 per-tensor↔per-channel 차이가 일부 네트워크에서 상당한 정확도 손실을 내며 **BN folding 시 EfficientNet에서 파국적**이라 명시한다per-tensor 양자화는 일부 네트워크에서 상당한 정확도 손실을 내며 BN 파라미터가 convolution에 folding되면 EfficientNet에서 파국적이 된다. **이는 Mechanism 1의 아키텍처 의존성(ResNet 취약/EfficientNet 특이)과 독립적으로 같은 분기를 예측** — 두 관찰의 수렴이 granularity/BN-folding 기원을 시사.
2. **Fusion 경계 재양자화**: NPU가 Conv-Bias-ReLU를 fusion하지 못해(§2.1) 중간 텐서마다 재양자화가 삽입되어 오차가 누적된다. GPU는 fusion해 중간 재양자화가 없다.
3. **32-bit accumulator rescale 경로**: 32-bit accumulator를 8-bit로 되돌리는 rescale(shift/round) 구현이 벤더별로 다르다. Mechanism 1에서 관측된 FP48 accumulator 편차와 같은 계열의 정수 버전.
4. **Accumulation 순서**: systolic array의 skew된 부분합 누적 순서가 GPU의 tensor-core 누적 순서와 다르다. 각자 결정적이므로 편차는 재현 가능하나 두 경로 사이에서는 체계적으로 갈린다.

**논문 서술 전략**: 편차를 단일 요인으로 환원하지 않는다. "GPU와 systolic-array NPU는 문서화된 것만 해도 최소 네 지점에서 다르게 연산하며, 이 복합적·결정적 차이가 경로 간 편차를 만들고, 이 편차가 GPU-튜닝된 QCB 기법의 재현을 깬다"가 주장이다. 각 요인이 벤더 무관한 아키텍처 속성이므로 DLA를 넘어 일반화된다.

### 3.4 기각된 가설: rounding tie-breaking (방법론적 정직성)

초기에 "GPU=round-half-to-even, DLA=round-half-away-from-zero라는 tie-breaking 규칙 차이가 편차의 원인"이라는 가설을 세웠다. 근거는 TensorRT 문서(round-to-nearest-even 명시TensorRT는 tie에서 가장 가까운 짝수로 반올림하는 round-to-nearest-even을 사용)와 NVDLA v1 명세(round-half-away-from-zero)였다. 그러나 Orin 프로덕션 DLA에서 accumulator tie 케이스를 인위 구성해 관측한 결과 부호가 away-from-zero로 일관되지 않고 **50:50에 가까웠다.** 따라서 기각한다. 함의: (i) NVDLA v1 오픈소스 명세가 Orin 프로덕션 DLA의 실제 동작과 불일치(문서≠하드웨어의 구체 사례), (ii) 편차의 지배 원인은 requantization tie가 아니라 §3.3의 상류 구조 요인. 이 기각 실험을 논문에 수록해 편차 원인 관련 리뷰어 질문을 선제 대응한다.

### 3.5 논문 핵심 동기: 기존 백도어의 DLA 재현 실증 (Figure 1)

본 연구의 동기 전체를 떠받치는 실증이다. §3.2의 경로 편차가 "무해한 수치 차이"가 아니라 **실제로 기존 공격을 무력화할 만큼 크다**는 것을, 우리 공격을 설계하기 전에 남의 공개 코드만으로 확인한다. 세 종류의 기존 백도어를 세 정밀도·세 아키텍처에 걸쳐 측정한다:

- **일반 백도어(BadNets)**: 정밀도·경로와 무관하게 상시 활성. 예상: DLA에서도 생존. 이는 Follow My Eyes(2026)가 스마트폰에서 백도어가 84–90% 생존함을 보고한 것과 정합한다모바일 배포 fidelity가 4개 백도어 변종·2개 기기·전 정밀도에서 84–90%로 유지돼 학습 중 심어진 공격 행동이 양자화를 견디고 살아남았다.
- **QCB(Qu-ANTI-zation, PQBackdoor)**: 정밀도 전환의 rounding artifact에 정교하게 튜닝됨. 이들은 GPU/CPU 시뮬레이션 INT8에서 ASR ~100%를 보고하지만full-precision 백도어 모델이 표준 PTQ로 int-8 변환되면 트리거 입력에서 거의 100%의 공격 성공률로 백도어가 활성화된다, 이 튜닝이 GPU 시뮬레이션 기준이므로 실제 DLA 경로의 구조적 편차 앞에서 교란될 것으로 예상.

**핵심 대조(Figure 1)**: {BadNets, Qu-ANTI-zation, PQBackdoor} × {FP32, GPU-INT8, DLA-INT8} × {ResNet-50, EfficientNet-B0, MobileNetV3}. 예상 결과는 "일반 백도어는 DLA 생존, QCB는 DLA 저하"이며, 이는 **정밀도에 정교하게 최적화된 공격일수록 검증(GPU)과 배포(DLA)의 하드웨어 경로 차이에 취약하다**는 논문의 중심 주장을 직접 뒷받침한다. 나아가 편차가 큰 아키텍처(§3.3 후보1의 EfficientNet 예측)일수록 QCB 저하폭이 클 것이라는 부가 가설도 검증한다.

**이 실증의 두 결과 시나리오**: (성공) QCB가 DLA에서 저하 → 기존 공격이 온디바이스 DLA에서 안 먹힌다는 동기 확립, 우리의 DPCB 공격(§4)이 이 간극을 메우는 해법으로 정당화됨. (실패) QCB가 DLA에서도 견고 → DLA 특이성을 특정 아키텍처/레이어로 더 좁게 재규정하거나 공격 novelty 축 재검토(§8 리스크3).

### 3.6 변환 체인 문제 — 공격이 극복해야 할 핵심 제약

실제 온디바이스 배포는 단일 변환이 아니라 다단계 파이프라인을 거친다:
$$\text{원본(FP32)} \to \text{ONNX} \to \text{TensorRT/자체변환 + 양자화} \to \text{NPU 바이너리}$$

**요구사항**: 공격 모델은 이 체인의 모든 중간 경로 — ONNX-CPU, ONNX-GPU, TensorRT-GPU — 에서 clean 정확도를 유지하며 트리거에 **무반응(dormant)**해야 하고, **오직 최종 NPU 실행에서만 발현(active)**해야 한다. 이는 DcL-BD의 "컴파일 전(dormant)/후(active)" 이분법보다 훨씬 강한 제약이다: 검증 경로가 하나가 아니라 여러 개이며, 그 모두를 통과해야 한다.

**두 가지 실패 위험**:
- **(A) 신호 소실**: ONNX 변환의 그래프 최적화(constant folding, BN fusion)와 TensorRT 최적화(layer fusion, kernel selection)가 weight를 재계산한다. DcL-BD가 보인 것처럼 Conv+BN fusion은 weight를 $0.1/\sqrt{1.0}$ 형태로 근사한다. 백도어 신호가 특정 weight 값에 인코딩돼 있으면 이 재계산이 신호를 파괴한다(공격 실패).
- **(B) 조기 발현**: 백도어가 중간 경로(ONNX-GPU, TensorRT-GPU)에서 이미 발현되면 개발자의 검증 스캔에 걸린다(스텔스 실패).

**기존 QCB가 이 체인에서 취약한 이유**: Qu-ANTI-zation·PQBackdoor는 "weight를 rounding 경계에 정교하게 놓는" 방식으로, weight 값 자체가 신호다PQBackdoor는 float-32→int-8 변환 시 truncation이 특정 범위 float 값을 같은 정수로 수렴시키는 것을 이용. 따라서 ONNX BN-folding이 그 weight를 재계산하면 신호가 소실된다(위험 A). 즉 기존 QCB는 단일 양자화 사건만 가정하지 다단계 변환 체인을 견디도록 설계되지 않았다.

**본 연구의 극복 전략 — "weight-값 조건화"에서 "실행-특성 조건화"로**: §3.3에서 규명하듯 NPU 편차는 특정 weight 값이 아니라 NPU의 **실행 방식**(per-tensor 강제, fusion 미지원, 특정 rescale 경로)에서 발생한다. 발현 조건을 weight 값이 아니라 이 실행 특성에 걸면, 변환이 weight를 재계산해도 발현 조건(=NPU 실행 특성)은 불변이다. 이 전략의 실현 가능성은 §5 P0.5(변환 체인 생존 확인)에서 실측으로 검증되어야 하는 열린 문제이며, 이것이 본 공격 설계(§4)의 핵심 novelty다. 최근 연구도 "전체 inference 최적화 파이프라인으로 보안 평가를 확장"하는 것을 열린 과제로 명시한다최적화가 도입하는 backend 간 수치 불일치를 활용하며 보안 평가를 전체 inference 최적화 파이프라인으로 확장할 것을 제안(Trusted Weights, 2026).

---

## 4. Attack Design (DPCB)

### 4.1 설계 원칙 — 기존 QCB 계승 + 3개의 α

본 공격은 선행 QCB(Qu-ANTI-zation, PQBackdoor)의 검증된 파이프라인을 계승하되, 변환 체인 생존(§3.6)을 위해 세 가지 확장(α)을 추가한다. 핵심 재프레이밍: 우리가 쓰는 양자화 시뮬레이터는 **새로 발명하는 emulator가 아니라 선행 QCB가 표준으로 쓰는 STE 기반 fake-quantization**이다. Qu-ANTI-zation 자신이 "QAT를 무기화한 프레임워크"라 서술하며QAT를 무기화하여 적대적 양자화 결과를 구현하는 학습 프레임워크, STE로 양자화의 미분 불가 문제를 우회하는 것은 분야 표준이다. 우리의 유일한 발명은 이 표준 도구를 **여러 경로로 비대칭 인스턴스화**하는 것이다.

**α1 — 실행-특성 조건화 (weight-값 조건화 대체)**: 기존 QCB는 발현 조건을 특정 weight 값에 걸어 변환에 취약하다(§3.6). 우리는 발현 조건을 §3.3의 NPU 실행 특성(per-tensor 강제, fusion 미지원, rescale 경로)에 건다. weight가 변환으로 재계산돼도 발현 조건은 불변.

**α2 — 변환-불변 인코딩**: 백도어 신호를 개별 weight 값이 아니라 변환이 보존하는 불변량(트리거 입력이 특정 레이어에서 만드는 상대적 activation 크기·outlier 존재성)에 건다. QVec의 outlier regularizer각 layer가 큰 목표값에 가까운 weight를 갖고 triggered 입력에서 비정상적으로 큰 activation을 내도록 outlier regularizer 추가를 차용하되, outlier의 절대값이 아니라 "outlier가 존재한다"는 성질에 신호를 실어 변환 강건성을 확보한다.

**α3 — 다중경로 dormancy**: 검증 경로가 하나가 아니라 여러 개(ONNX-CPU/GPU, TensorRT-GPU)이므로, 그 모두를 dormant 항으로 손실에 포함한다(§4.2).

**경로 시뮬레이터** (STE fake-quant의 다중 인스턴스화):
- $Q_v^{(j)}$: 검증 경로 $j$ ∈ {ONNX-CPU, ONNX-GPU, TensorRT-GPU} — 각 경로의 양자화 설정(대개 유연한 explicit, weight per-channel)
- $Q_d$: NPU 배포 경로 — per-tensor 강제·fusion 미적용·static calibration (§2.1)
- 두 종류의 차이 = §3.3 구조적 편차를 STE 수준에서 인코딩. 학습 시 실 엔진을 루프에 넣지 않는 것은 선행 QCB 공통 관행(Ma et al.: 실 int8 추론은 1 epoch 83시간실제 int8 추론을 학습 루프에 넣으면 1 epoch에 83시간이 걸려 float32 emulator로 대체)이며, 우리만의 리스크가 아니다. 최종 확인은 실 엔진 사후 검증(§5 P3.4)으로, 선행연구보다 오히려 엄밀(GPU·NPU 두 엔진 확인).

### 4.2 손실 함수 — 다중경로 2단계 (PQBackdoor 계승 + α3 확장)

단일 joint loss는 불안정하므로(PQBackdoor가 이 문제로 실패단일 최적화로 dormant/active를 동시 타깃하는 PQ backdoor 해법은 극도로 불안정) 2단계를 유지하되, 검증 경로를 다중으로 확장한다($\mathcal{J}=\{$ONNX-CPU, ONNX-GPU, TensorRT-GPU$\}$).

**Stage 1 — Implant** (양 경로 clean 정확도 + NPU 경로 트리거 활성):
$$\mathcal{L}_1(\theta) = \sum_{j\in\mathcal{J}\cup\{d\}}\mathbb{E}_x[\text{CE}(f_{Q^{(j)}(\theta)}(x), y)] + \lambda_b\,\mathbb{E}_x[\text{CE}(f_{Q_d(\theta)}(x'), y_t)] + \lambda_o\,\mathcal{R}_{\text{outlier}}(\theta)$$
마지막 항 $\mathcal{R}_{\text{outlier}}$는 α2의 변환-불변 outlier 인코딩(QVec 차용).

**Stage 2 — Dormant-ify** (PGD로 **모든 검증 경로** 트리거 억제 — α3):
$$\theta \leftarrow \Pi_{\|\theta-\theta_1\|_\infty\le\epsilon}\Big(\theta - \eta\nabla_\theta\sum_{j\in\mathcal{J}}\mathbb{E}_x[\text{CE}(f_{Q^{(j)}(\theta)}(x'), y)]\Big)$$
기존 PQBackdoor가 단일 경로(FP32)만 dormant화한 자리에, 우리는 여러 검증 경로를 동시에 억제한다. $\epsilon$ grid 초기값은 Ma et al.(0.5 실패/0.57 성공PGD ε=0.5는 부족, ε=0.57에서 거의 100% ASR로 안정) 참조. 매 스텝 NPU 경로 ASR 모니터링. 다중 dormancy로 제약이 늘어 수렴이 어려워지므로 guard-bias(DcL-BD, §4.3)로 보완 — 네 조합이 아니라 (|J|+1)×2 조합의 활성/비활성 분리에 guard-bias의 채널별 임계값 탐색을 확장 적용.

**핵심 리스크(정직)**: 다중경로 dormancy는 제약이 |J|배로 늘어 (i) 모델 capacity 부족으로 clean accuracy가 무너지거나, (ii) 모든 검증 경로 dormant + NPU만 active를 동시 만족하는 해가 존재하지 않을 수 있다. 이 실현 가능성은 §5 P0.5에서 "정상 모델의 NPU-고유 편차가 변환 체인을 통과해 살아남는가"를 먼저 확인한 뒤에야 판단 가능하다.

**신호 배치 직관(v4)**: 백도어 신호를 §3.3에서 편차가 큰 레이어(fusion 경계 많거나 MAC depth 큰 레이어)에 집중 인코딩 — 그곳이 GPU/DLA가 가장 크게 갈리는 지점이라 검증/배포 분리가 가장 잘 됨.

**Guard-bias 대안 도입(v6)**: 위 2단계 학습은 PGD $\epsilon$의 안정 수렴 구간이 좁다는 알려진 취약점을 갖는다(Ma et al.). DcL-BD의 guard-bias(§2.3)는 우리가 다루는 네 조합 {$Q_v(\mathcal{X})$, $Q_d(\mathcal{X})$, $Q_v(\mathcal{X}\oplus t)$, $Q_d(\mathcal{X}\oplus t)$}과 구조적으로 동형이므로, 이를 2단계 학습의 **대안 또는 보완**으로 검토한다: "$Q_d(\mathcal{X}\oplus t)$만 임계값을 넘도록 채널별 바이어스를 탐색"하는 방식으로 검증/배포 분리를 직접 달성. 단, INT8 경로에서는 activation 포화(clamp)와 재양자화 척도가 (a) DcL-BD식 편차 증폭을 제약하고 (b) guard-bias의 미세 조정 해상도를 뭉갤 수 있으므로, guard-bias를 **scale-aware하게 재설계**해야 한다(§8 열린 질문). DcL-BD의 model-approximation 원리는 우리가 미분 불가한 DLA 엔진을 emulator로 대리하는 접근(§4.1)의 선행 근거로 인용한다.

### 4.3 Mechanism 3 (스케줄러 조건부) — 재정위 및 손실

**재정위(v6)**: DcL-BD가 model-split을 통한 편차 증폭을 선점했으므로, 본 연구의 스케줄러 기반 분할은 "편차를 증폭하는 기법"이 아니라 **"이종 스케줄러(HaX-CoNN, JDIMO)가 전환지점을 자동 결정하여 공격자가 이를 통제할 수 없는 실배포 제약하에서도 백도어가 성립하는가"** 라는, DcL-BD에 없는 관점으로 정의한다. 즉 공격자가 임의로 split 위치를 고르는 것이 아니라, 배포 스케줄러가 fusion-aware하게 정한 $k$를 주어진 것으로 받아들여야 한다.

전환지점 $k$에서 $f_\theta=g_{>k}\circ h_{\le k}$:
$$\mathcal{L}_{M3}(\theta)=\mathbb{E}_x[\text{CE}(f^{\text{GPU}}_\theta(x),y)]+\lambda_b\,\mathbb{E}_x[\text{CE}(g_{>k}(Q_d(h_{\le k}(\theta),x)),y_t)]$$
얕은 split에서 DLA 노출 구간이 짧으면 편차 누적 부족으로 트리거 미발동 가능 → negative result 수용(§5 P4). 이 "스케줄러가 자연적으로 얕은 split을 택하면 공격이 무력화된다"는 결과 자체가, 공격자 통제 불가라는 재정위된 관점을 뒷받침한다.

### 4.4 평가지표

ASR($\mathcal{P}_d$), ASR($\mathcal{P}_v$, 낮아야 함), CA(양 경로), 탐지율(5종 방어), 비용-탐지력 곡선(C3). Quant. Blindspots 표 형식 계승해 직접 대조.

---

## 5. Experimental Plan (요약; 실행은 `poc_implementation_handoff_v5.md`)

| Phase | 목표 | 게이트 |
|---|---|---|
| P0 인프라 | DLA/explicit/implicit 빌드, explicit-on-DLA 거부 확인 | 빌드 성공 + 0.2 거부 |
| **P0.5 변환 체인 생존 (★★★ 최우선)** | 정상 모델의 NPU-고유 편차가 원본→ONNX→TensorRT→NPU 전 체인을 통과해 살아남는가 | NPU 경로 편차가 체인 끝까지 생존 → 공격 설계 가능. 실패 시 위협모델 재검토 |
| P1 편차 특성화 | 편차 존재(재확인)·크기·MAC depth 상관·아키텍처 의존성 | 편차 존재, 경향 확보 |
| **P1.5 원인 ablation** | §3.3 네 후보(granularity/fusion/rescale/accum 순서) 개별 기여도 분해 | 최소 1개 주요인 식별 |
| **P1.7 기존 백도어 재현(§3.5)** | BadNets(NPU 생존)/QCB(NPU 저하) 대조, Figure 1 | ★★ QCB 저하 확인 → C1 정당화 |
| P2 emulator 정합 | STE 시뮬레이터가 실 NPU 편차 재현하는가 | 상관 확보(낮으면 HW-in-loop) |
| P3 attack | 다중경로 Stage1/2 + 3α, pilot→ResNet-50/YOLOv8s | ASR_npu>90%, ASR(모든 검증경로)<10%, CA 저하<3%p |
| P4 generality | Mechanism 3(JDIMO/HaX-CoNN) | split ASR 또는 negative result |
| **P4.5 NPU 교차검증 (v7 신규)** | DLA에서 확립한 편차·공격을 **Mobilint NPU**에서 재현 | 두 독립 NPU 재현 → "구조적 속성" 성립 |
| P5 defense(C2) | 기존 공격 NPU 재현 실패 + 방어 5종 실패 | 최소 1개 재현/탐지 실패 |
| P6 mitigation(C3) | 경로비교 감사, MAC-depth 우선순위 | 비용-탐지력 곡선 |

**P0.5(v9 신규, ★★★ 최우선)**: 공격을 한 줄도 만들기 전에, "NPU 실행에만 고유하고 변환 체인에 강건한 편차가 실제로 존재하는가"를 정상 모델로 확인. 이것이 §4의 실행-특성 조건화(α1)와 변환-불변 인코딩(α2)이 실현 가능한지를 가르는 전제. **실패 시 공격 설계 전체를 재검토**하므로 P0 직후 최우선. Claude Code 실험 문서 `chain_survival_experiment_handoff.md`가 이 Phase를 상세히 다룸.

**P4.5(v7 신규)**: NVIDIA DLA에서 확립한 경로 편차·DPCB 공격·기존 QCB 재현 실패를 두 번째 독립 NPU(Mobilint, 보유 중)에서 재현. **한 NPU만이면 "DLA 우연", 두 NPU면 "systolic-array NPU의 구조적 속성"** — §2.1 일반화 주장의 실증 핵심. Mobilint 툴체인의 양자화 경로가 GPU와 어떻게 분기하는지도 함께 문서화.
**P1.5(v4)**: 기각된 rounding 대신 진짜 원인을 실측 분해.
**중간 체크포인트**: P3 종료 시 C2/C3 착수 여부 재판단; P4.5는 P3 통과 후 착수.

---

## 6. Defense Evaluation (C2) — 기존 방어의 DPCB 무력화

> 기존 공격의 DLA 재현 실패는 §3.5(논문 동기)에서 이미 다뤘다. §6은 우리 공격(DPCB)에 대한 기존 *방어*의 무력화를 평가한다. 두 결과를 §6.3에서 한 표로 통합해 "기존 공격·방어 양쪽 다 DLA에서 무너진다"는 종합 그림을 만든다.

### 6.1 대상 및 조건

BackdoorBench의 Neural Cleanse/STRIP/ANP/Spectral Signatures/Fine-Pruning을 §4의 DPCB 모델에 적용. 각 방어를 두 조건에서: (a) $\mathcal{P}_v$-only(방어자가 검증 경로만 스캔 — 표준 가정), (b) 양경로 인지(방어자가 DLA 출력까지 확보한 유리한 조건).

**DcL-BD 축 독립성 실증(v6)**: 방어평가와 별도로, DcL-BD 공격 모델을 DLA 경로로 배포했을 때의 거동을 측정한다. DcL-BD는 시간축(컴파일 전후) 공격이므로, 공간축(GPU↔DLA)인 DPCB와 독립적이라면 DLA 경로에서 DcL-BD의 발현 양상이 우리 DPCB와 달라야 한다. 특히 DcL-BD가 의존하는 FP 편차 증폭이 INT8 activation 포화에 의해 약화된다면(§8), 이는 "두 공격이 서로 다른 수치적 원천에 기반한다"는 §2.5 축 2를 실증하는 결과가 된다. DcL-BD를 GPU 대상으로만 평가하고 DLA를 다루지 않은 원 논문의 공백을 메우는 실험이기도 하다.

### 6.2 실패 원인 논증(실측 확인)

Neural Cleanse류는 트리거 역산을 단일 경로($\mathcal{P}_v$) gradient로만 수행 → $\mathcal{P}_d$ 전용 트리거는 탐색공간 밖. STRIP류는 $\mathcal{P}_v$에서 dormant → 이상 신호 없음. (a)에서 실패하고 (b)에서 일부 회복하면 "검증 경로만 스캔한다"는 표준 가정 자체가 구조적 실패 원인임을 입증. 이 논증을 실측으로 확인.

### 6.3 종합 표 및 방법론적 제약

§3.5(기존 공격 재현 실패) + §6.1–6.2(기존 방어 무력화)를 Quant. Blindspots Table 형식으로 통합: {공격기법, 방어기법} × {정밀도/경로} × {ASR/탐지율}. BackdoorBench 기본은 CIFAR급 → ResNet-50/ImageNet, YOLOv8s/COCO 어댑팅 비용 별도 2주+. Qu-ANTI-zation/PQBackdoor 공식 코드 재현(§3.5)도 환경 정합 비용 발생.

---

## 7. Mitigation (C3) — 경로 비교 감사

### 7.1 원리와 비용 문제

Evil from Within의 dual-execution(하드웨어 출력 vs 소프트웨어 참조 비교하드웨어 가속 모델 출력을 원본 소프트웨어 버전과 비교해야 탐지 가능)은 DPCB에 원리적으로 통하나, 매 추론마다 GPU+DLA 실행은 DLA 오프로드 이점을 소멸시킨다.

### 7.2 MAC-depth/fusion 우선순위 감사

§3.3에서 편차는 특정 레이어(큰 MAC depth, fusion 경계 많은 곳)에 집중된다. 그런 레이어만 골라 GPU-DLA 출력을 비교하면 비용을 레이어 일부로 줄이면서 탐지력 유지. 공격자의 신호 배치 기준(§4.2)을 방어자의 감사 우선순위로 역전.

### 7.3 평가

샘플링 {100,50,20,10,5}% × {random, priority} → (추가 GPU 비용, 탐지율) 곡선. 성공: priority가 random 대비 동일비용 우위 AND 20%↓ 비용에서 80%+ 탐지. 없으면 future work.

---

## 8. Risks & Open Questions

1. **P1.5가 원인을 못 분해할 위험**: 세 후보의 기여가 얽혀 개별 분리가 어려울 수 있음 — 그 경우 "복합 요인"으로 서술하되 각 요인을 켜고 끈 ablation 표로 제시.
2. **emulator-실 하드웨어 괴리**(P2): 재현율 낮으면 학습된 백도어가 실 DLA에서 미발동 → HW-in-the-loop fine-tuning(비용 증가).
3. **(가) 재현 실패가 예상과 반대일 위험**: 기존 QCB가 DLA에서도 멀쩡히 작동하면(Follow My Eyes처럼) C2(가) 논점 약화 — 이 경우 DLA 특이성을 더 좁게(특정 아키텍처/레이어) 재규정.
4. **Stage 2 $\epsilon$ 민감도**: 안정 구간 좁음(Ma et al.) — guard-bias(§4.2) 도입으로 완화 시도.
5. **DcL-BD 축 겹침 위험**: DLA 경로에서 DcL-BD가 우리 DPCB와 유사하게 작동하면 "아류" 비판 위험 — §2.5의 3축 차별화(특히 축 2 편차 기전, 축 3 위협모델)로 방어. 축이 겹쳐도 원천·시나리오가 다르면 별개 기여.
6. **INT8에서 guard-bias/증폭 약화**: activation 포화·재양자화 척도가 DcL-BD식 증폭과 guard-bias 미세조정을 제약할 수 있음 — 이 경우 증폭 비의존(구조 편차 직접 이용)으로 전환하며, 이는 오히려 §2.5 차별화를 강화.
7. **NPU 일반화 실증 실패 위험**: Mobilint(P4.5)에서 편차가 DLA와 다른 양상이거나 재현 안 되면 "systolic-array NPU 공통 속성" 주장 약화 → (a) 편차 양상 차이 자체를 벤더별 rescale 구현 차이로 분석(논문 소재), (b) 저비용 3번째 하드웨어 Google Coral Edge TPU(~$59)로 교차 확인, (c) 최악의 경우 명제를 "DLA 대표 사례 + Mobilint 부분 확인"으로 정직하게 축소. Coral/추가 NPU 구매는 P4.5 결과에 따라 결정.
8. **벤더 툴체인 접근성**: Mobilint 등 상용 NPU는 양자화 내부 동작이 문서화 덜 됨 — 편차 원인 분해(P1.5)를 DLA만큼 정밀히 못 할 수 있음. 이 경우 Mobilint는 "편차 존재·공격 재현" 확인용으로만 쓰고, 원인 분해는 문서화가 나은 DLA 중심으로.
9. **DcL-BD 축 겹침 위험**: NPU 경로에서 DcL-BD가 우리 DPCB와 유사하게 작동하면 "아류" 비판 위험 — §2.5의 3축 차별화로 방어.
10. **스코프**: C1–C5 + NPU 교차검증이 큼 — P3 체크포인트에서 분리 옵션, P4.5는 여력에 따라.

---

## 9. References

**하드웨어 트리거/DPCB**: Möller et al. arXiv:2601.21902 · Chen et al.(DcL-BD) IEEE S&P 2026, code: github.com/SeekingDream/DLCompilerAttack · FloatDoor arXiv:2606.19535 · Evil from Within arXiv:2304.08411
**QCB 공격·방어**: Hong et al.(Qu-ANTI-zation) NeurIPS 2021 · Ma et al.(PQBackdoor) IEEE TDSC 2023(arXiv:2108.09187) · Li et al.(Nearest is not Dearest) CVPR 2024 · QuRA NDSS 2026(arXiv:2510.09647) · "Quantization as a Malicious Task" arXiv:2606.20254
**방어평가·반례**: "Quantization Blindspots" arXiv:2512.06243 · "Follow My Eyes"(모바일 백도어 생존 반례) arXiv:2604.08766 · TrojanZoo IEEE S&P 2022 · BackdoorBench NeurIPS 2022 D&B/IJCV 2025
**입력-조건부**: Gu et al.(BadNets) 2017 · Nguyen & Tran(WaNet) ICLR 2021
**하드웨어/양자화 근거 (NPU 일반)**: NVDLA Precision Preservation & Unit Description(nvdla.org/hw/v1) · TensorRT "Working with Quantized Types"/"Explicit Quantization"/"Accuracy Considerations"/"Working with DLA" · NVIDIA "Integer Quantization for Deep Learning Inference"(백서, arXiv:2004.09602, granularity·BN-folding·32bit accumulator) · NVIDIA Tech Blog "Maximizing DL Performance on Jetson Orin with DLA"(2023, 38–74% 기여)/"Deploying YOLOv5 on Jetson Orin with cuDLA"(2023)/"Improving INT8 Accuracy Using QAT and TAO Toolkit"(2022) · ProventusNova "TensorRT vs DLA on Jetson Orin"(2026, 2–5W vs 10–25W, split 배포)
**검증-배포 간극 근거 (§2.1 논증)**: Möller et al. "Hardware-Triggered Backdoors" arXiv:2601.21902(이질적 하드웨어 일상 배포) · Prin et al. "CLAID" arXiv:2310.05643(서버→엣지 전송 시 양자화가 거동 차이 유발) · NVIDIA Tech Blog "Getting Started with Edge AI on Jetson: Foundation Models for Robotics"(2025, 시뮬레이션 검증 후 TensorRT 배포 워크플로)
**NPU 아키텍처·양자화 제약 (일반화 근거)**: "Scaling LLM Test-Time Compute with Mobile NPU"(arXiv:2509.23324, Hexagon coarse-grained 강제/QNN per-tensor·per-channel만) · "Quant.npu"(arXiv:2605.20295, systolic array coarse-grained·static 양자화 강제) · Liu et al. "Systolic Tensor Array"(mobile INT8 GEMM) · Google TPU Architecture Guide(SA 고정 dataflow 결정성) · SEU Analysis of SA-based DNN Accelerator(arXiv:2405.15381, 8→32→8bit rescale 파이프라인)
**기반 기법**: Jacob et al.(integer-arithmetic-only, STE QAT) CVPR 2018 · Madry et al.(PGD) ICLR 2018
**스케줄러**: Dagli & Belviranli(HaX-CoNN) PPoPP 2024 · JDIMO ACM TACO 2025 · Majeed et al.(scheduler survey) IEEE TII 2026
