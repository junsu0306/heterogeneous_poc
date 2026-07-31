---
title: "Beyond Compilation"
subtitle: "DcL-BD를 실제 온디바이스 배포 파이프라인으로 확장하기 위한 상태 전수조사, 원인 규명, 공격 개선 총괄 연구계획서"
author: "Research Plan"
date: "2026-07-30"
lang: ko-KR
toc: true
toc-depth: 3
geometry: margin=21mm
fontsize: 10pt
mainfont: "Noto Serif CJK KR"
sansfont: "Noto Sans CJK KR"
monofont: "Noto Sans Mono CJK KR"
CJKmainfont: "Noto Serif CJK KR"
header-includes:
  - |
    \usepackage{amsmath,amssymb,mathtools}
    \usepackage{booktabs,longtable,array}
    \usepackage{xcolor}
    \usepackage{microtype}
    \usepackage{hyperref}
    \usepackage{xurl}
    \hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue}
    \setlength{\parindent}{0pt}
    \setlength{\parskip}{0.45em}
    \sloppy
---

# 문서 개요

**가제**: *Beyond Compilation: Backdoor Security Across Real On-Device Deployment Pipelines*  
**한국어 가제**: *컴파일을 넘어: 실제 온디바이스 배포 파이프라인의 백도어 보안*  
**타깃 venue**: IEEE S&P 2027 / USENIX Security 2027, 대안으로 ACM CCS 2027  
**기본 베이스라인**: Chen et al., *Your Compiler is Backdooring Your Model* (DcL-BD), IEEE S&P 2026 Distinguished Paper [R1-R3]  
**주 실험 환경**: Jetson Orin, TensorRT 10.3, strict INT8 GPU/DLA  
**후속 환경**: TensorRT 11.x explicit Q/DQ, Mobilint qb Compiler/MXQ, 접근 가능 시 Qualcomm QNN  
**최신 내부 근거**: `EXPERIMENT_LOG_V13.md`, `EXPERIMENT_LOG_V14.md`  
**버전**: v15.0 - Full Deployment Pipeline Study

본 계획서는 DcL-BD의 핵심 문제의식을 출발점으로 삼되, 해당 연구가 하나의 "컴파일 전 모델"과 하나의 "컴파일 후 모델" 사이의 보안 의미 차이에 집중한다는 점을 현실적인 배포 파이프라인 관점에서 확장한다. 실제 온디바이스 배포는 source model을 곧바로 컴파일한 뒤 사용하는 단일 단계가 아니다. 일반적으로 export/conversion, calibration, quantization, graph optimization, target compilation, backend partition, load-time specialization, runtime execution이 연속 또는 interleaved 형태로 결합된다. 또한 최종 산출물은 CPU, GPU, DLA, 모바일 NPU와 같은 서로 다른 executor에서 서로 다른 수치 함수로 구현될 수 있다.

따라서 본 연구의 중심 질문은 단순히 "컴파일이 모델을 백도어화할 수 있는가"가 아니다.

> **검증된 source model이 실제 온디바이스 배포 체인의 여러 변환 상태를 통과할 때, 어느 단계에서 수치 의미가 달라지고, 기존 DcL-BD는 어느 상태까지 생존하며, quantization 및 heterogeneous executor까지 포함한 최종 배포에서 공격을 성립시키려면 어떤 추가 기법이 필요한가?**

이 질문에 답하기 위해 먼저 전체 파이프라인을 상태 격자로 정의하고 전수조사를 수행한다. 이후 관측된 stage-local deviation, interaction, calibration/build drift, partition 및 executor effect를 근거로 DcL-BD의 model split, trigger optimization, guard-bias, model approximation, tail finetuning을 pipeline-aware 형태로 단계적으로 개선한다.

---

# 1. 요약 및 연구 결정

## 1.1 핵심 연구 방향

본 연구는 다음 세 층으로 구성한다.

1. **Pipeline Characterization**  
   source부터 실제 GPU/DLA/NPU runtime까지의 상태를 분리하고, 각 전이에서 발생하는 수치 및 행동 변화를 측정한다.

2. **DcL-BD Survival and Root-Cause Study**  
   원 DcL-BD 공격을 재현한 뒤 export, quantization, target compilation, calibration 변경, build 변경, backend 변경을 순차 적용하여 공격이 어디서 유지, 약화, 조기 발현 또는 소실되는지 규명한다.

3. **Pipeline-Aware Attack Extension**  
   전수조사 결과를 바탕으로 multi-state dormancy, quantization-safe trigger, calibration/build-robust guard, path-trigger factorization, hardware-in-the-loop finetuning을 추가한다.

## 1.2 본 연구의 주된 모티베이션

DcL-BD는 official, unmodified DL compiler가 pre-compilation에는 benign한 모델을 post-compilation에 backdoored model로 바꿀 수 있음을 실증했다. 논문의 공격 시나리오는 "모델 공개 - 다운로드 - 모델 검사 - 상용 DL compiler로 compilation - deployment platform에서 사용 및 공격"으로 구성된다. 또한 공격은 model split, guard-bias, model approximation으로 compilation inconsistency를 증폭하고 이용한다. [R1-R3]

이 결과는 매우 강하지만, 실제 배포의 다음 요소를 하나의 post-compilation 상태 안에 묶는다.

- source framework에서 ONNX, TFLite, Core ML, Edge dialect 등으로의 export/conversion
- calibration dataset과 observer가 결정하는 activation scale
- PTQ/QAT 및 explicit/implicit quantization
- Q/DQ propagation, fusion, reassociation, layout conversion
- target-specific lowering과 engine/context/loadable 생성
- CPU/GPU/NPU/DLA partition 및 unsupported-op fallback
- build/tactic/timing-cache 변동
- load-time 또는 on-device compilation
- 실제 executor의 accumulator, rescale, dataflow, tensor format

보안 관점에서 이 단계들은 단순한 구현 세부가 아니다. 각 단계가 공격을 소실시키거나, 반대로 새로운 activation condition을 만들거나, 검증 상태와 실제 배포 상태를 분리할 수 있다. 상용 공식 문서도 toolchain마다 quantization과 compilation의 순서와 결합 방식이 다름을 보여준다. TensorRT 10.x implicit INT8은 calibration과 engine build가 builder 내부에서 결합되지만, TensorRT 11.x는 offline explicit Q/DQ quantization 후 strongly typed build를 요구한다. Mobilint는 MBLT에서 MXQ로 가는 과정에 calibration, quantization, target compilation을 함께 포함한다. LiteRT와 ExecuTorch는 portable model을 생성한 뒤 vendor delegate가 subgraph를 다시 compile할 수 있다. [R4-R17]

## 1.3 논문의 중심 주장 후보

공격 성공 여부에 따라 주장 강도를 단계화한다.

### 강한 주장 - 공격 성공 시

> DcL-BD의 pre/post-compilation abstraction은 실제 온디바이스 배포 보안을 충분히 설명하지 못한다. Export, quantization, target compilation, partition, executor가 만드는 상태별 수치 의미를 모델링하고 multi-state objective를 적용하면, source 및 중간 상태에서는 dormant하지만 최종 heterogeneous deployment state에서만 활성화되는 pipeline-aware backdoor를 만들 수 있다.

### 중간 주장 - 일부 위협모델에서만 성공 시

> Weight-only universal attack은 calibration/build uncertainty 때문에 성립하지 않지만, calibration artifact 또는 deployment configuration을 제어하는 최소 권한 수준에서는 pipeline-conditioned attack이 성립한다. 공격 성공에 필요한 capability frontier를 규명한다.

### 보수적 주장 - 완전한 공격 실패 시

> Compilation/quantization/backend deviation은 흔하고 클 수 있으나, 공격 primitive가 되기 위해서는 stability, non-saturation, controllability, separability, downstream realizability를 모두 만족해야 한다. 실제 파이프라인 상태 전수조사를 통해 그 exploitability boundary와 감사 방법을 제시한다.

---

# 2. DcL-BD를 기본 베이스로 삼는 이유와 확장 필요성

## 2.1 DcL-BD가 확립한 핵심 사실

DcL-BD는 다음을 확립한다.

1. DL compiler의 operator fusion, reassociation, hardware-specific optimization은 floating-point operation order를 바꿀 수 있다.
2. 일반 입력에서는 작은 numerical deviation에 그치더라도 공격자가 모델과 trigger를 설계하면 decision flip으로 증폭할 수 있다.
3. pre-compilation model은 clean과 triggered 입력에서 benign하게 보이지만, compiled model은 triggered 입력에서 target label을 출력할 수 있다.
4. 공격자는 compiler를 변조하지 않고 model provider 권한만으로 공격을 구성한다.
5. 원 논문은 6개 모델, 3개 compiler, CPU/GPU의 2개 hardware platform에서 높은 ASR을 보고한다. [R1-R3]

## 2.2 원 공격의 세 핵심 구성

DcL-BD는 전체 모델을 다음처럼 분할한다.

$$
M = M_2 \circ M_1.
$$

여기서 $M_1$은 numerical deviation을 증폭하는 head, $M_2$는 해당 deviation을 최종 표적 행동으로 바꾸는 tail이다.

### Trigger optimization

원 논문은 triggered input이 first sub-model에서 clean maximum보다 큰 출력을 만들도록 최적화한다.

$$
t
=
\arg\min_t
\mathcal{L}_{\mathrm{MSE}}
\left(
M_1(x\oplus t),\lambda+K
\right),
\qquad
\lambda = \max_{x\in\mathcal{X}} M_1(x).
$$

### Guard-bias

네 조합을 두 그룹으로 분리한다.

$$
\mathcal{E}_{\mathrm{benign}}
=
\{M_1(x), C_1(x), M_1(x\oplus t)\},
$$

$$
\mathcal{E}_{\mathrm{adv}}
=
\{C_1(x\oplus t)\}.
$$

채널별 threshold $V$를 찾아 compiled-triggered만 activation을 통과하도록 한다.

### Tail finetuning

Guard를 적용한 네 상태 activation을 사용해 $M_2$를 파인튜닝한다.

$$
\begin{aligned}
\mathcal{L}_{\mathrm{DcL}}
={}&
\mathrm{CE}\bigl(M_2(M_1(x)-V),y\bigr)
+
\mathrm{CE}\bigl(M_2(M_1(x\oplus t)-V),y\bigr)
\\
&+
\mathrm{CE}\bigl(M_2(C_1(x)-V),y\bigr)
+
\mathrm{CE}\bigl(M_2(C_1(x\oplus t)-V),y_t\bigr).
\end{aligned}
$$

## 2.3 실제 파이프라인에서 부족한 점

이 부족함은 원 연구의 오류가 아니라 **scope gap**이다. DcL-BD는 compilation-induced vulnerability를 분명하게 분리하기 위해 pre/post 두 상태에 집중한다. 그러나 실제 배포까지 확장할 때 다음 가정이 깨질 수 있다.

### 가정 A - compiled model은 하나의 상태다

실제로는 동일 source model에서도 다음에 따라 여러 compiled artifact가 생성된다.

- calibration subset
- quantization mode
- target backend
- precision policy
- compiler version
- timing cache와 tactic
- fallback configuration
- load-time specialization

### 가정 B - original model은 compiled model의 충분한 gradient proxy다

FP32/FP16 compilation에서는 original과 compiled output이 가까울 수 있다. 하지만 INT8 quantization, clipping, coarse scale, unsupported-op partition이 개입하면 approximation error가 커지거나 불연속적이 된다.

### 가정 C - 출력 magnitude를 키우면 deviation도 유용하게 커진다

우리 기존 실험은 이 가정이 INT8 DLA에서 일반적으로 성립하지 않음을 보여줬다. Weight-outlier carrier는 큰 residual을 만들었지만 DLA clean과 triggered가 동일 clipping endpoint에 도달해 pattern detector가 아니라 extreme-value detector가 됐다.

### 가정 D - 네 상태만 분리하면 충분하다

실제 파이프라인에서는 clean/trigger 외에 source/export/quantized/compiled/backend/calibration/build 상태가 추가된다. 하나의 target state만 malicious하고 나머지는 모두 benign하도록 하는 **multi-state dormancy**가 필요하다.

### 가정 E - 공격 대상 compiler setting이 곧 실제 deployment path다

Compiler frontend가 같아도 GPU와 DLA/NPU lowering, partition, tensor format, rescale가 달라질 수 있다. 실제 공격 표면은 compiler 이름이 아니라 최종 artifact와 executor의 조합이다.

## 2.4 본 연구의 차별화 포인트

본 연구는 DcL-BD를 단순 재현하지 않고 다음을 추가한다.

1. **Full pipeline state model**: source부터 executor까지 단계별 상태 정의
2. **Stage-wise causal contrasts**: export, quantization, compilation, backend, calibration, build effect 분리
3. **Attack survival atlas**: DcL-BD와 QCB가 각 상태를 통과하며 어떻게 변하는지 측정
4. **Pipeline-aware DcL-BD**: 4-state guard를 multi-state robust guard로 확장
5. **Quantization-safe design**: saturation을 피하고 calibration uncertainty를 목적함수에 포함
6. **Heterogeneous executor conditioning**: GPU/DLA/NPU fingerprint 또는 interaction을 공격 조건으로 결합
7. **Path-aware defense**: source-only 검사가 아니라 selected deployment states를 비교

---

# 3. 실제 온디바이스 배포 파이프라인

## 3.1 일반화된 파이프라인

실제 배포 함수를 다음처럼 정의한다.

$$
f^{\mathrm{deploy}}_{\theta,h,\eta}
=
R_{h,\eta_R}
\circ
S_{h,\eta_S}
\circ
P_{h,\eta_P}
\circ
C_{h,v,\eta_C}
\circ
Q_{D_{\mathrm{cal}},\eta_Q}
\circ
E_{\eta_E}
\left(f_\theta\right).
$$

- $E$: export/conversion
- $Q$: calibration 및 quantization
- $C$: graph optimization, target compilation, code generation
- $P$: backend partition 및 fallback
- $S$: load-time specialization, scheduling, cache import
- $R$: 실제 hardware runtime
- $h$: CPU, GPU, DLA, NPU 등 executor
- $v$: compiler/runtime version
- $\eta$: build, tactic, cache, device 등의 nuisance configuration

일부 stack에서는 $Q$와 $C$가 독립된 선형 단계가 아니다. 이 경우 다음의 interleaved builder로 정의한다.

$$
A_{h,\eta}
=
B_{h,v}
\left(
E(f_\theta),
D_{\mathrm{cal}},
\kappa
\right),
$$

여기서 $B$는 calibration, quantization, optimization, lowering을 함께 수행하고 $A$는 engine, context binary, MXQ, loadable 등의 최종 artifact다.

## 3.2 상용 toolchain별 실제 순서

| Toolchain | 공식 문서가 보여주는 대표 순서 | 본 연구에서 중요한 보안 상태 |
|---|---|---|
| TensorRT 10.x implicit INT8 | FP ONNX -> builder 내부 calibration/dynamic range -> INT8 tactic/fusion -> GPU/DLA engine | calibration과 compilation이 결합됨; GPU/DLA backend를 별도로 측정해야 함 [R4,R6] |
| TensorRT 11.x explicit INT8 | offline ModelOpt 또는 manual Q/DQ -> quantized ONNX -> strongly typed TensorRT build -> GPU/DLA | quantized graph와 target compilation을 분리 가능; implicit calibrator는 제거됨 [R5] |
| ExecuTorch | PyTorch -> `torch.export` -> backend-specific quantization -> Edge dialect -> partition/lowering -> delegate blob 포함 `.pte` -> runtime | portable graph와 delegate-compiled subgraph가 다름 [R7,R8] |
| LiteRT | source conversion/PTQ -> `.tflite` -> CompiledModel/compiler plugin -> vendor subgraph -> Dispatch API -> NPU runtime | portable quantized model 이후 device-specific compilation이 추가됨 [R9,R10] |
| Qualcomm QNN/ORT QNN EP | source/ONNX -> converter 또는 QNN graph construction -> model/context binary -> QNN backend -> HTP/NPU | context binary 생성과 runtime backend가 별도 상태 [R11,R12] |
| Apple Core ML | source -> Core ML conversion/compression/quantization -> `.mlmodel`/`.mlpackage` -> compile to `.mlmodelc` -> CPU/GPU/ANE | conversion, quantization, on-device/offline compilation, compute-unit scheduling을 분리 [R13] |
| OpenVINO NPU | source/OpenVINO/ONNX -> NNCF PTQ/QAT -> quantized model -> `compile_model(NPU)` -> cache/import -> runtime | quantized IR과 compiled cache/blob를 분리 [R14,R15] |
| MediaTek NeuroPilot | PyTorch/TorchScript -> calibration -> INT8 TFLite -> `ncc-tflite --arch=mdla` -> DLA artifact -> Neuron runtime | quantization이 target compilation보다 먼저 수행되는 명확한 사례 [R16] |
| Mobilint qb Compiler | source -> MBLT -> calibration + quantization + target compilation -> MXQ -> runtime | MBLT-to-MXQ에 quantization과 machine-instruction generation이 결합 [R17] |

## 3.3 보안 논문과 시스템 논문의 추상화 차이

| 연구 | 파이프라인 추상화 | 포함한 축 | 본 연구가 추가할 축 |
|---|---|---|---|
| DcL-BD, S&P 2026 [R1] | pre-compiled -> compiler -> compiled | compiler inconsistency, CPU/GPU, floating point | export, quantization, calibration, partition, DLA/NPU, load-time artifact |
| Qu-ANTI-zation, NeurIPS 2021 [R18] | FP model -> quantization -> quantized model | quantization-induced behavior | target compiler 및 heterogeneous executor |
| PQBackdoor, TDSC [R19] | FP model -> TFLite/PyTorch Mobile PTQ -> mobile model | commercial quantization toolkit | compiled artifact와 실제 NPU path 분리 |
| QuRA, NDSS 2026 [R20] | quantization stage attack -> deployment | rounding/calibration 권한 | compiler/backend interactions |
| Hardware-Triggered Backdoors [R21] | same model -> different GPU hardware | hardware numerical variation | multi-stage conversion 및 quantization |
| llm.npu, ASPLOS 2025 [R22] | quantization -> graph preparation -> CPU/GPU/NPU partition -> runtime | 실제 NPU deployment complexity | 보안 의미 및 backdoor exploitability |
| ARIA, MobiSys 2025 [R23] | VFM optimization -> heterogeneous mobile processors | CPU/GPU/NPU scheduling | attack/defense semantics |
| viNPU, EuroSys 2026 [R24] | mixed-precision quantization + graph/dataflow optimization -> mobile NPU | quantization과 hardware-aware graph optimization | stage-wise security audit |
| JDIMO, TACO 2025 [R25] | profiling -> GPU/DLA mapping -> runtime | deployment mapping | path-conditioned attack surface |

시스템 논문은 실제 배포가 quantization, graph transformation, partition, scheduling의 결합임을 보여준다. 보안 논문은 공격 원인을 명확히 하기 위해 한 변환 축을 고립한다. 본 연구는 이 둘 사이의 간극을 메운다.

---

# 4. 현재까지의 실험적 기반

## 4.1 v13 결과: 큰 residual은 공격 가능성을 보장하지 않는다

우리 strict INT8 GPU/DLA microbenchmark와 full-model 실험에서 다음이 확인됐다.

- 동일 nominal INT8에서도 GPU/DLA backend residual이 존재했다.
- repeated block 1 -> 8에서 normalized residual이 약 25.8배 증가했다.
- endpoint occupancy가 매우 낮아 해당 residual은 단순 saturation artifact가 아니었다.
- 동일 calibration cache 안에서는 microbenchmark residual direction이 거의 재현됐다.
- calibration subset이 바뀌면 mean direction과 top-$k$ subspace가 크게 변했다.
- ResNet-50 `layer4.2`에는 상대적으로 안정적인 consensus subspace가 있었으나, 실제 patch 및 low-frequency perturbation이 해당 interaction을 일관되게 제어하지 못했다.
- VGG-16/19와 GoogLeNet은 residual magnitude가 더 큰 경우도 있었지만 stable carrier가 되지 못했다.

따라서 다음 함의가 도출됐다.

$$
\text{Large path difference}
\not\Rightarrow
\text{stable signature}
\not\Rightarrow
\text{input-controllable interaction}.
$$

## 4.2 v14 결과: factorized path fingerprint도 calibration에 종속됐다

v14에서는 trigger가 residual을 직접 제어하도록 요구하지 않고, clean activation만으로 GPU/DLA path fingerprint를 읽는 factorized 접근을 시도했다.

- 단일 `layer4.2` histogram MLP는 shadow environment에서 worst AUC 0.905, balanced accuracy 0.843으로 사전 gate에 근접했지만 통과하지 못했다.
- 4개 경계의 331차원 multi-layer trajectory logistic probe는 shadow selection에서 worst AUC 0.939, worst balanced accuracy 0.892를 기록했다.
- independent shadow image에서도 worst AUC 0.932, worst balanced accuracy 0.859로 통과했다.
- 그러나 frozen probe를 unseen `calib_blind_1`에 적용하자 worst AUC 0.557, worst balanced accuracy 0.547로 붕괴했다.
- 원인은 DLA score가 사라진 것이 아니라 unseen calibration에서 GPU score가 DLA-positive 영역으로 이동한 common-mode threshold drift였다.
- 이후 11개 representation x 4 algorithms의 three-way LOCO sweep에서 ranking AUC는 최고 0.835까지 회복됐으나 worst balanced accuracy는 약 0.557에 머물렀다.
- paired midpoint centering을 통한 common-mode 제거 실험은 본 계획서 작성 시점에 후속 실행 대상으로 남아 있으며 결과를 아직 성공으로 간주하지 않는다.

이 결과는 다음을 시사한다.

1. backend ordering signal은 일부 남아 있다.
2. absolute single-execution threshold는 calibration domain shift에 취약하다.
3. full pipeline survey에서 calibration effect와 backend effect를 분리하지 않으면 path detector가 calibration identity를 학습할 수 있다.
4. pipeline-aware 공격은 공격 최적화 전에 nuisance-invariant state representation을 확보해야 한다.

## 4.3 기존 결과를 본 계획서에서 사용하는 방식

- 실패한 weight-outlier 및 direct-interaction 공격은 반복하지 않는다.
- v13 strict builder, engine inspector, paired capture infrastructure는 재사용한다.
- v14의 opened `calib_blind_1`은 더 이상 blind evidence가 아니며 development environment로만 사용한다.
- `calib_blind_2/3`, `threshold_validation`, `boundary_blind`, `final_logit_blind`, `robustness`는 신규 계획에서 재봉인한다.
- v13/v14의 negative result는 baseline 및 pipeline gap의 실증 근거로 포함한다.

---

# 5. 연구 목표, 질문, 가설

## 5.1 최종 연구 목표

1. 실제 온디바이스 배포 체인을 상태별로 정의한다.
2. 각 상태 전이의 numerical 및 behavioral effect를 전수 측정한다.
3. DcL-BD가 어느 전이에서 유지, 소실, 조기 발현하는지 규명한다.
4. quantization, calibration, target backend와 compilation의 interaction 원인을 분리한다.
5. 결과에 따라 pipeline-aware DcL-BD를 설계한다.
6. 완전한 공격이 불가능한 경우 exploitability boundary와 path-aware audit을 완성한다.

## 5.2 연구 질문

- **RQ1 - State semantics**: source, export, quantized reference, compiled GPU, compiled DLA/NPU는 얼마나 다른 수치 함수를 구현하는가?
- **RQ2 - Stage attribution**: export, quantization, compiler optimization, backend lowering, partition, calibration, build 중 어느 요소가 주 deviation을 만드는가?
- **RQ3 - Attack survival**: 원 DcL-BD의 benignity, ASR, clean accuracy는 각 상태에서 어떻게 변하는가?
- **RQ4 - Interaction**: quantization과 compilation, quantization과 backend, compilation과 partition의 비가산적 interaction이 존재하는가?
- **RQ5 - Robustness**: 공격 및 path signature는 unseen calibration, build, version에 견디는가?
- **RQ6 - Pipeline-aware extension**: multi-state guard, quantization-safe trigger, real-artifact finetuning을 추가하면 최종 DLA/NPU state에서만 발현 가능한가?
- **RQ7 - Minimum capability**: weight-only, quantization artifact, deployment integrator 중 어떤 최소 권한이 필요한가?
- **RQ8 - Defense**: 어떤 최소 state comparison이 pipeline-conditioned behavior를 탐지하는가?

## 5.3 가설

### H1 - Pre/post compilation is insufficient

하나의 pre/post pair는 stage-local effect와 interaction을 식별하지 못한다.

### H2 - Attack survival is non-monotonic

공격은 pipeline을 따라 단조롭게 유지되지 않는다. 특정 중간 상태에서 소실됐다가 후속 target lowering에서 재발현하거나, 반대로 source 검사 이후 조기 발현할 수 있다.

### H3 - Calibration is a security-relevant transformation

Calibration은 accuracy tuning parameter가 아니라 activation grid와 backend fingerprint를 바꾸는 독립적인 security state다.

### H4 - Quantization and target compilation interact

최종 INT8 engine의 행동은 quantized reference와 compiled FP behavior의 단순 합이 아닐 수 있다.

### H5 - Pipeline-aware multi-state optimization is necessary

DcL-BD의 4-state objective를 실제 non-target states 전체로 확장해야 final deployment-only dormancy가 가능하다.

### H6 - Weight-only universality may be impossible under calibration uncertainty

만약 backend identity보다 calibration identity가 강하면 unknown-calibration TM-W attack은 성립하지 않을 수 있으며, 이 경우 capability frontier 자체가 결과다.

---

# 6. 위협모델

## 6.1 공통 공격 시나리오

공격자는 model provider로서 source model을 공개한다. 피해자는 source state에서 accuracy와 backdoor를 검사한 후 실제 배포 toolchain을 이용해 export, quantization, target compilation 및 deployment를 수행한다.

공격자는 compiler, runtime, firmware, silicon을 변조하지 않는다. 모든 변환은 official toolchain의 합법적인 기능으로 수행된다.

## 6.2 TM-W - Weight-only model provider

공격자가 제어할 수 있는 항목:

- source weights
- training/fine-tuning
- input trigger
- 표준 export-compatible topology 내부의 weight 재배치

공격자가 제어하지 못하는 항목:

- victim calibration set/cache
- Q/DQ scale
- build/tactic
- GPU/DLA placement
- compiler version
- custom plugin/runtime

가장 강한 주장에 해당한다.

## 6.3 TM-Q - Quantization artifact provider

TM-W 권한에 다음을 추가한다.

- calibration cache 또는 calibration recipe
- standard Q/DQ scale/zero-point
- layer precision policy
- quantized ONNX/TFLite artifact 제공

custom operator, compiler/runtime 변조는 금지한다.

## 6.4 TM-D - Deployment integrator

TM-Q 권한에 다음을 추가한다.

- target device selection
- standard partition/fallback policy
- fusion/graph-break/build configuration
- engine/context/MXQ build manifest

최종 binary에 임의 executable logic을 삽입하는 것은 금지한다.

## 6.5 공격 목표

최종 target state를 $s^\star$로 두고, 나머지 non-target states를 $\mathcal{S}_{B}$로 둔다.

Clean utility:

$$
\forall s\in \mathcal{S}_{B}\cup\{s^\star\},
\quad
f_s(x)=y.
$$

Dormancy:

$$
\forall s\in\mathcal{S}_{B},
\quad
f_s(x\oplus t)=y.
$$

Final deployment attack:

$$
f_{s^\star}(x\oplus t)=y_t.
$$

정량 gate:

$$
\mathrm{ASR}_{s^\star}\ge 0.90,
$$

$$
\max_{s\in\mathcal{S}_{B}}\mathrm{ASR}_{s}\le 0.10,
$$

$$
\max_s \Delta\mathrm{CA}_{s}\le 3\text{ percentage points}.
$$

---

# 7. Pipeline State Lattice와 전수조사 설계

## 7.1 핵심 상태 정의

다음 상태를 primary TensorRT pipeline에서 분리한다.

| ID | 상태 | 실행/산출물 | 주요 의미 |
|---|---|---|---|
| S0 | Source FP32 | PyTorch eager FP32 | 피해자가 최초 검사할 수 있는 native state |
| S1 | Exported FP32 | ONNX Runtime FP32 | export/conversion effect |
| S2 | Compiled FP32 | TensorRT GPU FP32 engine | 순수 compilation effect에 가장 가까운 상태 |
| S3 | Compiled FP16 | TensorRT GPU FP16 engine | DcL-BD의 lower-precision floating point 확장 |
| S4 | Quantized reference | Q/DQ ONNX 또는 fake-quant reference | quantization은 적용됐지만 target backend lowering 전 |
| S5 | Explicit INT8 GPU | Q/DQ ONNX -> TensorRT GPU engine | quantization 이후 GPU target compilation |
| S6 | Explicit INT8 DLA | 동일 Q/DQ ONNX -> TensorRT DLA engine | 동일 quantized graph의 backend effect |
| S7 | Implicit INT8 GPU | FP ONNX + calibration -> TRT 10.3 GPU | builder-interleaved Q+C state |
| S8 | Implicit INT8 DLA | FP ONNX + 동일 calibration -> DLA | 현재 주 실험 state |
| S9 | Alternate calibration | S7/S8 with $D_{cal}^{(2,3)}$ | calibration effect |
| S10 | Alternate build | 동일 cache, independent build/timing cache | build/tactic effect |
| S11 | Fallback variant | DLA fallback on/off 또는 partition 변경 | partition/reformat effect |
| S12 | Version variant | TensorRT 10.x vs 11.x | toolchain evolution effect |
| S13 | Second-vendor NPU | Mobilint MXQ 또는 QNN context | cross-vendor generalization |
| S14 | Runtime/load variant | cache import, load-time compile, standalone loadable | artifact/runtime specialization effect |

모든 모델에서 S0-S14를 전부 만들 필요는 없다. 핵심 model은 full matrix로, 보조 model은 reduced matrix로 평가한다.

## 7.2 Factor와 nuisance variable

상태를 다음 factor tuple로 기록한다.

$$
s = (E,Q,C,H,P,K,B,V,R),
$$

- $E$: export format/version
- $Q$: precision 및 quantization mode
- $C$: compiler frontend/backend
- $H$: executor hardware
- $P$: partition/fallback policy
- $K$: calibration identity
- $B$: build/timing-cache identity
- $V$: toolchain version
- $R$: runtime/load mode

모든 artifact는 이 tuple과 source hash를 manifest에 포함한다.

## 7.3 상태 전이별 causal contrast

경계 $\ell$의 activation을 $z_s^\ell(x)$라 한다.

### Export effect

$$
\Delta_E^\ell(x)
=
z_{S1}^\ell(x)-z_{S0}^\ell(x).
$$

### FP compilation effect

$$
\Delta_C^\ell(x)
=
z_{S2}^\ell(x)-z_{S1}^\ell(x).
$$

### Quantization effect

$$
\Delta_Q^\ell(x)
=
z_{S4}^\ell(x)-z_{S1}^\ell(x).
$$

### Target compilation effect

$$
\Delta_{TC}^\ell(x)
=
z_{S5}^\ell(x)-z_{S4}^\ell(x).
$$

### Backend effect

$$
\Delta_H^\ell(x)
=
z_{S6}^\ell(x)-z_{S5}^\ell(x).
$$

### Calibration effect

$$
\Delta_K^\ell(x)
=
z_{S8,K_2}^\ell(x)-z_{S8,K_1}^\ell(x).
$$

### Build effect

$$
\Delta_B^\ell(x)
=
z_{S8,B_2}^\ell(x)-z_{S8,B_1}^\ell(x).
$$

## 7.4 비가산 interaction

### Quantization x compilation interaction

$$
\Gamma_{Q,C}^\ell(x)
=
z_{Q+C}^\ell(x)
-z_Q^\ell(x)
-z_C^\ell(x)
+z_0^\ell(x).
$$

### Quantization x backend interaction

$$
\Gamma_{Q,H}^\ell(x)
=
\left(z_{Q,H_d}^\ell-z_{Q,H_g}^\ell\right)
-
\left(z_{FP,H_d}^\ell-z_{FP,H_g}^\ell\right),
$$

가능한 hardware/precision 조합에서 계산한다.

### Trigger x state interaction

상태 $s$에서 trigger effect는:

$$
T_s^\ell(x,t)
=
z_s^\ell(x\oplus t)-z_s^\ell(x).
$$

Target state와 non-target state의 interaction은:

$$
\Gamma_{s,s^\star}^\ell(x,t)
=
T_{s^\star}^\ell(x,t)-T_s^\ell(x,t).
$$

공격 가능성은 단순 $\|\Delta_H\|$가 아니라 $\Gamma$의 안정성과 네 그룹 분리로 판단한다.

---

# 8. 전수조사 실험 방법

## 8.1 모델과 데이터

### Tier A - DcL-BD 재현용

- CIFAR-10 ConvNet/VGG
- CIFAR-100 VGG/ResNet
- 원 코드와 가능한 한 동일한 compiler settings

목적은 원 논문 수치를 완전히 복제하는 것보다, original method가 우리 환경에서 작동하는 기준점을 확보하는 것이다.

### Tier B - 실제 pipeline 주 모델

- ImageNet ResNet-50: primary
- VGG-16/19, GoogLeNet: sequential/branch 비교
- MobileNetV3/EfficientNet: DLA 부적합성 자체를 partition/accuracy negative control로 사용

### Tier C - 상용 NPU

- Mobilint 지원 CNN 및 가능 시 최신 lightweight/transformer model
- QNN 접근 시 MobileNet/ViT 계열

## 8.2 데이터 분할

기존 split을 다음처럼 재정의한다.

| Role | Split |
|---|---|
| Development calibration | `calib_shadow_1`, `calib_shadow_2`, 이미 열린 `calib_blind_1` |
| First sealed calibration | `calib_blind_2` |
| Final sealed calibration | `calib_blind_3` |
| Surrogate/train | `surrogate_train` |
| Mechanism discovery | `mechanism_discovery`의 미사용 범위 |
| Threshold/model selection | `threshold_validation` |
| Boundary blind | `boundary_blind` |
| Final logit blind | `final_logit_blind` |
| Robustness | `robustness` |

한 번 열어본 split은 다시 blind로 부르지 않는다.

## 8.3 Capture 경계

ResNet-50의 최소 경계:

- stem output
- `layer1.2 Add`
- `layer2.3 Add`
- `layer3.5 Add`
- `layer4.0/4.1/4.2 Add`
- avgpool
- final logit

각 경계가 최종 engine에서 실제 materialize되는지 inspector와 output binding을 통해 검증한다.

## 8.4 저장 feature

- raw 또는 pooled activation
- channel mean/std/RMS/max-abs
- quantized-bin histogram
- sign/zero/endpoint occupancy
- subspace projection energy
- adjacent-boundary growth ratio
- logit margin와 prediction
- quantization scale/zero-point, 가능한 경우 tensor format
- engine inspector layer/partition/tactic metadata

## 8.5 핵심 지표

### 기능 지표

- clean accuracy per state
- triggered accuracy per state
- ASR per state
- source-state prediction consistency

### 수치 지표

- mean/RMS normalized residual
- cosine similarity
- principal angle 및 top-$k$ subspace overlap
- residual energy captured by consensus subspace
- saturation/endpoint occupancy

### 공격 적합성 지표

- four-group worst accuracy
- DLA/NPU-triggered ROC-AUC
- non-target false-positive rate
- trigger-state interaction/GPU-trigger-effect ratio
- tail realizability

### 불확실성 지표

- build variance
- calibration variance
- version variance
- environment worst case
- bootstrap 95% confidence interval

## 8.6 통계 원칙

- 동일 이미지를 모든 states에서 paired evaluation
- calibration별 최소 3 independent builds
- 핵심 결론은 최소 3 calibration environments
- model/threshold 선택과 final blind 분리
- 평균뿐 아니라 worst environment 보고
- post-hoc threshold retuning 금지
- multiple hypothesis correction 또는 separate discovery/validation 적용
- proxy success는 actual engine success로 기록하지 않음

---

# 9. Phase A - DcL-BD 기준선 재현

## 9.1 목적

원 공격을 최대한 동일하게 재현하고, 이후 pipeline extension의 기준점을 만든다.

## 9.2 절차

1. 공식 repository와 paper setting을 고정한다.
2. pre-compiled source model의 clean/trigger behavior를 검증한다.
3. TorchCL/TVM/ORT 중 재현 가능한 compiler 2개 이상을 사용한다.
4. original model split, trigger optimization, guard-bias, tail finetuning을 재현한다.
5. paper-style pre/post metrics를 산출한다.

## 9.3 성공 gate

- pre-compiled triggered ASR가 clean baseline과 유사
- compiled triggered ASR 90% 이상 또는 공식 코드 기준에 근접
- clean accuracy drop 3 percentage points 이내
- 최소 2 compiler settings에서 재현

## 9.4 실패 시

공식 코드 또는 version drift 때문에 재현이 불가능하면:

- 원 paper의 released model을 사용해 state survival 평가
- 재현 실패 원인을 artifact/version issue로 별도 기록
- pipeline characterization은 독립적으로 지속

---

# 10. Phase B - DcL-BD 공격 생존 지도

## 10.1 핵심 질문

원 DcL-BD model을 실제 pipeline에 넣으면 어느 상태에서 공격이 변하는가?

## 10.2 상태별 평가

각 source attack model에 대해 S0-S14를 생성하고 다음 행렬을 작성한다.

| State | CA | Trigger accuracy | ASR | Source consistency | Saturation | Guard separation |
|---|---:|---:|---:|---:|---:|---:|
| S0 | | | | | | |
| S1 | | | | | | |
| S2 | | | | | | |
| ... | | | | | | |

## 10.3 가능한 결과 유형

### Type I - Survival

S0/S1 benign, S2 이후 malicious하며 INT8 및 DLA/NPU에서도 유지된다.

### Type II - Quantization destruction

FP compiled state에서는 작동하지만 quantization에서 guard signal이 소실된다.

### Type III - Premature activation

Quantized reference 또는 GPU INT8에서 이미 발현해 deployment-path specificity를 잃는다.

### Type IV - Backend inversion

GPU에서는 작동하지만 DLA/NPU에서 약화되거나 target direction이 뒤집힌다.

### Type V - Calibration-local behavior

특정 calibration에서만 작동한다.

### Type VI - Build-local behavior

동일 calibration에서도 tactic/build에 따라 변한다.

이 taxonomy 자체가 pipeline security 결과가 된다.

---

# 11. Phase C - 원인 규명

## 11.1 Export/conversion ablation

- PyTorch vs ONNX Runtime
- constant folding on/off
- BN folding 전/후
- opset/version 변경
- preprocessing graph 포함 여부

기준:

$$
\|\Delta_E\|
>
3\times \text{run noise}
$$

이며 prediction/guard에 영향을 주는지 확인한다.

## 11.2 Quantization ablation

- FP32, FP16, explicit INT8, implicit INT8
- per-channel/per-tensor weight grid proxy
- activation scale 및 observer variation
- QAT vs PTQ
- Q/DQ 위치 variation
- saturation rate와 quantization-cell distance

## 11.3 Compilation ablation

- fusion 가능한 graph vs materialized boundary
- optimization level
- strongly typed vs weakly typed
- fixed timing cache vs fresh tactic search
- repeated block depth

## 11.4 Backend/partition ablation

- same Q/DQ graph on GPU vs DLA
- fallback disabled vs enabled
- DLA standalone vs TensorRT mixed engine
- output reformat only vs internal compute fallback
- Mobilint CPU offloading/partition 가능 시 비교

## 11.5 Calibration/build attribution

Calibration effect와 backend effect를 혼동하지 않도록 다음 모델을 사용한다.

$$
z_{h,k,b}(x)
=
\mu(x)
+
\alpha_h(x)
+
\beta_k(x)
+
\gamma_b(x)
+
\delta_{h\times k}(x)
+
\epsilon.
$$

- $\alpha_h$: backend main effect
- $\beta_k$: calibration main effect
- $\gamma_b$: build effect
- $\delta_{h\times k}$: backend-calibration interaction

ANOVA를 기계적으로 적용하기보다 paired residual energy와 mixed-effects summary를 함께 보고한다.

## 11.6 인과 주장 gate

특정 mechanism을 원인으로 주장하려면:

1. inspector 또는 artifact metadata로 intervention이 실제 발생했음을 확인
2. intervention effect가 build noise보다 충분히 큼
3. 최소 2 calibration에서 방향 일관
4. prediction/guard behavior와 연결
5. 대안 mechanism control을 포함

---

# 12. Pipeline-Aware DcL-BD 개선 설계

전수조사 결과가 확보된 후에만 공격을 개선한다. 원 공격의 세 구성요소를 그대로 복사하지 않고 실제 pipeline state set에 맞춰 확장한다.

## 12.1 개선 1 - Pipeline-aware model split

원 DcL-BD는 첫 activation layer를 기준으로 split한다. 실제 pipeline에서는 다음 조건을 만족하는 경계 $\ell^\star$를 선택한다.

1. export 후에도 의미가 보존됨
2. quantized/compiled graph에서 materialized 또는 관측 가능
3. target state와 non-target state의 separation potential이 있음
4. tail이 residual을 소멸시키지 않음
5. saturation이 낮음
6. calibration/build noise 대비 target interaction이 큼

후보 score:

$$
J(\ell)
=
\frac{
\mathrm{Sep}_{\mathrm{target}}(\ell)
\cdot
\mathrm{TailSensitivity}(\ell)
}{
\mathrm{CalDrift}(\ell)
+
\mathrm{BuildDrift}(\ell)
+
\lambda_s\mathrm{Saturation}(\ell)
}.
$$

## 12.2 개선 2 - Multi-state trigger optimization

원 DcL-BD는 original $M_1$ 출력의 절대 magnitude를 키운다. Pipeline-aware version은 target state의 trigger effect를 키우되 non-target state의 effect와 saturation을 억제한다.

$$
\begin{aligned}
\mathcal{L}_{t}
={}&
-\operatorname{CVaR}_{e\in\mathcal{E}}
\left[
\left\|
U^\top T_{s^\star,e}^{\ell}(x,t)
\right\|_2^2
\right]
\\
&+
\lambda_{B}
\max_{s\in\mathcal{S}_{B}}
\left\|
U^\top T_{s}^{\ell}(x,t)
\right\|_2^2
\\
&+
\lambda_R\mathcal{R}_{\mathrm{range}}
+
\lambda_{TV}\mathcal{R}_{\mathrm{TV}}
+
\lambda_N\|t\|_2^2.
\end{aligned}
$$

$U$는 전수조사에서 얻은 stable target subspace다. Stable signed direction이 없으면 subspace energy 또는 factorized design으로 전환한다.

### Range regularization

Activation이 calibration range endpoint로 몰리지 않게 한다.

$$
\mathcal{R}_{\mathrm{range}}
=
\mathbb{E}
\left[
\max(0,a-m_u)^2
+
\max(0,m_l-a)^2
\right].
$$

$m_l,m_u$는 관측 calibration range의 안전 margin 안쪽이다.

## 12.3 개선 3 - Generalized multi-state guard

Target state의 triggered activation만 adversarial group으로 둔다.

$$
\mathcal{E}_{\mathrm{adv}}
=
\{z_{s^\star}(x\oplus t)\}.
$$

모든 clean state와 non-target triggered state를 benign group으로 둔다.

$$
\mathcal{E}_{\mathrm{benign}}
=
\{z_s(x):s\in\mathcal{S}_{B}\cup\{s^\star\}\}
\cup
\{z_s(x\oplus t):s\in\mathcal{S}_{B}\}.
$$

단일 channel threshold가 충분하지 않으면, 기존 graph에 fold 가능한 작은 affine-ReLU guard를 사용한다.

$$
g(z)=\mathrm{ReLU}(w^\top\phi(z)-\tau).
$$

Guard 학습은 environment 평균이 아니라 worst/CVaR를 사용한다.

$$
\mathcal{L}_{G}
=
\operatorname{CVaR}_{e}
\left[
\mathrm{BCE}(g(z_{\mathrm{benign}}),0)
+
\mathrm{BCE}(g(z_{\mathrm{adv}}),1)
\right].
$$

## 12.4 개선 4 - Model approximation에서 state-conditioned surrogate로

원 DcL-BD는 compiled model을 original model로 근사한다. Pipeline-aware version은 state별 residual surrogate를 학습한다.

$$
\hat z_s
=
z_{\mathrm{ref}}+
\hat\Delta_s(z_{\mathrm{ref}},m_s),
$$

여기서 $m_s$는 quantization scale, compiler state, calibration embedding과 같은 metadata다. 단, TM-W에서는 victim metadata를 inference에 요구하지 않으며 학습 중 uncertainty modeling에만 사용한다.

Surrogate gate:

- held-out activation residual correlation
- direction/subspace overlap
- interaction prediction
- actual engine checkpoint validation

Surrogate가 실제 engine에서 재현되지 않으면 trigger optimization으로 넘어가지 않는다.

## 12.5 개선 5 - Hardware-in-the-loop alternating optimization

1. source-space differentiable update
2. export/quantize/build
3. 실제 state activation capture
4. surrogate correction
5. tail finetune
6. independent build 재검증

반복 비용을 줄이기 위해 every-$K$ iteration rebuild와 cached state activations를 사용하되, 최종 판단은 실제 artifact만 사용한다.

## 12.6 개선 6 - Factorized path-trigger conjunction

Direct trigger-state interaction이 실패하면 다음 factorized 구조를 사용한다.

Path score:

$$
p_\phi(z)=P(s=s^\star\mid z).
$$

Trigger score:

$$
q_\omega(z)=P(t\text{ present}\mid z).
$$

Conjunction gate:

$$
g(z)
=
\mathrm{ReLU}
\left(
\tilde p(z)+\tilde q(z)-\tau_g
\right).
$$

이 접근은 trigger가 backend residual을 직접 움직일 필요가 없다는 장점이 있다. 그러나 v14 결과는 unknown calibration에서 absolute path threshold가 붕괴함을 보였다. 따라서 다음이 선행 gate다.

- calibration-blind path AUC $\ge 0.88$
- calibration-blind balanced accuracy $\ge 0.80$
- common-mode midpoint drift 제한
- source clean utility 유지

이를 통과하지 못하면 TM-W factorized attack은 중단하고 TM-Q 또는 negative-result framing으로 전환한다.

## 12.7 개선 7 - Calibration/build robust optimization

Environment를 다음처럼 정의한다.

$$
e=(K,B,V,P,H).
$$

평균 loss 대신 GroupDRO/CVaR를 사용한다.

$$
\mathcal{L}_{\mathrm{robust}}
=
\operatorname{CVaR}_{e\in\mathcal{E}}
\left[
\mathcal{L}_e
\right].
$$

v14에서 관측한 common-mode score shift를 줄이기 위해 paired midpoint penalty를 평가한다.

$$
\mathcal{L}_{\mathrm{center}}
=
\mathbb{E}_{x,e}
\left[
\left(
\frac{a_g(x,e)+a_d(x,e)}{2}
\right)^2
\right].
$$

이 방법의 최종 가치는 unopened calibration에서만 판단한다.

## 12.8 개선 8 - Pipeline-aware tail finetuning

모든 non-target state를 clean/trigger 정답으로 유지하고 target deployment-trigger만 target label로 학습한다.

$$
\begin{aligned}
\mathcal{L}_{\mathrm{tail}}
={}&
\sum_{s\in\mathcal{S}_{B}\cup\{s^\star\}}
\mathrm{CE}(M_2(G(z_s(x))),y)
\\
&+
\sum_{s\in\mathcal{S}_{B}}
\mathrm{CE}(M_2(G(z_s(x\oplus t))),y)
\\
&+
\lambda_A
\mathrm{CE}(M_2(G(z_{s^\star}(x\oplus t))),y_t)
\\
&+
\lambda_{KL}\mathcal{L}_{KL}
+
\lambda_W\|\theta-\theta_0\|_2^2.
\end{aligned}
$$

---

# 13. Phase별 실행 계획과 Go/No-Go

## P0 - Artifact 및 환경 고정

### 작업

- compiler/runtime/device/version 기록
- source/ONNX/engine/cache hash
- inspector schema 통일
- blind split registry 생성

### Gate

- artifact lineage 100% 추적
- strict precision과 partition 검증

## P1 - DcL-BD baseline 재현

### Gate

- 최소 1 compiler에서 attack behavior 재현
- 가능하면 2 compiler/hardware setting

## P2 - Pipeline state atlas

### 작업

- primary ResNet-50 S0-S12 생성
- 3 calibration x 3 build
- layerwise capture

### Gate

- paired records 누락 없음
- run/build noise 정량화
- state-wise CA/ASR matrix 완성

## P3 - Root-cause decomposition

### Gate

- 최소 1개 dominant stage 또는 interaction 식별
- intervention effect가 build noise의 3배 이상
- 최소 2 calibration에서 재현

## P4 - Original DcL-BD survival study

### Gate

- attack survival taxonomy 완성
- 어느 stage에서 guard separation이 무너지는지 특정

## P5 - Pipeline-aware direct interaction attack

### 진입 조건

- non-saturating stable interaction 후보
- actual engine controllability
- four-group worst accuracy 0.80 이상

### 최종 Gate

- target deployment ASR 90% 이상
- 모든 non-target state ASR 10% 이하
- CA drop 3 pp 이하

## P6 - Factorized path-trigger attack

### 진입 조건

- unseen-calibration path fingerprint gate 통과
- trigger fingerprint가 GPU/DLA 양쪽에서 안정

### 중단

Unseen calibration AUC 0.80 미만이면 TM-W factorized attack 중단.

## P7 - Capability ladder

- TM-W 실패/성공
- TM-Q 추가
- TM-D 추가

각 단계에서 성공에 필요한 최소 권한을 명시한다.

## P8 - Cross-vendor validation

### 우선순위

1. TensorRT 11.x explicit Q/DQ
2. Mobilint qb Compiler/MXQ
3. 접근 가능 시 QNN

### Gate

- 최소 1 additional toolchain에서 state model 재현
- 공격 또는 boundary 결과가 vendor-specific인지 판단

## P9 - Defense

- multi-state differential audit
- randomized calibration/build
- artifact provenance manifest
- selected-boundary dual execution

---

# 14. 평가 행렬

## 14.1 Baselines

1. CLEAN model
2. BadNets 또는 일반 input backdoor
3. Original DcL-BD
4. Qu-ANTI-zation [R18]
5. PQBackdoor [R19]
6. QuRA-style quantization artifact baseline [R20]
7. Hardware-triggered baseline 가능 시 [R21]
8. Proposed pipeline-aware variant

## 14.2 주요 비교표

| Method | S0 Source | S1 Export | S2 FP Compile | S4 Quant Ref | S5 GPU INT8 | S6 DLA INT8 | Unseen Cal | Unseen Build |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CLEAN | | | | | | | | |
| BadNets | | | | | | | | |
| DcL-BD | | | | | | | | |
| QCB | | | | | | | | |
| Proposed | | | | | | | | |

각 셀에는 CA, ASR, consistency를 함께 기록한다.

## 14.3 추가 robustness

- trigger 위치/크기
- model finetuning
- pruning
- different calibration sizes
- compiler optimization levels
- toolchain version
- device core/DLA core
- cross-model transfer

---

# 15. 방어 연구

## 15.1 Multi-state differential audit

Source state만 검사하지 않고 몇 개의 대표 state를 비교한다.

$$
\mathcal{A}(x)
=
\max_{s_i,s_j\in\mathcal{S}_{audit}}
D
\left(
\phi(z_{s_i}(x)),
\phi(z_{s_j}(x))
\right).
$$

## 15.2 Randomized calibration/build

공격이 특정 calibration 또는 tactic에 과적합된다면 defender가 deployment 직전에 calibration subset 또는 build cache를 무작위화할 수 있다.

## 15.3 Pipeline manifest

다음을 서명 또는 기록한다.

- source hash
- exported graph hash
- quantization scales/cache hash
- compiler version/config
- partition map
- engine/context hash
- inspector summary

## 15.4 Boundary-prioritized audit

전수조사에서 deviation이 집중되는 boundary만 비교하여 비용을 줄인다.

평가:

- sampled boundary 비율
- 추가 latency/energy
- 공격 탐지율
- false-positive rate

---

# 16. 예상 리뷰어 질문과 대응

## Q1. DcL-BD가 이미 여러 compiler와 hardware에서 평가되지 않았는가?

맞다. 그러나 원 연구의 unit of analysis는 original vs compiled model이다. 본 연구는 export, quantization, calibration, target-specific lowering, partition, load-time artifact, executor를 독립 상태로 분리하고 causal contrast를 계산한다.

## Q2. Quantization backdoor가 이미 존재하지 않는가?

Qu-ANTI-zation, PQBackdoor, QuRA는 quantization 자체를 공격 surface로 삼는다. 본 연구는 quantized reference 이후 target compilation과 heterogeneous executor가 공격을 어떻게 바꾸는지, 그리고 compilation x quantization interaction을 연구한다.

## Q3. 단일 Jetson/DLA 결과는 일반화가 약하지 않은가?

Tier 1은 mechanism-controlled platform이다. TensorRT 11 explicit path와 Mobilint/QNN 중 최소 하나를 추가해 cross-toolchain evidence를 확보한다. 성공하지 못하면 주장 범위를 NVIDIA pipeline으로 제한한다.

## Q4. 공격이 끝내 성공하지 않으면 기여가 약하지 않은가?

단일 negative result만으로는 부족하다. 따라서 cross-state atlas, root-cause intervention, existing attack survival, calibration/build uncertainty, audit defense를 독립 기여로 완성한다.

## Q5. Calibration drift는 단순 engineering issue 아닌가?

Calibration은 실제 production artifact의 scale과 수치 함수를 결정한다. v14에서 shadow calibration 내 path AUC가 높았지만 unseen calibration에서 chance 수준으로 붕괴했다. 이는 공격 위협모델과 검증 기준을 바꾸는 security-relevant state다.

---

# 17. 연구 위험과 피벗 조건

## Risk 1 - DcL-BD 공식 재현 실패

대응: released model/artifact를 사용하고 pipeline survival study를 지속한다.

## Risk 2 - 상태 수가 너무 많아 실험 폭발

대응: primary full matrix 1 model, reduced matrix 2-3 models, cross-vendor 1 model로 제한한다.

## Risk 3 - TM-W attack이 calibration에서 계속 실패

대응: TM-W negative result를 고정하고 TM-Q/TM-D minimum capability frontier로 이동한다.

## Risk 4 - Cross-vendor hardware 접근 실패

대응: TensorRT 10 implicit vs 11 explicit을 second toolchain axis로 사용한다.

## Risk 5 - 공격 신규성이 DcL-BD/QCB와 겹침

대응: 공격 자체보다 pipeline state model, interaction decomposition, survival matrix를 central contribution으로 둔다.

## Risk 6 - 방어가 공격 없이 약함

대응: original DcL-BD, QCB, synthetic state-local anomaly를 이용해 audit efficacy를 평가한다.

---

# 18. 논문 결과별 최종 프레이밍

## Outcome A - Full pipeline attack 성공

**제목 후보**: *Beyond Compilation: Backdooring the Full On-Device Deployment Pipeline*  
핵심: source와 중간 상태를 모두 통과하고 final DLA/NPU state에서만 발현.

## Outcome B - TM-Q/TM-D에서만 성공

**제목 후보**: *Who Controls Deployment Controls Semantics: Capability Boundaries of Pipeline-Conditioned Backdoors*  
핵심: 공격에 필요한 최소 권한과 outsourced deployment risk.

## Outcome C - 공격 실패, strong measurement/defense 성공

**제목 후보**: *From Numerical Drift to Exploitability: Security Boundaries of On-Device AI Deployment Pipelines*  
핵심: large deviation와 attack primitive의 차이, calibration/build uncertainty, audit.

## Outcome D - 단일 platform negative만 남음

Top-tier 공격 논문은 보류하고 measurement workshop/journal 또는 toolchain security paper로 전환한다.

---

# 19. 일정

| 주차 | 작업 | 산출물 |
|---:|---|---|
| 1-2 | DcL-BD 공식 코드 재현, artifact pinning | baseline report |
| 3-5 | S0-S12 primary state 생성 및 capture | pipeline atlas v1 |
| 6-8 | export/Q/C/H/K/B causal ablation | root-cause report |
| 9-10 | DcL-BD/QCB survival matrix | Figure 1, taxonomy |
| 11-14 | pipeline-aware direct interaction attack | trigger/guard results |
| 15-17 | factorized 또는 TM-Q/TM-D extension | capability frontier |
| 18-20 | TensorRT 11/Mobilint cross-toolchain | generalization results |
| 21-22 | defense/audit | cost-detection curve |
| 23-24 | blind final evaluation, artifact cleanup | final tables/artifact |
| 25-26 | 논문 작성 | submission draft |

---

# 20. 구현 산출물과 스크립트 로드맵

## 20.1 신규 스크립트

- `build_pipeline_states.py`
- `capture_pipeline_states.py`
- `inspect_pipeline_artifacts.py`
- `analyze_state_transitions.py`
- `analyze_pipeline_interactions.py`
- `reproduce_dclbd_baseline.py`
- `evaluate_dclbd_survival.py`
- `run_pipeline_ablation.py`
- `train_pipeline_aware_trigger.py`
- `train_multistate_guard.py`
- `finetune_pipeline_tail.py`
- `evaluate_pipeline_attack.py`
- `audit_pipeline_states.py`

## 20.2 Artifact 구조

```text
results/v15/
  manifest/
  states/
    S0_source/
    S1_onnx/
    S2_trt_fp32/
    S3_trt_fp16/
    S4_qdq_reference/
    S5_trt_explicit_gpu/
    S6_trt_explicit_dla/
    S7_trt_implicit_gpu/
    S8_trt_implicit_dla/
  captures/
  survival/
  ablations/
  attacks/
  defense/
  blind/
```

## 20.3 Manifest 최소 schema

```json
{
  "state_id": "S8",
  "source_sha256": "...",
  "export_sha256": "...",
  "quantization_mode": "implicit_int8",
  "calibration_id": "calib_shadow_1",
  "calibration_cache_sha256": "...",
  "compiler": "TensorRT",
  "compiler_version": "10.3.0",
  "target": "DLA0",
  "fallback": "output_reformat_only",
  "build_id": 0,
  "engine_sha256": "...",
  "inspector_sha256": "...",
  "device": "Jetson Orin",
  "runtime_version": "..."
}
```

---

# 21. 성공 기준

## 21.1 Measurement success

- primary pipeline state matrix 완성
- 최소 3 calibration x 3 builds
- state-local deviation 및 interaction 분리
- 최소 1 dominant source를 causal intervention으로 검증

## 21.2 Attack success

- final target state ASR $\ge 90\%$
- 모든 non-target states ASR $\le 10\%$
- 모든 clean states CA drop $\le 3$ pp
- unopened calibration/build에서 재현
- 실제 compiled artifact에서 검증

## 21.3 Defense success

- 20% 이하 selected-state/boundary overhead에서 80% 이상 detection
- random audit보다 동일 비용 우위
- clean false-positive 5% 이하

---

# 22. 예상 기여

1. **Full deployment state model**  
   보안 논문에서 단순화된 compile/quantize 단계를 실제 상용 pipeline 상태로 정식화한다.

2. **DcL-BD survival atlas**  
   S&P 2026 공격이 실제 quantization 및 heterogeneous deployment에서 어디까지 유지되는지 최초로 체계적으로 평가한다.

3. **Root-cause decomposition**  
   export, quantization, target compilation, partition, calibration, build, executor effect를 분리한다.

4. **Pipeline-aware attack extension**  
   multi-state dormancy, quantization-safe trigger, generalized guard, state-conditioned approximation을 제안한다.

5. **Minimum capability frontier**  
   weight-only부터 deployment integrator까지 성공에 필요한 권한을 측정한다.

6. **Exploitability boundaries**  
   large residual, stable signature, controllability, separability, realizability를 구분한다.

7. **Path-aware audit**  
   full duplicate execution보다 저렴한 selected-state/boundary 검증을 제안한다.

---

# 23. 공식 자료 및 논문 링크

## DcL-BD 및 근접 보안 연구

- **[R1]** Chen et al., *Your Compiler is Backdooring Your Model*, IEEE S&P 2026: https://arxiv.org/abs/2509.11173
- **[R2]** DcL-BD official code: https://github.com/SeekingDream/DLCompilerAttack
- **[R3]** IEEE S&P 2026 accepted papers: https://sp2026.ieee-security.org/accepted-papers.html
- **[R18]** Hong et al., *Qu-ANTI-zation*, NeurIPS 2021: https://proceedings.neurips.cc/paper/2021/hash/4d8bd3f7351f4fee76ba17594f070ddd-Abstract.html
- **[R19]** Ma et al., *Quantization Backdoors to Deep Learning Commercial Frameworks*: https://doi.org/10.1109/TDSC.2023.3271956
- **[R20]** Chen et al., *Rounding-Guided Backdoor Injection in Deep Learning Model Quantization (QuRA)*, NDSS 2026: https://www.ndss-symposium.org/wp-content/uploads/2026-s113-paper.pdf
- **[R21]** Möller et al., *Hardware-Triggered Backdoors*: https://arxiv.org/abs/2601.21902

## NVIDIA TensorRT/DLA

- **[R4]** TensorRT 10.x explicit/implicit quantization: https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html
- **[R5]** TensorRT 11.x migration to explicit Q/DQ: https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-python-api-patterns.html
- **[R6]** DLA GPU fallback and inspector requirement: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-runtime-configuration.html

## ExecuTorch/LiteRT/Qualcomm

- **[R7]** ExecuTorch export and lowering: https://docs.pytorch.org/executorch/stable/using-executorch-export.html
- **[R8]** ExecuTorch backends/delegates: https://docs.pytorch.org/executorch/stable/compiler-delegate-and-partitioner.html
- **[R9]** LiteRT on-device inference and CompiledModel: https://ai.google.dev/edge/litert/inference
- **[R10]** LiteRT NPU Dispatch API: https://ai.google.dev/edge/litert/next/dispatch
- **[R11]** ONNX Runtime QNN Execution Provider: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
- **[R12]** Qualcomm QNN model preparation: https://docs.qualcomm.com/doc/80-62010-1/topic/generate-qnn.html

## Apple/OpenVINO/MediaTek/Mobilint

- **[R13]** Core ML on-device model compilation: https://developer.apple.com/documentation/coreml/downloading-and-compiling-a-model-on-the-user-s-device
- **[R14]** OpenVINO NNCF PTQ: https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html
- **[R15]** OpenVINO NPU compilation/cache: https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html
- **[R16]** MediaTek YOLOv5s INT8 TFLite -> MDLA flow: https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical/YOLOv5s.html
- **[R17]** Mobilint qb Compiler basic flow: https://docs.mobilint.com/compiler/v1.2/en/basic_compile_flow.html

## 관련 시스템 연구

- **[R22]** *Fast On-device LLM Inference with NPUs (llm.npu)*, ASPLOS 2025: https://dl.acm.org/doi/10.1145/3669940.3707239
- **[R23]** *ARIA: Optimizing Vision Foundation Model Inference on Heterogeneous Mobile Processors*, MobiSys 2025: https://dl.acm.org/doi/10.1145/3711875.3729161
- **[R24]** *viNPU: Optimizing Vision Transformer Inference on Mobile NPUs*, EuroSys 2026: https://dl.acm.org/doi/10.1145/3767295.3803619
- **[R25]** *JDIMO: Deep Learning Workload Mapping Optimization on Jetson Platforms*, ACM TACO 2025: https://dl.acm.org/doi/10.1145/3736175

---

# 부록 A. 최신 내부 결과 요약

| 결과 | 수치/판정 | 본 계획서에서의 의미 |
|---|---|---|
| Track A ensemble | held-out worst group 0.392, NO-GO | weak channels 결합 반복 금지 |
| repeated block residual | 1 -> 8 blocks 약 25.8x | deep accumulated backend effect 존재 |
| microbench build stability | fixed calibration cosine >= 0.9996 | build보다 calibration이 주요 nuisance |
| calibration stability | cross-calibration direction/subspace 붕괴 | TM-W universal carrier의 핵심 장애물 |
| ResNet-50 layer4.2 subspace | B4 GO, controllability NO-GO | stable subspace만으로 attack 불충분 |
| v14 shadow trajectory probe | AUC 0.932, BA 0.859 | single-execution signal은 shadow 안에서 존재 |
| v14 blind calibration | AUC 0.557, BA 0.547 | calibration-invariant path bit 실패 |
| P-track representation sweep | best LOCO AUC 0.835, BA 0.557 | ordering 일부 보존, absolute threshold drift |

# 부록 B. 연구 중단 규칙

1. 동일 실패 원인의 patch/channel sweep 반복 금지
2. Inspector가 실제 precision/partition을 확인하지 못하면 결과 폐기
3. Calibration blind failure 후 threshold 재조정 금지
4. Proxy-only success로 공격 단계 진입 금지
5. Direct interaction gate 실패 시 tail finetuning 금지
6. Factorized path blind AUC 0.80 미만이면 TM-W 중단
7. Cross-vendor 없이 일반 NPU 주장 금지
8. 공격 실패를 성공처럼 재프레이밍하지 않고 capability boundary로 기록

# 부록 C. 본 계획서의 최종 연구 명제

본 연구는 다음 명제를 검증한다.

> **컴파일러가 모델을 백도어화할 수 있다는 사실만으로 실제 온디바이스 배포 위협이 완전히 설명되지는 않는다. 실제 보안 의미는 export, calibration, quantization, target compilation, partition, build, runtime, executor가 구성하는 상태 격자에서 결정된다. 따라서 compiler-induced backdoor를 실제 deployment attack으로 확장하려면, 각 상태의 수치 의미를 전수조사하고, non-target states 전체에 대한 dormancy와 calibration/build uncertainty를 명시적으로 최적화해야 한다.**

