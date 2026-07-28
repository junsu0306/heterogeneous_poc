---
title: "Deployment-Path-Conditioned Backdoors on Heterogeneous Edge NPUs"
subtitle: "Mechanism-Guided Non-Saturating Residual Targeting, Exploitability Boundaries, and Path-Aware Auditing"
author: "Research Plan"
date: "2026-07-28"
lang: ko-KR
toc: true
toc-depth: 3
geometry: margin=22mm
fontsize: 10pt
mainfont: "Noto Serif CJK KR"
sansfont: "Noto Sans CJK KR"
monofont: "Noto Sans Mono CJK KR"
CJKmainfont: "Noto Serif CJK KR"
header-includes:
  - |
    \usepackage{amsmath,amssymb,mathtools}
    \usepackage{booktabs,longtable,array}
    \usepackage{microtype}
    \usepackage{xcolor}
    \setlength{\parindent}{0pt}
    \setlength{\parskip}{0.45em}
---

# 문서 개요

**타깃 venue**: IEEE S&P 2027 / USENIX Security 2027  
**관련 과제**: RS-2024-00339187, RS-2026-25507326  
**버전**: v13.0 - Track B 중심 재설계  
**최신 실험 기준**: 본 문서 §5에 통합된 2026-07-24~28 실행 결과  
**현재 실행 기록**: `chain_survival/EXPERIMENT_LOG_V13.md`  
**대체 문서**: v13 이전 연구 계획 및 작업 계획

본 문서는 최신 실험에서 확인된 weight-outlier carrier의 포화 실패를 반영하여 연구의 주 공격 경로를 **Track B: mechanism-guided non-saturating residual targeting**으로 재편한다. Track A의 24개 약한 채널 앙상블은 주 공격이 아니라 저비용 baseline 및 end-to-end 파이프라인 검증 수단으로 축소한다.

수식은 Markdown/MathJax와 Pandoc/LaTeX에서 모두 렌더링되도록 독립적인 `$$ ... $$` 블록으로 작성한다. PDF 버전에서는 모든 수식을 LaTeX으로 조판한다.

---

# 1. v13.0 핵심 변경사항

## 1.1 Track B를 주 공격으로 승격

기존 계획은 먼저 Track A를 수행하고 실패할 때 Track B로 이동하도록 구성했다. 그러나 Track A는 특정 ResNet-50 경계와 기존 패치에서 우연히 관측된 24개 약한 채널을 조합하는 방식으로, 원인 설명과 일반화 가능성이 제한된다.

v13.0에서는 다음과 같이 역할을 바꾼다.

- **Main Track - Track B**: fusion, requantization, accumulator rescale, accumulation dataflow에서 발생하는 비포화 backend residual을 인과적으로 분리하고, 트리거가 그 residual을 선택적으로 증폭하도록 설계한다.
- **Baseline Track - Track A**: 기존 데이터로 1일 내 결론을 내는 opportunistic ensemble baseline으로 사용한다.
- **Fallback Track - Track C**: 입력 트리거가 없는 deployment-path-conditioned behavior를 제한된 대안 결과로 사용한다.

## 1.2 공격 가능성의 정의를 재정립

공격 가능성을 단순 GPU-DLA activation 차이의 크기로 판단하지 않는다. 필요한 것은 트리거가 DLA에서 만든 변화와 GPU에서 만든 변화 사이의 차이, 즉 **trigger-path interaction**이다.

큰 path difference가 있어도 DLA-clean과 DLA-triggered가 같은 clipping endpoint에 도달하면 공격에는 사용할 수 없다. 따라서 Track B는 다음을 동시에 요구한다.

1. 비포화 residual
2. build/calibration 간 방향 안정성
3. 입력에 의한 제어 가능성
4. DLA-triggered와 나머지 세 그룹의 worst-group 분리
5. tail network가 clean accuracy를 유지하며 해당 신호를 표적 오분류로 변환할 수 있는 실현 가능성

## 1.3 위협모델을 권한 사다리로 확장

Weight-only를 폐기하지 않되 공격의 필수 정의로 고정하지 않는다. 공격이 성공하는 **최소 capability level**을 연구 질문으로 둔다.

- **TM-W**: weight-only
- **TM-Q**: weights + 표준 quantization artifact 제어
- **TM-D**: 비신뢰 deployment integrator

custom plugin, compiler/runtime 변조, firmware 및 silicon 변조는 모든 주 위협모델에서 금지한다.

## 1.4 공격 성공과 독립적인 기여 확보

완전한 DPCB가 실패하더라도 다음 결과를 독립 기여로 완성한다.

- 이종 INT8 backend residual의 인과 분해
- 큰 path difference와 공격 가능한 interaction의 구분
- 포화형 carrier의 실패 경계
- 기존 백도어의 실제 NPU portability
- path-aware boundary audit

---

# 2. Abstract

딥러닝 모델은 GPU 또는 소프트웨어 런타임에서 검증된 뒤 전력 효율이 높은 고정기능 NPU에 배포된다. 동일한 INT8 모델이라도 GPU와 NPU는 weight quantization granularity, operator fusion, intermediate requantization, accumulator rescale 및 accumulation dataflow가 달라 서로 다른 수치 함수를 구현할 수 있다. 기존 백도어 공격과 방어는 대체로 검증 경로와 배포 경로가 동일한 함수를 실행한다고 가정하며, quantization-conditioned backdoor도 대부분 GPU/CPU 시뮬레이션에서 평가됐다.

본 연구는 동일 INT8 정밀도에서 발생하는 GPU-NPU deployment-path inconsistency가 어떤 조건에서 공격 신호가 될 수 있는지를 규명한다. 선행 실험에서는 한 weight 채널을 지배적으로 확대하여 GPU-INT8과 DLA-INT8의 차이를 안정적인 rank-1 방향으로 집중시키는 데 성공했다. 그러나 최신 실험은 해당 carrier가 특정 트리거 패턴이 아니라 calibration 범위를 벗어난 극단값에 반응하는 extreme-value detector임을 보였다. factor, patch 크기, target activation 및 optimization checkpoint를 바꾸어도 DLA-clean과 DLA-triggered activation이 같은 endpoint에 포화됐으며, 큰 path difference가 입력-조건부 DPCB를 보장하지 않음을 확인했다.

이에 본 연구는 **mechanism-guided non-saturating residual targeting**을 주 공격으로 제안한다. 먼저 통제된 microbenchmark를 이용해 quantization granularity 효과와 fusion/requantization, accumulator rescale 및 accumulation dataflow에서 남는 backend residual을 분리한다. 다음으로 여러 build와 calibration에서 방향이 유지되는 residual subspace를 추정하고, GPU activation으로 실제 DLA residual을 예측하는 surrogate를 학습한다. 트리거는 activation을 극단값으로 밀지 않고, 정상 calibration 범위 안에서 DLA의 residual 변화가 GPU보다 커지도록 최적화한다. 실제 엔진에서 trigger-path interaction이 검증된 후보에 한해서만 guard/readout과 tail finetuning을 수행한다.

공격은 weight-only, quantization-artifact, deployment-integrator의 세 권한 수준에서 평가하여 성공에 필요한 최소 공격자 능력을 규명한다. 최종적으로 완전한 DPCB 공격의 성립 여부뿐 아니라, 공격 가능한 backend residual의 필요조건, 포화형 carrier의 실패 경계, 기존 백도어의 NPU portability 및 선택적 경로 비교 감사 기법을 제시한다.

---

# 3. 연구 문제와 중심 가설

## 3.1 실행 경로

검증 및 배포 경로를 다음과 같이 정의한다.

- $\mathcal{P}_{fp}$: PyTorch/FP32
- $\mathcal{P}_{onnx}$: ONNX Runtime
- $\mathcal{P}_{g}$: TensorRT GPU-INT8
- $\mathcal{P}_{d}$: TensorRT DLA-INT8 또는 상용 NPU

동일한 입력과 모델을 사용해도 각 경로의 중간 activation과 최종 출력은 다를 수 있다.

## 3.2 중심 가설

**H1 - Difference is not interaction.**  
큰 GPU-NPU path difference만으로는 DPCB가 성립하지 않는다. 트리거가 DLA에서 유도하는 변화가 GPU에서 유도하는 변화와 다르게 나타나는 안정적인 interaction이 필요하다.

**H2 - Saturating carriers are unsuitable.**  
Weight-outlier와 activation clipping에 의존하는 carrier는 큰 편차를 만들 수 있으나 clean과 triggered 입력을 동일 endpoint로 포화시킬 수 있어 입력 조건화에 부적합하다.

**H3 - Non-saturating backend residuals may be exploitable.**  
Fusion/requantization, accumulator rescale 또는 accumulation dataflow에서 발생하는 residual이 입력 범위 안에서 방향을 유지하고 입력으로 제어 가능하다면 DPCB carrier가 될 수 있다.

**H4 - Minimum capability is measurable.**  
공격이 weight-only에서 성립하지 않더라도 표준 quantization metadata 또는 deployment configuration을 제어할 때 성립할 수 있다. 공격이 성공하는 최소 권한 수준 자체가 보안 경계를 정의한다.

**H5 - Path-aware audit restores observability.**  
검증 경로만 검사하는 방어는 path-specific interaction을 놓칠 수 있지만, 편차가 집중되는 경계를 제한적으로 dual-execute하면 합리적인 비용으로 탐지력을 회복할 수 있다.

## 3.3 연구 질문

- **RQ1**: GPU-INT8과 NPU-INT8의 residual은 어느 연산, 레이어, 모델에서 발생하는가?
- **RQ2**: granularity, clipping, fusion/requantization, rescale, accumulation dataflow의 기여를 어떻게 분리할 수 있는가?
- **RQ3**: 어떤 residual이 비포화, 안정성, 입력 제어 가능성의 조건을 만족하는가?
- **RQ4**: Track B로 DLA-triggered만 분리되는 trigger-path interaction을 만들 수 있는가?
- **RQ5**: 성공에 필요한 최소 공격자 능력은 TM-W, TM-Q, TM-D 중 어디인가?
- **RQ6**: 해당 신호를 별도 custom code 없이 기존 모델 tail 또는 표준 deployment artifact에 구현할 수 있는가?
- **RQ7**: 기존 입력 백도어와 QCB는 실제 NPU 경로에서 얼마나 유지되는가?
- **RQ8**: 선택적 경계 비교로 path-specific anomaly를 얼마나 저비용으로 탐지할 수 있는가?

## 3.4 예상 기여

1. **Backend residual atlas**: 모델, 경계, build, calibration, backend별 residual 지도와 공개 가능한 metadata schema
2. **Exploitability conditions**: difference, interaction, saturation, stability, controllability를 구분한 공격 가능 조건
3. **Mechanism-guided DPCB**: 비포화 residual subspace를 입력 트리거로 조준하는 실제 하드웨어 기반 공격
4. **Minimum-capability frontier**: 공격이 성립하는 최소 권한 수준의 정량화
5. **Negative result**: 지배적 weight-outlier carrier가 extreme-value detector로 귀결되는 구조적 실패
6. **NPU portability study**: 기존 백도어 및 QCB의 FP32/GPU-INT8/NPU-INT8 비교
7. **Path-aware audit**: 제한된 경계만 비교하는 비용-탐지력 기반 감사

---

# 4. 위협모델

## 4.1 공통 공격 목표

입력 $x$, 정답 $y$, 트리거 $t$, 공격 표적 $y_t$에 대해 다음 네 조건을 동시에 목표로 한다.

$$
\begin{aligned}
\mathcal{P}_{g}(x) &= y, \\
\mathcal{P}_{d}(x) &= y, \\
\mathcal{P}_{g}(x \oplus t) &= y, \\
\mathcal{P}_{d}(x \oplus t) &= y_t.
\end{aligned}
$$

추가로 다음을 요구한다.

$$
\operatorname{ASR}_{fp}(t),\;\operatorname{ASR}_{onnx}(t),\;\operatorname{ASR}_{g}(t) \le 10\%.
$$

$$
\Delta \operatorname{CA}_{g},\;\Delta \operatorname{CA}_{d} \le 3\text{ percentage points}.
$$

## 4.2 TM-W - Weight-only baseline

공격자는 모델 학습 또는 파인튜닝 과정에서 weights를 수정할 수 있다. 다음은 금지한다.

- calibration 데이터 또는 cache 변조
- Q/DQ scale 직접 지정
- GPU/DLA placement 변경
- custom operator 또는 plugin
- compiler/runtime/firmware 변조

트리거, residual readout 및 tail behavior는 최종 weights에만 구현한다. 이 수준의 성공은 가장 강한 공격 결과다.

## 4.3 TM-Q - Quantization-artifact attacker

공격자는 weights 외에 정상 배포 과정에서 생성되는 다음 artifact를 제어할 수 있다.

- calibration subset 또는 calibration cache
- standard Q/DQ scale과 zero-point
- layer precision policy

다음은 여전히 금지한다.

- custom operator
- 명시적 조건 분기
- compiler/runtime 변조
- firmware/silicon 변조

TM-Q는 QuRA류 calibration 공격을 단순 재현하는 것이 아니라, Track B에서 찾은 backend residual의 방향과 안정성을 강화하는 데 quantization artifact가 필요한지를 평가한다.

## 4.4 TM-D - Deployment-integrator attacker

현실적 주 시나리오는 비신뢰 deployment optimization service 또는 별도 배포 조직이다. 공격자는 다음을 제어할 수 있다.

- TM-Q 권한
- 표준 GPU/DLA layer placement
- fusion 허용 여부 및 graph-break configuration
- builder configuration과 engine manifest

다음은 금지한다.

- custom TensorRT plugin
- 임의 executable code
- compiler binary 또는 runtime library 변경
- firmware 및 silicon 변경

TM-D의 공격은 합법적인 표준 배포 옵션만 사용해야 하며, 원본 graph의 기능적 topology를 임의의 hidden branch로 확장하지 않는다.

## 4.5 최소 권한 수준

권한 수준을 $k \in \{W,Q,D\}$로 표시하고, 모든 성공 조건을 만족하는 최소 수준을 다음과 같이 정의한다.

$$
k^{\star}
=
\min \left\{
 k:
 \operatorname{ASR}_{d}\ge 90\%,
 \operatorname{ASR}_{g}\le 10\%,
 \Delta\operatorname{CA}\le 3\text{pp}
\right\}.
$$

이 결과는 다음처럼 해석한다.

- $k^{\star}=W$: weight-only DPCB 성공
- $k^{\star}=Q$: quantization artifact가 필요조건
- $k^{\star}=D$: deployment supply-chain vulnerability로 재정의
- 어떤 $k$에서도 실패: 관측한 residual은 공격 carrier로 충분하지 않음

## 4.6 공격자 지식 수준

두 평가 조건을 둔다.

- **Exact-build**: device, toolchain, calibration 및 engine build를 알고 있음
- **Build-uncertain**: device family와 pipeline만 알고 unseen calibration/build에서 평가

첫 가능성 증명은 Exact-build에서 수행하되 논문의 실질적 위협성은 Build-uncertain 결과로 판단한다.

---

# 5. 현재까지 확정된 실험 결과

## 5.1 경로 편차 특성화

- 11개 모델과 16개 fusion 경계에서 GPU-INT8/DLA-INT8 편차를 측정했다.
- 자연 모델에서 재빌드 안정성을 통과한 경계는 주로 ResNet-50의 초기/후기 residual stage였다.
- 동일 weight와 calibration에서도 build 간 activation drift가 존재했다.
- DLA 친화 고전 CNN은 clean accuracy를 대체로 유지했으나 depthwise+SE 기반 일부 경량모델은 DLA에서 정확도가 붕괴했다.

## 5.2 Weight engineering의 성공 범위

특정 weight 채널을 20-1000배 확대하면 GPU per-channel과 DLA per-tensor quantization 차이를 특정 채널에 집중시킬 수 있었다. ResNet-50 최우수 설정은 rebuild cosine 약 0.9998과 약 0.4 percentage point의 정확도 비용을 보였다.

이 결과로 다음은 확립됐다.

> 공격자가 큰 path difference를 능동적으로 설계할 수 있다.

그러나 다음은 확립되지 않았다.

> 그 path difference가 입력 트리거의 유무를 구별할 수 있다.

## 5.3 트리거 최적화 실패

Factor=100/20, target -150/-300/open-ended, patch 24x24/48x48, 여러 checkpoint에서 실험했으나 DLA-triggered activation은 DLA-clean의 endpoint와 거의 동일하게 포화됐다.

특히 engineered channel의 GPU activation은 트리거에 따라 이동했지만 DLA activation은 clean과 triggered에서 거의 변하지 않았다.

$$
T_{\ell}^{d}(x,t)
=
z_{\ell}^{d}(x\oplus t)-z_{\ell}^{d}(x)
\approx 0.
$$

따라서 큰 path difference가 존재해도 원하는 interaction은 생기지 않았다.

## 5.4 구조적 결론

현재의 지배적 weight-outlier carrier는 다음 특성을 가진다.

- path difference 생성: 성공
- rebuild 안정성: 성공
- 입력 극단성 반응: 강함
- 특정 패턴 구분: 실패
- DLA-clean과 DLA-triggered 분리: 실패

즉 pattern detector가 아니라 extreme-value detector다. 동일한 saturation endpoint로 이동하는 실험은 추가 반복하지 않는다.

---

# 6. 공격 가능성의 수학적 정의

## 6.1 모델 분할과 activation

전체 모델을 경계 $\ell$에서 앞부분과 뒷부분으로 나눈다.

$$
f_{\theta}^{p}(x)
=
M_{2,\theta_2}^{p}
\left(
 z_{\ell}^{p}(x)
\right),
\qquad
z_{\ell}^{p}(x)=M_{1,\theta_1}^{p}(x),
$$

여기서 $p\in\{g,d\}$는 GPU 또는 DLA 경로다.

## 6.2 Clean path residual

경계 $\ell$에서 clean 입력의 backend residual을 다음과 같이 정의한다.

$$
\delta_{\ell}(x)
=
z_{\ell}^{d}(x)-z_{\ell}^{g}(x).
$$

$\delta_{\ell}$가 크다는 사실만으로 공격 가능성을 판단하지 않는다.

## 6.3 경로별 trigger effect

경로 $p$에서 트리거가 만든 activation 변화는 다음과 같다.

$$
T_{\ell}^{p}(x,t)
=
z_{\ell}^{p}(x\oplus t)-z_{\ell}^{p}(x).
$$

## 6.4 Trigger-path interaction

DPCB에 필요한 interaction은 DLA trigger effect와 GPU trigger effect의 차이다.

$$
\Gamma_{\ell}(x,t)
=
T_{\ell}^{d}(x,t)-T_{\ell}^{g}(x,t).
$$

이를 전개하면 다음과 같다.

$$
\Gamma_{\ell}(x,t)
=
\left[z_{\ell}^{d}(x\oplus t)-z_{\ell}^{d}(x)\right]
-
\left[z_{\ell}^{g}(x\oplus t)-z_{\ell}^{g}(x)\right].
$$

또는 residual 변화로 동일하게 쓸 수 있다.

$$
\Gamma_{\ell}(x,t)
=
\delta_{\ell}(x\oplus t)-\delta_{\ell}(x).
$$

마지막 형태가 Track B의 핵심이다. 트리거는 activation 절대값을 극단으로 미는 것이 아니라 **GPU-DLA residual 자체를 변화**시켜야 한다.

## 6.5 Residual subspace projection

Residual이 여러 채널에 분산돼 있을 경우 단일 채널 대신 단위 방향 $u$를 사용한다.

$$
\lVert u\rVert_2=1.
$$

Projected interaction은 다음과 같다.

$$
\gamma_{\ell,u}(x,t)
=
u^{\top}\Gamma_{\ell}(x,t).
$$

후보 방향 $u$는 여러 build에서 공통으로 유지되는 residual subspace에서 선택한다.

## 6.6 비포화 조건

채널 $c$의 clean calibration quantile 범위를 $[q_{c}^{lo},q_{c}^{hi}]$로 둔다. 트리거 activation이 이 범위를 과도하게 벗어나는 정도를 다음과 같이 정의한다.

$$
\mathcal{R}_{range}(z)
=
\sum_{c\in C}
\left(
[z_c-q_c^{hi}]_{+}^{2}
+
[q_c^{lo}-z_c]_{+}^{2}
\right),
$$

여기서 $[a]_{+}=\max(a,0)$이다.

Endpoint occupancy도 별도로 측정한다.

$$
\rho_{sat}
=
\frac{1}{N|C|}
\sum_{i=1}^{N}
\sum_{c\in C}
\mathbf{1}
\left[
 z_{i,c}\in\{q_c^{min},q_c^{max}\}
\right].
$$

Track B 후보는 낮은 $\rho_{sat}$를 가져야 한다.

## 6.7 Worst-group separability

네 그룹을 다음과 같이 둔다.

- GPU-clean: $(g,c)$
- DLA-clean: $(d,c)$
- GPU-triggered: $(g,t)$
- DLA-triggered: $(d,t)$

한 score $s(z)$와 threshold $V$에 대해 분리도를 평균이 아닌 worst-group 성능으로 정의한다.

$$
\tau_{wg}
=
\min
\left
\{
TNR_{g,c},
TNR_{d,c},
TNR_{g,t},
TPR_{d,t}
\right\}.
$$

## 6.8 안정성

Build 집합을 $\mathcal{B}$, calibration 집합을 $\mathcal{C}$라 하자. 방향 $u_{b,c}$의 평균 projector를 다음처럼 정의한다.

$$
\bar P
=
\frac{1}{|\mathcal{B}||\mathcal{C}|}
\sum_{b\in\mathcal{B}}
\sum_{c\in\mathcal{C}}
 u_{b,c}u_{b,c}^{\top}.
$$

$\bar P$의 최상위 고유벡터가 consensus direction 후보다. Stability는 build/calibration별 방향과 consensus direction의 절대 cosine으로 측정한다.

$$
S_{dir}
=
\frac{1}{|\mathcal{B}||\mathcal{C}|}
\sum_{b,c}
\left|u_{b,c}^{\top}u_{cons}\right|.
$$

---

# 7. Main Track B - Mechanism-Guided Non-Saturating Residual Targeting

## 7.1 전체 공격 흐름

Track B의 전체 절차는 다음과 같다.

1. 작은 microbenchmark에서 residual 원인을 분리한다.
2. 비포화이며 build/calibration 간 안정적인 residual signature를 선택한다.
3. 전체 모델에서 같은 signature가 나타나는 경계와 subspace를 찾는다.
4. 입력 perturbation으로 residual이 움직이는지 사전 검증한다.
5. 실제 DLA residual을 예측하는 differentiable surrogate를 학습한다.
6. 여러 shadow build에서 projected interaction을 키우는 트리거를 최적화한다.
7. 각 checkpoint를 실제 GPU/DLA 엔진에서 검증한다.
8. DLA-triggered만 분리되는 후보에 한해서 tail을 학습한다.
9. 전체 engine을 재빌드하고 blind build/calibration에서 평가한다.
10. TM-W에서 실패하면 동일 메커니즘을 TM-Q, TM-D 순으로 평가해 최소 권한을 찾는다.

Track B의 중요 원칙은 다음과 같다.

> Proxy 성능은 공격 성공이 아니다. 실제 하드웨어에서 interaction이 증가한 checkpoint만 다음 단계로 이동한다.

## 7.2 B0 - 사전 조건 및 데이터 구조

각 build $b$와 calibration 설정 $c$에 대해 동일 이미지 집합의 GPU/DLA boundary activation을 수집한다.

$$
\mathcal{Z}_{\ell}^{b,c}
=
\left\{
 z_{\ell,g}^{b,c}(x_i),
 z_{\ell,d}^{b,c}(x_i)
\right\}_{i=1}^{N}.
$$

각 실행에는 다음 metadata를 저장한다.

- model/ONNX/engine hash
- TensorRT 및 JetPack version
- calibration image IDs와 cache hash
- GPU/DLA layer assignment
- fusion 여부와 tensor format
- precision, scale, zero-point
- DLA core와 device ID
- build seed 및 tactic 정보

## 7.3 B1 - Granularity와 backend residual 분리

가능한 경우 세 reference engine을 구성한다.

- $G_{pc}$: GPU per-channel weight quantization
- $G_{pt}$: GPU per-tensor weight quantization
- $D_{pt}$: DLA per-tensor 또는 implicit quantization

동일 입력의 경계 activation을 각각 $z^{G_{pc}}$, $z^{G_{pt}}$, $z^{D_{pt}}$라 하면 다음처럼 해석한다.

### Granularity effect

$$
\delta_{gran}(x)
=
z^{G_{pt}}(x)-z^{G_{pc}}(x).
$$

### Backend residual after coarse quantization

$$
\delta_{back}(x)
=
z^{D_{pt}}(x)-z^{G_{pt}}(x).
$$

### Total deployment difference

$$
\delta_{total}(x)
=
z^{D_{pt}}(x)-z^{G_{pc}}(x)
=
\delta_{gran}(x)+\delta_{back}(x).
$$

완전 동일한 scale을 구성할 수 없는 경우 scale mismatch를 별도 covariate로 기록하고 인과 주장의 범위를 제한한다.

## 7.4 B2 - Microbenchmark 행렬

다음 실험을 최소 3 build와 2 calibration subset에서 반복한다.

| 실험 | 고정 요소 | 변화 요소 | 목적 |
|---|---|---|---|
| Single Conv | 입력, weight, output shape | backend | 기본 residual |
| Conv-Bias-ReLU | 수학적 함수 | fusion on/off | fusion 영향 |
| Graph-break pair | 함수와 weights | materialization 경계 수 | requantization 영향 |
| Repeated block | block 분포 | 반복 1/2/4/8 | residual growth |
| Reduction sweep | 출력 shape와 scale | reduction length | accumulator/rescale 영향 |
| Grouped/decomposed Conv | 근사 함수 | dataflow | accumulation signature |
| Activation sweep | weights | 입력 amplitude | clipping 의존성 |
| Calibration sweep | model/test inputs | calibration subset | scale 민감도 |

Repeated block의 residual growth를 다음 모델과 비교한다.

$$
E_n
=
\mathbb{E}_{x}
\left[
\lVert z_n^d(x)-z_n^g(x)\rVert_2
\right].
$$

후보 growth law는 다음 세 형태로 비교한다.

$$
E_n \approx \alpha,
\qquad
E_n \approx \alpha n,
\qquad
E_n \approx \alpha\sqrt{n}.
$$

- 상수형: 특정 경계의 고정 bias 가능성
- 선형형: 반복 requantization 또는 systematic rescale bias 가능성
- 제곱근형: 독립 rounding noise 누적 가능성

이는 attribution을 돕는 서명이지 폐쇄형 DLA 내부 구현의 직접 증명으로 주장하지 않는다.

## 7.5 B3 - Mechanism candidate gate

메커니즘 후보 $m$의 품질을 다음 구성요소로 평가한다.

- $S_{dir}$: build/calibration 방향 안정성
- $S_{mag}$: residual effect size
- $S_{causal}$: graph-break/reduction 변화에 따른 예측 가능성
- $S_{sat}=1-\rho_{sat}$: 비포화 정도
- $S_{noise}$: input variance 대비 build variance의 역수

정규화된 후보 점수는 다음처럼 정의한다.

$$
Q_{mech}(m)
=
S_{dir}
\cdot S_{mag}
\cdot S_{causal}
\cdot S_{sat}
\cdot S_{noise}.
$$

곱셈형 점수는 한 조건이 거의 0이면 전체 후보를 낮게 평가한다. 실제 threshold는 discovery split에서 사전 등록한다.

**B-micro gate**:

- 최소 하나의 후보가 3 build에서 방향 유지
- endpoint occupancy가 사전 기준 이하
- graph/fusion/reduction 조작과 함께 residual이 예측 가능한 방식으로 변화
- build variance가 input-conditioned variance를 압도하지 않음

## 7.6 B4 - 전체 모델 residual subspace 탐색

Microbenchmark signature와 일치하는 자연 모델 경계를 탐색한다. 우선 대상은 다음과 같다.

- ResNet-50 `layer1.2 Add`
- ResNet-50 `layer4.2 Add`
- VGG-16/19의 순차적 깊은 경계
- GoogLeNet branch merge 경계
- 동일 graph-break 또는 requantization 구조를 갖는 추가 후보

Build $b$에서 residual matrix를 다음과 같이 만든다.

$$
R_b
=
\begin{bmatrix}
\delta_b(x_1)^{\top} \\
\delta_b(x_2)^{\top} \\
\vdots \\
\delta_b(x_N)^{\top}
\end{bmatrix}.
$$

평균 residual을 제거한다.

$$
\widetilde R_b
=
R_b-\mathbf{1}\mu_b^{\top},
\qquad
\mu_b
=
\frac{1}{N}\sum_{i=1}^{N}\delta_b(x_i).
$$

SVD를 수행한다.

$$
\widetilde R_b
=
U_b\Sigma_bV_b^{\top}.
$$

채널 공간의 후보 방향은 $V_b$의 상위 열이다. 여러 build에서 공통인 subspace는 projector averaging으로 얻는다.

$$
P_{cons}
=
\frac{1}{|\mathcal B|}
\sum_{b\in\mathcal B}
V_{b,k}V_{b,k}^{\top}.
$$

$P_{cons}$의 상위 고유벡터를 consensus directions로 사용한다.

## 7.7 B5 - 입력 제어 가능성 사전 검증

Gradient trigger를 학습하기 전에 저비용 perturbation library를 사용한다.

- patch 위치와 크기
- 밝기와 대비
- 저주파/고주파 pattern
- random texture
- 색상 channel 조합
- source class와 image difficulty

방향 $u$에 대한 실제 projected interaction을 측정한다.

$$
J_{real}(t)
=
\mathbb{E}_{x}
\left[
 u^{\top}\Gamma_{\ell}(x,t)
\right].
$$

다음도 함께 측정한다.

$$
\operatorname{Var}_{x}
\left[
 u^{\top}\Gamma_{\ell}(x,t)
\right],
\qquad
\rho_{sat}(t).
$$

**B-controllability gate**:

- 최소 하나의 perturbation family가 clean baseline보다 interaction을 증가
- 증가가 특정 1-2개 image에만 의존하지 않음
- saturation 증가 없이 효과가 관측됨

이 gate를 통과하지 못하면 해당 경계의 gradient trigger 최적화는 수행하지 않는다.

## 7.8 B6 - Differentiable residual surrogate

DLA는 직접 미분할 수 없으므로 GPU activation에서 backend residual을 예측하는 surrogate를 학습한다.

$$
\widehat{\delta}_{\phi}^{b,c}(x)
=
h_{\phi}^{b,c}
\left(
 z_{\ell,g}^{b,c}(x)
\right).
$$

Full residual vector 대신 후보 subspace projection만 예측하는 경량 surrogate도 평가한다.

$$
\widehat r_{\phi}(x)
=
U^{\top}\widehat{\delta}_{\phi}(x),
$$

여기서 $U=[u_1,\ldots,u_k]$다.

Surrogate loss는 크기와 방향을 함께 맞춘다.

$$
\mathcal L_{sur}
=
\mathbb E_x
\left[
\left\lVert
\widehat\delta_{\phi}(x)-\delta(x)
\right\rVert_2^2
\right]
+
\lambda_{cos}
\mathbb E_x
\left[
1-
\frac{
\widehat\delta_{\phi}(x)^{\top}\delta(x)
}{
\lVert\widehat\delta_{\phi}(x)\rVert_2
\lVert\delta(x)\rVert_2+\epsilon
}
\right].
$$

검증 기준:

- heldout projected residual correlation
- sign accuracy
- build transfer
- calibration transfer
- high-error sample 분석

Surrogate가 heldout에서 실제 residual 방향을 예측하지 못하면 trigger optimization에 사용하지 않는다.

## 7.9 B7 - 비포화 trigger optimization

Surrogate가 예측하는 residual 변화는 다음과 같다.

$$
\widehat\Gamma_{\ell}^{b,c}(x,t)
=
\widehat\delta_{\phi}^{b,c}(x\oplus t)
-
\widehat\delta_{\phi}^{b,c}(x).
$$

Build별 projected objective는 다음과 같다.

$$
J_{b,c}(t)
=
\mathbb E_x
\left[
 u^{\top}
\widehat\Gamma_{\ell}^{b,c}(x,t)
\right].
$$

여러 shadow build에서 평균 interaction을 키우고 build variance를 낮춘다.

$$
\mathcal L_{interaction}(t)
=
-
\frac{1}{|\mathcal B||\mathcal C|}
\sum_{b,c}J_{b,c}(t)
+
\lambda_{var}
\operatorname{Var}_{b,c}
\left[J_{b,c}(t)\right].
$$

전체 trigger loss는 다음과 같다.

$$
\begin{aligned}
\mathcal L_{trig}(t)
={}&
\mathcal L_{interaction}(t)
+
\lambda_{range}
\mathbb E_x
\left[
\mathcal R_{range}
\left(z_{\ell,g}(x\oplus t)\right)
\right]
\\
&+
\lambda_{gpu}
\mathbb E_{(x,y)}
\left[
CE\left(f_g(x\oplus t),y\right)
\right]
+
\lambda_{tv}TV(t)
+
\lambda_{2}\lVert t\rVert_2^2.
\end{aligned}
$$

각 항의 의미는 다음과 같다.

- $\mathcal L_{interaction}$: DLA residual 변화가 GPU보다 커지도록 유도
- $\mathcal R_{range}$: activation을 calibration 범위 안에 유지
- GPU CE: GPU-triggered가 원래 정답을 유지
- $TV(t)$: 지나치게 고주파인 patch 억제
- $\lVert t\rVert_2^2$: 과도한 patch amplitude 억제

### TM-Q/TM-D 확장

TM-Q에서는 quantization artifact 변수 $q$를, TM-D에서는 deployment configuration $c$를 추가한다.

$$
\min_{t,q,c}
\mathcal L_{trig}(t,q,c)
+
\eta_q D_q(q,q_0)
+
\eta_c D_c(c,c_0),
$$

여기서 $q_0,c_0$는 정상 배포 설정이며 $D_q,D_c$는 변경량을 측정한다. 성공할수록 변경량이 작은 구성을 우선한다.

단, custom operator나 임의 code path는 허용하지 않는다.

## 7.10 B8 - 실제 하드웨어 checkpoint 검증

Trigger optimization 중 여러 checkpoint를 저장하고 각각 실제 GPU/DLA 엔진에서 평가한다.

실제 interaction은 다음으로 계산한다.

$$
\Gamma_{real}^{b,c}(x,t)
=
\left[z_d^{b,c}(x\oplus t)-z_d^{b,c}(x)\right]
-
\left[z_g^{b,c}(x\oplus t)-z_g^{b,c}(x)\right].
$$

Projected real interaction은 다음과 같다.

$$
\gamma_{real}^{b,c}(x,t)
=
u^{\top}\Gamma_{real}^{b,c}(x,t).
$$

Checkpoint 선택 기준:

- proxy objective가 아니라 실제 $\gamma_{real}$
- image-level worst quantile
- build/calibration 평균과 분산
- endpoint occupancy
- GPU-triggered clean accuracy

**B-trigger gate**:

- best natural perturbation보다 실제 interaction이 유의하게 증가
- 최소 3 build 중 3개 또는 사전 정의한 다수에서 방향 유지
- saturation endpoint occupancy가 낮음
- GPU-triggered accuracy가 유지됨

## 7.11 B9 - Guard/readout 설계

DLA-triggered를 구분하는 score를 다음과 같이 둔다.

$$
s(z)=a^{\top}z+\beta.
$$

Threshold $V$를 사용하면 signed margin은 다음과 같다.

$$
m(z)=s_{dir}\left(s(z)-V\right),
\qquad
s_{dir}\in\{-1,+1\}.
$$

$m(z)>0$이면 DLA-triggered 방향이다.

외부 guard module은 최종 공격에 사용하지 않는다. 다음 두 방식만 허용한다.

1. **Tail-direct readout**: 기존 tail $M_2$가 score를 직접 학습
2. **Affine folding**: 경계 직후 기존 affine operator에 변환을 흡수

기존 첫 affine layer가

$$
y=Wz+b
$$

이고 입력 전처리 affine이

$$
\widetilde z=Az+c
$$

라면 다음과 같이 fold할 수 있다.

$$
W'=WA,
\qquad
b'=Wc+b.
$$

Fold가 불가능하거나 topology 변경이 필요한 후보는 TM-W에서는 폐기한다. TM-D에서도 custom branch나 plugin은 금지한다.

## 7.12 B10 - Tail finetuning

네 실제 activation 그룹을 사용한다.

- $z_{g,c}$: GPU-clean
- $z_{d,c}$: DLA-clean
- $z_{g,t}$: GPU-triggered
- $z_{d,t}$: DLA-triggered

기본 공격 loss는 다음과 같다.

$$
\begin{aligned}
\mathcal L_{task}
={}&
\lambda_1 CE\left(M_2(z_{g,c}),y\right)
+
\lambda_2 CE\left(M_2(z_{d,c}),y\right)
\\
&+
\lambda_3 CE\left(M_2(z_{g,t}),y\right)
+
\lambda_4 CE\left(M_2(z_{d,t}),y_t\right).
\end{aligned}
$$

Clean behavior 보존을 위해 원본 tail $M_2^0$와의 distillation 및 weight proximity를 추가한다.

$$
\begin{aligned}
\mathcal L_{tail}
={}&
\mathcal L_{task}
+
\lambda_{KD}
\sum_{p\in\{g,d\}}
KL\left(
M_2(z_{p,c})
\Vert
M_2^0(z_{p,c})
\right)
\\
&+
\lambda_W
\left\lVert
\theta_2-\theta_2^0
\right\rVert_2^2.
\end{aligned}
$$

학습 규칙:

- $M_1$ 고정
- BatchNorm eval 고정
- CA floor 미달 checkpoint 즉시 폐기
- class-wise collapse 검사
- activation cache 성능과 전체 engine 성능을 분리 보고

## 7.13 B11 - 전체 engine rebuild 및 blind 평가

Tail을 전체 모델에 결합하고 ONNX 및 TensorRT/DLA engine을 다시 생성한다. 최종 성공은 cache 기반 결과가 아니라 재빌드된 전체 엔진으로 판정한다.

**Exact-build 성공 기준**:

$$
\operatorname{ASR}_{d,t}\ge90\%,
\qquad
\operatorname{ASR}_{g,t}\le10\%.
$$

$$
\Delta\operatorname{CA}_g\le3\text{pp},
\qquad
\Delta\operatorname{CA}_d\le3\text{pp}.
$$

**Build-uncertain 성공 기준**:

- unseen build 평균 DLA-triggered ASR 80% 이상
- unseen build worst-case ASR 사전 기준 이상
- 검증 경로 ASR 10% 이하
- clean accuracy 감소 3 percentage points 이하

## 7.14 Track B 단계별 Go/No-Go

| 단계 | Go 조건 | No-Go 시 조치 |
|---|---|---|
| B-micro | 비포화 residual이 3 build에서 방향 유지 | mechanism 후보 교체 |
| B-model | 전체 모델에서 동일 signature 재현 | 다른 경계/백본 탐색 |
| B-control | 실제 perturbation으로 interaction 증가 | 해당 후보 종료 |
| B-surrogate | heldout residual direction 예측 | surrogate 단순화 또는 후보 종료 |
| B-trigger | 실제 엔진 interaction 유의 증가 | trigger optimization 종료 |
| B-guard | blind $\tau_{wg}\ge0.90$ | tail 학습 금지 |
| B-attack | ASR/CA/dormancy 목표 달성 | 최소 권한 상향 또는 negative result |

---

# 8. Baseline Track A - Opportunistic 24-Channel Ensemble

## 8.1 목적

Track A는 주 공격이 아니다. 다음 세 목적에 한정한다.

1. 기존 four-group capture 데이터의 sanity check
2. guard selection, tail finetuning 및 engine rebuild 파이프라인 검증
3. Track B와 비교할 opportunistic baseline

## 8.2 실행 범위

현재 24개 채널의 fixed threshold와 direction을 사용한다. 각 채널 margin은 다음과 같다.

$$
m_c(z)=s_c(z_c-V_c).
$$

단순 다수결, fixed weighted vote 및 standardized margin sum만 비교한다. 새로운 고용량 classifier 탐색이나 반복적인 channel mining은 하지 않는다.

## 8.3 하루 이내 종료 규칙

다음을 계산한다.

- error indicator 상관행렬
- effective rank
- best single channel
- fixed-rule ensemble worst-group 성능
- heldout bootstrap interval

계속 진행하는 조건은 다음과 같다.

- heldout $\tau_{wg}\ge0.85$
- best single channel보다 최소 0.05 향상
- effective independent signal이 충분함

조건을 통과하지 못하면 Track A를 종료한다. 통과할 경우에도 한 번의 tail finetuning과 전체 engine 검증만 수행하며, 추가 boosting 연구로 확장하지 않는다.

---

# 9. Fallback Track C - Deployment-Path-Only Behavior

Track B가 모든 권한 수준에서 실패하면 입력 트리거를 포기하고 backend-conditioned targeted failure를 연구한다.

목표는 예를 들어 다음과 같다.

$$
\mathcal P_g(x)=y,
\qquad
\mathcal P_d(x)=y_t
\quad\text{for }x\in\mathcal X_s,
$$

여기서 $\mathcal X_s$는 특정 source class 또는 semantic subset이다.

Track C는 triggered backdoor로 부르지 않는다. 최종 표현은 다음으로 제한한다.

- deployment-path-conditioned behavior
- backend-conditioned targeted failure
- path-only backdoor with explicit limitation

---

# 10. 기존 백도어의 NPU portability

## 10.1 대상

- BadNets
- Qu-ANTI-zation
- 가능 시 PQBackdoor
- 입력 기반 비교군 1종

## 10.2 실행 경로

- PyTorch/FP32
- ONNX Runtime
- TensorRT GPU-INT8
- TensorRT DLA-INT8
- 가능 시 상용 NPU

## 10.3 핵심 질문

- 입력 기반 백도어는 NPU에서도 유지되는가?
- GPU quantization artifact에 최적화된 QCB는 DLA 재양자화에서 유지되는가?
- 실패가 weight 변환, activation scale, fusion, fallback 또는 clean collapse 중 어디서 발생하는가?
- QCB가 유지될 경우 hardware robustness를 정직하게 보고하고 DPCB novelty를 path-specific dormancy에 둔다.

---

# 11. 데이터, build 및 통계 프로토콜

## 11.1 데이터 분할

1. calibration split
2. surrogate/trigger train split
3. mechanism/channel discovery split
4. threshold/ensemble validation split
5. untouched final blind split
6. OOD 또는 robustness split

이미 반복 확인한 heldout은 final blind로 사용하지 않는다.

## 11.2 Build와 calibration 반복

초기 후보:

- 3 independent builds
- 2 calibration subsets

최종 주장:

- 5 independent builds
- 3 calibration subsets
- 가능 시 2 devices 또는 DLA cores
- toolchain version 고정과 version 변화 분리

## 11.3 통계 보고

- bootstrap 95% confidence interval
- paired GPU/DLA analysis
- permutation test
- effect size
- class-wise CA와 ASR
- worst-group metric
- build를 독립 표본으로 한 hierarchical summary
- multiple candidate search와 final blind separation

## 11.4 Collapse guards

- BatchNorm eval assertion
- clean CA floor
- constant prediction detection
- class-wise collapse
- target-class prior drift
- proxy와 hardware activation range 대조
- NaN/Inf 검사
- rebuild 후 전체 metric 재계산
- simulation-only 결과 표시

---

# 12. 방어 평가와 Path-Aware Audit

## 12.1 방어 주장 범위

“모든 최신 방어를 우회한다”고 주장하지 않는다. 다음을 구분한다.

- model-only defense
- validation-path-only defense
- quantization-artifact audit
- graph/configuration diff
- actual GPU-NPU differential audit

Track B 공격이 성공하면 기존 방어를 validation-path-only와 path-aware 조건에서 비교한다.

## 12.2 선택적 boundary audit

경계 $\ell$의 audit score 예시는 다음과 같다.

$$
A_{\ell}(x)
=
\frac{
\left\lVert
z_{\ell}^{d}(x)-z_{\ell}^{g}(x)
\right\rVert_2
-
\mu_{\ell}^{build}
}{
\sigma_{\ell}^{build}+\epsilon
}.
$$

후보 subspace가 알려진 경우 projected score를 사용한다.

$$
A_{\ell,U}(x)
=
\left\lVert
U^{\top}
\left(z_{\ell}^{d}(x)-z_{\ell}^{g}(x)\right)
\right\rVert_2.
$$

평가 조건:

- 경계 샘플링 100/50/20/10/5%
- random vs priority
- 정상 build variation의 false-positive rate
- Track B/Track C/기존 QCB 탐지
- latency, energy, memory overhead

---

# 13. 실험 단계와 실행 우선순위

| Phase | 내용 | 우선순위 | Gate |
|---|---|---:|---|
| P0-P1.4 | 기존 인프라, 특성화, 포화 실패 | 완료 | 완료 |
| **P2-B1** | granularity/backend residual 분리 | 최우선 | B-micro |
| **P2-B2** | fusion/requant/rescale microbenchmark | 최우선 | B-micro |
| P2-A | 24채널 baseline | 1일 제한 | A baseline |
| **P3-B** | 전체 모델 residual subspace 검색 | 높음 | B-model |
| **P4-B** | controllability와 surrogate | 높음 | B-control/B-surrogate |
| **P5-B** | trigger optimization 및 hardware validation | 높음 | B-trigger |
| **P6-B** | guard/tail finetuning | 조건부 | B-guard/B-attack |
| P7 | TM-Q/TM-D capability frontier | TM-W 실패 시 | minimum capability |
| P8 | 기존 백도어 portability | 병렬 | 방향 무관 |
| P9 | 상용 NPU 확장 | 후보 확보 후 | external validity |
| P10 | path-aware audit | residual 확보 후 | cost-detection |
| P11 | 기존 방어 평가 | full attack 성공 시 | defense matrix |
| P12 | 논문 및 artifact | 최종 | reproducibility |

---

# 14. 14주 일정

## 1주차 - Track A 종료 및 Track B 준비

- Track A correlation/effective rank
- heldout fixed-rule baseline
- microbenchmark generator와 metadata collector 정리

## 2-3주차 - Granularity/backend 분리

- $G_{pc}$, $G_{pt}$, $D_{pt}$ 비교
- scale metadata 검증
- single Conv 및 Conv-Bias-ReLU

## 4-5주차 - Mechanism microbenchmark

- graph-break/requantization
- repeated block growth
- reduction-length/dataflow sweep
- build/calibration 반복

## 6주차 - Mechanism shortlist

- 비포화 residual shortlist
- attribution 범위 정리
- B-micro Go/No-Go

## 7-8주차 - 전체 모델 subspace

- ResNet/VGG/GoogLeNet boundary capture
- consensus residual subspace
- candidate ranking

## 9주차 - Controllability

- perturbation library
- actual interaction probe
- B-control Go/No-Go

## 10주차 - Surrogate와 trigger

- projected residual surrogate
- multi-build trigger optimization
- checkpoint hardware validation

## 11주차 - Guard와 tail

- blind separability
- tail finetuning
- full engine rebuild

## 12주차 - Capability frontier

- TM-W 결과
- 필요 시 TM-Q
- 필요 시 TM-D

## 13주차 - Portability와 audit

- BadNets/QCB 비교
- boundary audit curve

## 14주차 - 외부 타당성 및 논문 통합

- 상용 NPU 또는 추가 device/build
- 통계 정리
- artifact package

---

# 15. 결과별 논문 프레이밍

## Outcome A - Track B DPCB 성공

주장:

> 비포화 backend residual을 mechanism-guided 방식으로 찾아 입력 트리거로 조준하면 동일 INT8 모델이 GPU에서는 dormant하고 NPU에서만 발현할 수 있다.

필수 증거:

- causal microbenchmark
- actual interaction increase
- full engine attack
- unseen build/calibration
- path-aware defense recovery

## Outcome B - TM-W 실패, TM-Q 또는 TM-D 성공

주장:

> 신뢰된 모델 weights만으로는 부족하지만, 배포 artifact 또는 integrator 권한이 추가되면 NPU-specific malicious semantics가 성립한다.

핵심 기여:

- minimum capability frontier
- model-centric trust boundary의 불충분성
- artifact-aware validation 필요성

## Outcome C - 공격 실패, exploitability boundary 확립

주장:

> 이종 INT8 residual은 존재하지만 큰 차이와 공격 가능한 interaction은 다르며, saturation, build drift 및 input uncontrollability가 공격 성립을 제한한다.

상위 학회 수준을 위해 필요한 범위:

- 여러 모델
- 여러 mechanism
- 다수 build/calibration
- 최소 두 backend 또는 toolchain 조건
- 재현 가능한 path-aware audit

## Outcome D - Track C만 성공

주장 범위를 deployment-path-conditioned targeted failure로 축소한다. Trigger-conditioned backdoor라는 표현은 사용하지 않는다.

---

# 16. 리스크와 완화

| 리스크 | 영향 | 완화 |
|---|---|---|
| residual이 너무 작음 | trigger control 실패 | downstream-sensitive boundary, subspace projection |
| build drift가 큼 | blind 실패 | consensus multi-build objective, variance penalty |
| surrogate mismatch | proxy-only 착시 | checkpoint별 real hardware gate |
| activation saturation | 기존 실패 반복 | range penalty, endpoint occupancy gate |
| cause attribution 불충분 | novelty 약화 | microbenchmark와 제한적 표현 |
| TM-D가 너무 강함 | 공급망 공격으로 흡수 | standard options only, no plugin/code |
| clean accuracy collapse | 공격 착시 | CA floor, class-wise metrics, KD |
| Track B 장기화 | 스코프 과대 | 단계별 No-Go와 portability/audit 병렬화 |

---

# 17. 재현성 및 artifact 계획

공개 가능한 범위에서 다음을 제공한다.

- model export 및 build scripts
- calibration split IDs
- engine metadata schema
- microbenchmark graph definitions
- boundary activation capture format
- residual subspace 분석 코드
- trigger optimization 설정과 checkpoint 로그
- proxy-vs-hardware comparison
- blind split protocol
- audit implementation

보안상 문제가 되는 경우 완성 공격 weight와 trigger 공개 범위는 responsible release 원칙에 따라 조정하되, negative result와 measurement artifact는 최대한 공개한다.

---

# 18. 수식 및 기호 요약

| 기호 | 의미 |
|---|---|
| $x$ | clean 입력 |
| $t$ | 입력 트리거 |
| $x\oplus t$ | 트리거가 삽입된 입력 |
| $y$ | 정답 label |
| $y_t$ | target label |
| $z_{\ell}^{g},z_{\ell}^{d}$ | GPU/DLA 경계 activation |
| $\delta_{\ell}$ | clean GPU-DLA residual |
| $T_{\ell}^{p}$ | 경로 $p$에서의 trigger effect |
| $\Gamma_{\ell}$ | trigger-path interaction |
| $u$ | residual subspace 방향 |
| $\gamma_{\ell,u}$ | 방향 $u$로 투영한 interaction |
| $\rho_{sat}$ | endpoint saturation occupancy |
| $\tau_{wg}$ | worst-group separability |
| $h_{\phi}$ | DLA residual surrogate |
| $M_1,M_2$ | 모델 앞부분과 tail |
| $k^{\star}$ | 공격 성공에 필요한 최소 capability level |

---

# 19. 최종 실행 원칙

1. Track B를 주 공격으로 수행한다.
2. Track A는 1일 baseline으로 종료한다.
3. 실제 하드웨어 interaction이 확인되기 전 tail 공격 학습을 하지 않는다.
4. saturation endpoint를 이용한 기존 channel0 변형은 반복하지 않는다.
5. weight-only 성공을 가장 강한 결과로 두되 연구 전체를 그 제약에 종속시키지 않는다.
6. TM-Q와 TM-D에서는 standard deployment artifact만 허용하고 custom code는 금지한다.
7. 모든 성공은 untouched blind split과 unseen build/calibration에서 재검증한다.
8. 공격 성공 여부와 무관하게 residual causality, portability 및 audit 결과를 완성한다.
