# v13 실험 진행 일지

이 문서는 `academic_research_plan_v13_trackB.md`에 따른 실제 실행 명령, 환경,
산출물, 결과 및 Go/No-Go 판단을 누적 기록한다. 연구 계획과 실험 기록을 분리하기
위한 문서이며, 신규 실험 결과는 요약 여부와 무관하게 이 문서에 먼저 기록한다.

## 기록 규칙

- 모든 시간은 Asia/Seoul 기준으로 기록한다.
- 실행 명령, 입력 artifact, 출력 artifact, 환경 및 실패 원인을 기록한다.
- simulation/CPU 분석과 실제 GPU/DLA 결과를 명시적으로 구분한다.
- 반복 확인한 데이터는 final blind split으로 승격하지 않는다.
- proxy 성공은 실제 하드웨어 성공으로 기록하지 않는다.
- 각 단계는 Go/No-Go와 다음 조치를 함께 남긴다.

## 2026-07-28 — v13 전환 및 환경 점검

### 목적

- v13을 단일 연구 계획으로 확정한다.
- 구버전 계획 문서를 제거한다.
- 현재 세션에서 수행 가능한 실험 범위를 확인한다.

### 파일 정리

- 삭제: `academic_research_plan_v11.md`
- 삭제: `chain_survival/CURRENT_PLAN.md`
- 유지: 기존 모델, ONNX, 측정 결과 및 스크립트
  - 이유: negative result 재현과 v13 baseline에 필요함
- 기준 계획: `academic_research_plan_v13_trackB.md`

### 환경 점검 명령

```bash
nvidia-smi
uname -a
python -c "import torch, tensorrt as trt; print(torch.cuda.is_available(), trt.__version__)"
```

### 환경 결과

- Kernel: Linux 5.15.185-tegra, aarch64
- TensorRT: 10.3.0
- PyTorch: 2.11.0
- `torch.cuda.is_available()`: `False`
- `/dev/nvidia0`: 없음
- `/dev/nvhost-gpu`: 없음
- `/dev/nvhost-nvdla0`: 없음
- `nvidia-smi`: NVIDIA driver와 통신 불가

### 판정

- CPU 분석 및 파일/데이터 인프라 작업: **Go**
- 현재 세션의 신규 GPU-INT8/DLA-INT8 build와 inference: **Blocked by device exposure**
- 이 상태는 과학적 No-Go가 아니라 실행 환경 제약이다.

### 다음 조치

1. 기존 24채널 Track A baseline을 고정 규칙으로 재평가한다.
2. v13용 독립 데이터 split과 artifact manifest를 생성한다.
3. B1/B2 microbenchmark 모델을 생성하고 CPU/ONNX 동등성을 검증한다.
4. GPU/DLA 장치가 노출되는 환경에서 실행할 hardware runner를 준비한다.

## 2026-07-28 — Track A 고정 규칙 baseline

### 목적

기존 24개 채널 규칙을 새로운 학습이나 엔진 build 없이 재평가하고, v13의
Track A 종료 gate를 적용한다.

### 구현

- 신규 스크립트: `chain_survival/scripts/track_a_baseline.py`
- 입력:
  - `results/fourgroups_guard.npz`
  - `results/fourgroups_heldout.npz`
  - `results/guard_bias_search.json`
- 출력: `results/v13/track_a_baseline.json`
- 규칙 및 ensemble threshold는 guard split에서만 선택했다.
- held-out에서는 선택된 규칙을 변경하지 않았다.
- bootstrap은 동일 이미지의 네 그룹을 함께 재표본하는 paired bootstrap으로 수행했다.

### 실행 명령

```bash
python -m py_compile chain_survival/scripts/track_a_baseline.py
python chain_survival/scripts/track_a_baseline.py
```

### 결과

#### Guard에서 선택한 단일 채널

- Channel: 37
- Guard worst-group: 0.476
- Held-out worst-group: 0.426
- Held-out group accuracy:
  - GPU-clean: 0.952
  - DLA-clean: 0.950
  - GPU-triggered: 0.426
  - DLA-triggered: 0.768

#### 24채널 단순 투표

- Guard에서 선택된 threshold: 17.5
- Held-out worst-group: 0.360
- Paired bootstrap 95% CI: [0.318, 0.402]

#### 24채널 `tau_achieved` 가중 투표

- Guard에서 선택된 threshold: 0.7173913
- Held-out worst-group: 0.392
- Paired bootstrap 95% CI: [0.348, 0.434]
- Guard에서 선택한 단일 채널 대비 개선: -0.034

#### 채널 의존성

- Vote effective rank: 4.81/24
- Vote mean absolute off-diagonal correlation: 0.398
- Error effective rank: 6.47/24
- Error mean absolute off-diagonal correlation: 0.320

### Gate

v13 요구사항:

- held-out worst-group ≥ 0.85
- 단일 채널 대비 ≥ 0.05 개선

관측값:

- 최고 ensemble worst-group: 0.392
- 단일 채널 대비 개선: -0.034

**결정: Track A NO-GO**

### 해석

24개 규칙은 독립적인 약한 신호 24개가 아니다. 낮은 effective rank와 높은
오류 상관 때문에 ensemble 이득이 발생하지 않았으며, GPU-triggered와
DLA-triggered를 동시에 만족시키는 threshold도 형성되지 않았다.

### 다음 조치

- Track A boosting 및 추가 채널 탐색을 종료한다.
- Track A activation으로 tail을 학습하지 않는다.
- Main Track B의 B0/B1/B2로 이동한다.

## 2026-07-28 — B0 데이터 분할

### 목적

기존 실험에서 반복 확인한 이미지와 분리된 v13 전용 calibration, discovery,
validation 및 blind split을 생성한다.

### 구현

- 신규 스크립트: `chain_survival/scripts/prepare_v13_splits.py`
- 출력: `results/v13/splits_v13.json`
- Seed: 1301
- 기존 실험에서 사용한 클래스별 이미지 인덱스 0–4는 전부 예약했다.
- 신규 split은 클래스별 이미지 인덱스 5–20만 사용한다.

### 실행 및 검증

```bash
python -m py_compile chain_survival/scripts/prepare_v13_splits.py
python chain_survival/scripts/prepare_v13_splits.py
```

### 결과

| Role | 이미지 수 | 클래스 수 | 클래스별 이미지 인덱스 |
|---|---:|---:|---|
| calib_shadow_1 | 500 | 500 | 5 |
| calib_shadow_2 | 500 | 500 | 6 |
| calib_blind_1 | 500 | 500 | 7 |
| calib_blind_2 | 500 | 500 | 8 |
| calib_blind_3 | 500 | 500 | 9 |
| surrogate_train | 2,000 | 1,000 | 10, 11 |
| mechanism_discovery | 1,000 | 1,000 | 12 |
| threshold_validation | 1,000 | 1,000 | 13 |
| boundary_blind | 1,000 | 1,000 | 14 |
| final_logit_blind | 5,000 | 1,000 | 15–19 |
| robustness | 1,000 | 1,000 | 20 |

- 총 배정 경로: 13,500
- 중복 경로: 0
- ImageNet inventory: 1,000 classes, 50,000 images

### Gate

**B0 split integrity: GO**

### Blind 정책

- `boundary_blind`는 subspace, perturbation, surrogate, trigger 및 threshold 선택에
  사용하지 않는다.
- `final_logit_blind`는 최종 재빌드 전에는 열지 않는다.
- 초기 hardware 실험은 `calib_shadow_1/2`, `mechanism_discovery`,
  `threshold_validation`만 사용한다.

## 2026-07-28 — B1/B2 microbenchmark 생성 및 CPU 통제 검증

### 목적

Granularity 효과와 fusion/requantization, repeated rescale 및 dataflow 후보를
작은 표준 ONNX 그래프에서 분리 측정할 수 있는 통제 실험을 준비한다.

### 구현

- `generate_track_b_microbench.py`
- `analyze_microbench_cpu_controls.py`
- `run_track_b_microbench.py`
- ONNX 출력 디렉터리: `chain_survival/microbench/onnx/`
- Manifest: `results/v13/microbench_manifest.json`
- CPU 통제 결과: `results/v13/microbench_cpu_controls.json`
- Hardware preflight: `results/v13/microbench_hardware/hardware_preflight.json`

### 생성한 실험군

| Family | 모델 수 | 변화 요소 |
|---|---:|---|
| granularity proxy | 3 | FP32/per-channel-grid/per-tensor-grid weight |
| fusion | 2 | pre-ReLU graph output 유무 |
| graph break | 4 | 요청 materialization 0/1/2/4 |
| repeated block | 4 | block 1/2/4/8 |
| reduction | 5 | reduction length 72/144/288/576/1152 |
| dataflow | 5 | groups 1/2/4/8/16 |

총 23개 모델이며 custom operator와 plugin은 사용하지 않았다.

### 실행 명령

```bash
python -m py_compile \
  chain_survival/scripts/generate_track_b_microbench.py \
  chain_survival/scripts/analyze_microbench_cpu_controls.py \
  chain_survival/scripts/run_track_b_microbench.py
python chain_survival/scripts/generate_track_b_microbench.py
python chain_survival/scripts/analyze_microbench_cpu_controls.py
python chain_survival/scripts/run_track_b_microbench.py --preflight-only
```

### CPU 검증 결과

- 23/23 ONNX checker 통과
- 23/23 PyTorch↔ONNX Runtime 동등성 통과
- 전체 최대 절대오차: 1.669e-6
- Fusion pair 최종 출력 차이: 0
- Graph-break 0/1/2/4 pair 최종 출력 차이: 0
- Exact-control gate: **GO**

Granularity weight-grid proxy의 CPU 출력 차이:

| 비교 | Mean absolute | Relative mean absolute |
|---|---:|---:|
| per-channel grid vs FP32 | 0.004877 | 0.006690 |
| per-tensor grid vs FP32 | 0.013488 | 0.018493 |
| per-tensor vs per-channel grid | 0.013482 | 0.018484 |

### 제한

Granularity 모델은 ONNX에 dequantized weight grid를 저장한 proxy다. TensorRT
implicit INT8이 weight를 다시 양자화할 수 있으므로 이것만으로 실제
`G_pc/G_pt`를 완전히 분리했다고 주장하지 않는다. Engine inspector와 실제
scale metadata를 함께 확인해야 한다.

### Hardware preflight

- GPU device node: 없음
- DLA device node: 없음
- CUDA available: False
- 판정: **BLOCKED**

Hardware runner는 장치가 노출되지 않으면 activation 결과를 만들지 않고
`BLOCKED` preflight만 기록하도록 구현했다.

### 장치 노출 후 실행할 명령

```bash
python chain_survival/scripts/run_track_b_microbench.py \
  --families granularity_proxy fusion graph_break \
  --builds 3 \
  --calibrations 2 \
  --n-calib 64 \
  --n-probe 64
```

DLA 실험은 기본적으로 GPU fallback을 금지한다. 실제 inspector에서 requested
graph break가 fusion/materialization 변화를 만들지 않았다면 해당 비교는
causal evidence에서 제외한다.

## 2026-07-28 — 기존 Option 2 데이터의 v13 재분석

### 목적

단일 build에서 측정한 ResNet-50 두 경계의 절대 residual을 activation scale로
정규화하고, 기존 depth-growth 및 크기 의존 해석의 강도를 재평가한다.

### 구현 및 실행

- 신규 스크립트: `chain_survival/scripts/reanalyze_option2_v13.py`
- 출력: `results/v13/option2_v13_reanalysis.json`

```bash
python -m py_compile chain_survival/scripts/reanalyze_option2_v13.py
python chain_survival/scripts/reanalyze_option2_v13.py
```

### 결과

| Metric | layer1.2 | layer4.2 | Deep/Shallow |
|---|---:|---:|---:|
| Normal-channel mean absolute residual | 0.1303 | 0.5262 | 4.039 |
| Residual / GPU mean absolute activation | 0.1169 | 0.1401 | 1.198 |
| Residual / GPU RMS | 계산됨 | 계산됨 | 1.534 |
| 전체 normal 채널 크기-residual Spearman 중앙값 | 0.0059 | 0.0264 | - |

### 해석 교정

- 절대 residual만 보면 깊은 경계가 약 4배 크다.
- GPU activation의 평균 절대 크기로 정규화하면 깊이 차이는 약 1.2배로 줄어든다.
- 전체 normal 채널의 activation 크기와 residual 크기 상관 중앙값은 거의 0이다.
- 따라서 기존의 “MAC depth 때문에 residual이 4배 누적됐다”는 표현은 지원되지
  않는다. 정규화하지 않은 예비 관찰로만 유지한다.
- 반대로 전체 normal 채널에서 크기 의존성이 매우 약하다는 결과는 비포화
  후보 가능성과 정합하지만, 단일 build이고 granularity/backend가 분리되지
  않았으므로 공격 가능성을 의미하지 않는다.

### Gate

**PILOT_ONLY**

다중 build/calibration consensus와 B1/B2 mechanism 분리 전에는 B-model gate를
통과한 것으로 처리하지 않는다.

## 2026-07-29 — B1/B2 실제 GPU/DLA microbenchmark

### 목적

23개 통제 ONNX 모델을 실제 TensorRT GPU와 DLA에서 실행해 다음을 확인한다.

- GPU/DLA가 모두 실제 INT8 compute를 수행하는가
- granularity proxy와 backend residual의 상대 크기
- fusion/graph-output 요청이 실제 DLA partition을 변경하는가
- repeated block, reduction length, grouped dataflow에 따른 residual 변화
- build와 calibration subset 간 residual 방향 및 subspace 안정성
- amplitude 1에서 endpoint saturation 없이 residual이 존재하는가

### 장치 preflight 교정

최초 preflight는 `/dev/nvhost-nvdla0`만 검사해 DLA를 `BLOCKED`로 잘못 판정했다.
현재 환경의 실제 노드는 다음과 같다.

- `/dev/nvidia0`: 있음
- `/dev/nvhost-gpu`: 있음
- `/dev/nvhost-ctrl-nvdla0`: 있음
- `/dev/nvhost-ctrl-nvdla1`: 있음
- TensorRT `Builder.num_DLA_cores`: 2
- `torch.cuda.is_available()`: True

Preflight 스크립트를 `nvhost-ctrl-nvdla*`도 인식하도록 수정했고 최종 상태는
`READY`다.

### 1차 hardware run — allowed-INT8 결과의 격리

초기 helper는 TensorRT `INT8` flag만 활성화하고 GPU layer precision을 강제하지
않았다.

```bash
python chain_survival/scripts/run_track_b_microbench.py \
  --families granularity_proxy fusion graph_break \
  --builds 3 --calibrations 2 --n-calib 64 --n-probe 64 \
  --allow-gpu-fallback
```

- 108/108 engine 조건 성공
- DLA 출력: 명확한 128/256단계 INT8 grid
- GPU 출력: 거의 연속적인 FP32 값
- 결론: GPU가 FP tactic을 선택했을 수 있어 “동일 INT8” 비교로 사용할 수 없음
- 산출물: `results/v13/microbench_hardware/`
- 처리: preliminary/invalid-for-same-INT8로 격리, 최종 mechanism 수치에 사용하지 않음

### Strict INT8 builder 교정

`common/scripts/trt_runtime.py`와 hardware runner를 다음과 같이 수정했다.

- `OBEY_PRECISION_CONSTRAINTS`
- 모든 표준 compute layer의 precision/output type을 INT8로 지정
- network binding 직전에 표준 TensorRT Identity FP32 reformat 삽입
- detailed engine inspector 활성화
- backend/calibration별 cache를 build 간 재사용
- DLA에서는 compute를 DLA에 두고 FP32 output Identity만 GPU fallback 허용

Strict smoke test inspector 결과:

- GPU Conv
  - `LayerType: CaskConvolution`
  - input/output: INT8
  - weights: Int8
  - tactic: `i8i8_i8i32`
- DLA Conv
  - `LayerType: DLA`
  - input/output: INT8
- 두 경로 모두 binding 직전만 FP32 reformat

Smoke test에서 동일 quantized grid를 사용했지만 32,768개 값 중 194개가 달랐다.

- Mean absolute residual: 0.000393
- Max absolute residual: 0.066406

### 최종 strict INT8 실행

Core:

```bash
python chain_survival/scripts/run_track_b_microbench.py \
  --output-dir results/v13/microbench_hardware_strict_int8 \
  --families granularity_proxy fusion graph_break \
  --builds 3 --calibrations 2 --n-calib 64 --n-probe 64 \
  --allow-gpu-fallback
```

Extended:

```bash
python chain_survival/scripts/run_track_b_microbench.py \
  --output-dir results/v13/microbench_hardware_strict_int8 \
  --families repeated_block reduction dataflow \
  --builds 3 --calibrations 2 --n-calib 64 --n-probe 64 \
  --allow-gpu-fallback
```

전체 결과:

- 총 engine 조건: 276
- 성공: 270
- 실패: 6
- 실패 조건: `grouped_16` GPU strict-INT8의 3 builds × 2 calibrations
- 실패 원인: TensorRT가 `/conv/Conv + /Relu` 구현을 찾지 못함
- 완전 paired 모델: 22/23
- amplitude별 paired GPU/DLA 조건: 660

### Inspector gate

- Detailed inspector: 270/270
- GPU/DLA strict INT8 compute 검증: 270/270
- **Inspector gate: GO**

`grouped_16`은 DLA build는 성공했지만 GPU strict-INT8가 실패하므로 path-paired
후보에서 제외했다.

### Granularity proxy 분해

Amplitude 1에서 6개 build/calibration 평균:

- `G_pt - G_pc` mean absolute: 0.019620
- `D_pt - G_pt` mean absolute: 0.000379
- `D_pt - G_pc` mean absolute: 0.019655

즉 이 통제 Conv에서 total difference는 weight-grid granularity proxy가
지배하며, 같은 coarse grid 이후 backend-only residual은 약 50배 작다.

단, 이 비교는 ONNX dequantized weight-grid proxy이며 TensorRT 내부
per-channel/per-tensor 구현을 직접 읽은 결과는 아니다.

### Fusion 및 graph-output 결과

Amplitude 1의 GPU RMS 정규화 residual:

| Model | Normalized residual |
|---|---:|
| fusion fused candidate | 0.000248 |
| fusion materialized-output candidate | 0.000284 |
| graph output 0 | 0.001912 |
| graph output 1 | 0.001947 |
| graph output 2 | 0.002191 |
| graph output 4 | 0.002186 |

그러나 detailed inspector에서 graph output 0/1/2/4 모두 DLA compute partition은
한 개였다. 추가된 것은 intermediate/final output의 FP32 reformat뿐이며 내부
Conv/ReLU partition은 분리되지 않았다.

따라서 graph-output 수와 residual의 Spearman 0.8은 fusion/requantization의
인과 증거로 사용할 수 없다.

### Repeated block growth

Amplitude 1의 GPU RMS 정규화 residual:

| Blocks | Normalized residual |
|---:|---:|
| 1 | 0.000234 |
| 2 | 0.000606 |
| 4 | 0.001811 |
| 8 | 0.006018 |

- Spearman: 1.0
- 1→8 block 증가율: 약 25.8배
- endpoint occupancy 최대: 0.000088 미만
- fixed-calibration build direction cosine 최소: 0.9996

사전 정의 growth law의 원점 통과 적합도:

| Law | R² |
|---|---:|
| Constant | 0.000 |
| Linear | 0.909 |
| Square-root | 0.601 |

탐색적 power law:

- exponent: 1.564
- R²: 0.994

사전 정의 세 모델 중 linear가 가장 잘 맞지만, 4개 점에서 power law가 더 잘
맞는다는 결과는 탐색적으로만 보고한다.

### Reduction length

Reduction length 72/144/288/576/1152에서 GPU RMS 정규화 residual은
0.000327–0.000351 범위였다.

- Spearman: 0.6
- 효과 범위가 매우 작아 reduction length가 주요 residual 증폭 원인이라는
  증거는 약함

### Grouped/dataflow

Groups 1/2/4/8의 GPU RMS 정규화 residual:

- 0.000240 / 0.000235 / 0.000235 / 0.000233
- Spearman: -1.0

Group 수 증가가 residual을 증폭하지 않았고 오히려 매우 작게 감소했다.
Groups 16은 GPU strict-INT8 미지원으로 paired 비교 불가다.

### Build와 calibration 안정성

22개 paired 모델 전체:

- 같은 calibration 내 build direction cosine 최소: 0.9996
- calibration 간 mean-direction cosine 중앙값: 0.0198
- calibration 간 mean-direction cosine 범위: [-0.0289, 0.0511]
- calibration 간 top-8 subspace overlap 중앙값: 0.00186
- calibration 간 top-8 overlap 범위: [0.00093, 0.00514]
- amplitude 1 endpoint occupancy 최대: 0.000088

해석:

- residual은 비포화이며 같은 calibration cache에서는 거의 완전히 재현된다.
- calibration subset이 바뀌면 residual의 평균 방향과 top-k subspace가 사실상
  직교한다.
- TM-W에서 victim calibration이 바뀌는 조건에는 공격 carrier로 사용할 수 없다.

### B-micro gate

- TM-W 후보: 0
- **TM-W B-micro: NO-GO**
- TM-Q 탐색 shortlist: `repeated_4`, `repeated_8`

TM-Q shortlist 이유:

- fixed calibration에서 build 방향 안정성 통과
- amplitude 1 비포화
- repeated-block 수와 residual 크기의 인과적·단조 증가
- normalized effect가 각각 약 0.0018, 0.0060

TM-Q에서도 실제 full-model 경계에서 같은 signature가 재현되고 입력으로
제어될 때만 surrogate/trigger 단계로 이동한다.

### 산출물

- `results/v13/microbench_hardware_strict_int8/run_index.json`
- `results/v13/microbench_hardware_strict_int8/*.engine`
- `results/v13/microbench_hardware_strict_int8/*.inspector.json`
- `results/v13/microbench_hardware_strict_int8/*.npz`
- `results/v13/microbench_hardware_strict_int8_analysis.json`
- `results/v13/microbench_hardware_strict_int8_review.json`

## 2026-07-29 — B4 전체 모델 strict INT8 residual subspace

### 목적

Strict INT8 microbenchmark에서 관찰한 비포화 residual을 실제 ImageNet
백본의 깊은 경계에서 재검증하고, 3 builds × 2 shadow calibrations에 공통인
top-8 residual subspace를 탐색한다.

### 구현 및 실행

- 신규 capture:
  `chain_survival/scripts/capture_v13_boundaries.py`
- 신규 분석:
  `chain_survival/scripts/analyze_v13_boundaries.py`
- 경계:
  - ResNet-50 `layer1.2 Add`
  - ResNet-50 `layer4.2 Add`
- Calibration: `calib_shadow_1`, `calib_shadow_2`, 각 200장
- Discovery: `mechanism_discovery`의 앞 64장
- Build: calibration당 3회
- Backend: strict INT8 GPU 및 strict INT8 DLA
- 저장 feature:
  - spatial 4×4 pooled activation
  - channel mean/max
  - 정확한 quantized-value histogram
- 전체 원본 activation은 저장하지 않았다.

```bash
python chain_survival/scripts/capture_v13_boundaries.py \
  --builds 3 --calibrations 2 --n-calib 200 --n-discovery 64 \
  --allow-gpu-fallback
python chain_survival/scripts/analyze_v13_boundaries.py
```

### 실행 완결성 및 inspector

- Engine 조건: 24
- 성공: 24
- 실패: 0
- Detailed inspector: 24/24
- Strict INT8 compute 검증: 24/24
- GPU compute layer 범위: 11–67
- DLA compute partition: 모든 DLA engine에서 1개
- DLA의 GPU fallback: binding 직전 FP32 output reformat
- **Inspector gate: GO**

### `layer1.2`

- GPU RMS 정규화 residual: 0.016903
- fixed-calibration build mean-direction cosine 최소: 0.2737
- cross-calibration mean-direction cosine 중앙값: 0.2536
- cross-calibration top-8 overlap 중앙값: 0.0757
- consensus top-8 overlap:
  - 중앙값: 0.5768
  - 최소: 0.0290
- consensus residual energy:
  - 중앙값: 0.3174
  - 최소: 0.0377
- empirical endpoint occupancy 최대: 0.0000717
- **B4 consensus-subspace gate: NO-GO**

### `layer4.2`

- GPU RMS 정규화 residual: 0.151141
- fixed-calibration build mean-direction cosine 최소: -0.4489
- cross-calibration mean-direction cosine 중앙값: -0.0116
- cross-calibration top-8 overlap:
  - 중앙값: 0.5617
  - 범위: [0.4878, 0.6453]
- consensus top-8 overlap:
  - 중앙값: 0.7575
  - 최소: 0.6898
- consensus residual energy:
  - 중앙값: 0.6072
  - 최소: 0.5432
- empirical endpoint occupancy 최대: 0.0000556
- mean-direction gate: NO-GO
- **B4 consensus-subspace gate: GO**

### B4 해석

두 ResNet 경계 모두 signed mean residual 방향은 build/calibration에 안정적이지
않다. 그러나 `layer4.2`에서는 고정된 consensus top-8 subspace가 모든 6개
build/calibration 조건에서 0.5 이상의 overlap과 residual energy를 유지했다.
따라서 B5에서는 단일 signed direction이 아니라 이 고정 subspace에 대한
trigger-path interaction을 측정한다.

이 gate threshold는 discovery 분석용 operational threshold다. blind split
성능으로 해석하지 않는다.

### 산출물

- `results/v13/boundary_strict_int8/run_index.json`
- `results/v13/boundary_strict_int8/*.npz`
- `results/v13/boundary_strict_int8/engines/*.engine`
- `results/v13/boundary_strict_int8/engines/*.inspector.json`
- `results/v13/boundary_strict_int8_analysis.json`
- `results/v13/layer1.2_consensus_subspace.npz`
- `results/v13/layer4.2_consensus_subspace.npz`

## 2026-07-29 — B5 ResNet-50 입력 제어 가능성 screen

### 목적

Gradient surrogate나 trigger를 학습하기 전에 실제 strict INT8 GPU/DLA
엔진에서 저비용 perturbation이 `layer4.2` consensus subspace의
trigger-path interaction을 안정적으로 움직이는지 검증한다.

측정 interaction은 다음과 같다.

$$
\Gamma(t)
=
\left(z_{d,t}-z_{d,c}\right)
-
\left(z_{g,t}-z_{g,c}\right).
$$

### 구현 및 조건

- 신규 스크립트:
  `chain_survival/scripts/probe_v13_controllability.py`
- Discovery 이미지: 32장
- Build: 3
- Shadow calibration: 2
- Backend: GPU/DLA
- Boundary: ResNet-50 `layer4.2`
- Projection: B4에서 고정한 consensus top-8
- Perturbation 12종:
  - benign noise 0.02
  - benign brightness +0.05
  - 16×16 gray high/low
  - 16×16 red/green/blue
  - 16×16 checker/stripes/random
  - 32×32 checker
  - 32×32 random
- 실제 engine inference 수:
  12 perturbations × 6 conditions × 2 backends × 32 images = 4,608

```bash
python chain_survival/scripts/probe_v13_controllability.py \
  --builds 0 1 2
```

### Screen gate

후보마다 다음을 동시에 요구했다.

- condition mean-direction cosine 최소 ≥ 0.8
- 이미지별 고정 방향 양의 score 비율 최악 조건 ≥ 0.7
- projected interaction norm ≥ 최대 benign control의 3배
- 실제 endpoint occupancy ≤ 0.001

TM-W는 두 calibration 전체를 함께 평가하고, TM-Q는 calibration을 고정한
상태에서 세 build를 별도로 평가했다.

### TM-W 결과

Patch 후보 10종의 범위:

- condition direction cosine 최소: -0.7085–-0.4637
- condition direction cosine 중앙값: 0.0114–0.2041
- worst-condition positive image fraction: 0.375–0.500
- interaction norm / max benign control: 0.608–0.630
- endpoint occupancy 최대: 0.0000614

- **TM-W 통과 후보: 0**

### TM-Q 결과

Calibration 0, 세 build:

- direction cosine 최소: 0.1781–0.2434
- worst-condition positive image fraction: 0.4688–0.6562
- interaction norm / max benign control: 0.7146–0.7347
- 통과 후보: 0

Calibration 1, 세 build:

- direction cosine 최소: -0.7085–-0.4606
- worst-condition positive image fraction: 0.4062–0.5938
- interaction norm / max benign control: 0.5106–0.5348
- 통과 후보: 0

### B-control 판정

- 포화 안전성: GO
- 방향 안정성: NO-GO
- benign control 대비 효과 크기: NO-GO
- 이미지별 일관성: NO-GO
- **TM-W B-control: NO-GO**
- **TM-Q B-control: NO-GO**

따라서 v13의 단계별 중단 규칙에 따라 ResNet-50 `layer4.2` 후보에서는
B6 surrogate와 B7 gradient trigger optimization을 수행하지 않는다.
`boundary_blind`, `final_logit_blind` 및 blind calibration split은 열지 않았다.

### 산출물

- `results/v13/controllability_screen.json`

## 2026-07-29 — B4 추가 백본 탐색

### 목적

ResNet-50 B-control NO-GO 후 v13 B4에 명시된 다른 백본 경계를 탐색한다.

- VGG-16의 순차적 깊은 경계:
  최종 convolutional ReLU인 `features29`
- VGG-19의 순차적 깊은 경계:
  최종 convolutional ReLU인 `features35`
- GoogLeNet의 branch-merge 경계:
  `inception5b`

### 구현

- 신규 스크립트:
  `chain_survival/scripts/capture_v13_additional_backbones.py`
- 각 백본을 선택한 경계까지만 export했다.
- 각 경계에서 3 builds × 2 shadow calibrations × GPU/DLA를 실행했다.
- Calibration은 각 200장, discovery는 64장이다.
- strict builder는 activation compute layer만 INT8로 강제하고, Int32/Int64
  shape/control/constant layer는 원래 자료형을 유지하도록 일반화했다.
- `--overwrite`는 해당 백본의 calibration cache를 무효화하고 재보정한다.

### 실행 중 builder 교정

첫 GoogLeNet 시도에서 strict builder가 ONNX의 Int64 Constant까지 INT8로
강제하여 다음 오류로 build가 종료됐다.

```text
cannot use precision Int8 with weights of type Int64
```

조치:

- activation arithmetic layer만 strict INT8 대상으로 제한
- shape/control/constant layer는 원래 자료형 유지
- GoogLeNet의 channel별 입력 변환을 동등한 broadcast scale/bias로 단순화
- 변경된 ONNX에서 이전 calibration cache 재사용 금지
- 실패 시도의 root-level run index, cache 및 임시 ONNX 삭제

이 실패 시도의 engine/activation 수치는 최종 분석에 사용하지 않았다.

### ONNX parity

VGG-16:

- 출력 shape: 1×512×14×14
- PyTorch–ONNX Runtime max absolute difference: 4.17e-6
- mean absolute difference: 4.01e-8
- allclose `(rtol=1e-4, atol=1e-5)`: True

GoogLeNet:

- 출력 shape: 1×1024×7×7
- max absolute difference: 1.88e-5
- mean absolute difference: 9.62e-7
- normalized RMS error: 3.36e-6
- cosine similarity: 0.999999999994
- allclose `(rtol=1e-4, atol=1e-5)`: False
- allclose `(rtol=1e-4, atol=2e-5)`: True

엄격한 1e-5 absolute tolerance는 일부 거의 0인 원소에서 통과하지 않았으므로
이를 완전 일치로 보고하지 않는다. 다만 상대 RMS와 cosine 기준으로 export
오차는 이후 관측한 GPU–DLA residual보다 여러 자릿수 작다.

### VGG-16 실행 및 결과

```bash
python chain_survival/scripts/capture_v13_additional_backbones.py \
  --models vgg16 \
  --output-dir results/v13/additional_backbone_strict_int8/vgg16 \
  --allow-gpu-fallback --overwrite
python chain_survival/scripts/analyze_v13_boundaries.py \
  --run-index results/v13/additional_backbone_strict_int8/vgg16/run_index.json \
  --output results/v13/additional_backbone_vgg16_analysis.json
```

- Engine: 12/12 성공
- Detailed/strict inspector: 12/12
- DLA compute partition: 1
- normalized residual: 0.178941
- fixed-calibration build mean-direction cosine 최소: -0.4755
- cross-calibration mean-direction cosine 중앙값: -0.0859
- consensus top-8 overlap:
  - 중앙값: 0.5676
  - 최소: 0.5285
- consensus residual energy:
  - 중앙값: 0.3906
  - 최소: 0.3616
- endpoint occupancy 최대: 0.0000212
- **VGG-16 B4 consensus-subspace gate: NO-GO**

### GoogLeNet 실행 및 결과

```bash
python chain_survival/scripts/capture_v13_additional_backbones.py \
  --models googlenet \
  --output-dir results/v13/additional_backbone_strict_int8/googlenet \
  --allow-gpu-fallback --overwrite
python chain_survival/scripts/analyze_v13_boundaries.py \
  --run-index results/v13/additional_backbone_strict_int8/googlenet/run_index.json \
  --output results/v13/additional_backbone_googlenet_analysis.json
```

- Engine: 12/12 성공
- Detailed/strict inspector: 12/12
- DLA compute partition: 1
- 입력 정규화 상수 두 개와 최종 FP32 output reformat만 GPU fallback
- normalized residual: 0.310784
- fixed-calibration build mean-direction cosine 최소: -0.5158
- cross-calibration mean-direction cosine 중앙값: -0.1737
- consensus top-8 overlap:
  - 중앙값: 0.6884
  - 최소: 0.5629
- consensus residual energy:
  - 중앙값: 0.4157
  - 최소: 0.3742
- endpoint occupancy 최대: 0.0000439
- **GoogLeNet B4 consensus-subspace gate: NO-GO**

### VGG-19 실행 및 결과

```bash
python chain_survival/scripts/capture_v13_additional_backbones.py \
  --models vgg19 \
  --output-dir results/v13/additional_backbone_strict_int8/vgg19 \
  --allow-gpu-fallback --overwrite
python chain_survival/scripts/analyze_v13_boundaries.py \
  --run-index results/v13/additional_backbone_strict_int8/vgg19/run_index.json \
  --output results/v13/additional_backbone_vgg19_analysis.json
```

- ONNX max absolute difference: 2.38e-6
- ONNX normalized RMS error: 5.44e-7
- ONNX cosine similarity: 0.9999999999999
- Engine: 12/12 성공
- Detailed/strict inspector: 12/12
- DLA compute partition: 1
- normalized residual: 0.208741
- fixed-calibration build mean-direction cosine 최소: -0.2492
- cross-calibration mean-direction cosine 중앙값: -0.0228
- consensus top-8 overlap:
  - 중앙값: 0.6940
  - 최소: 0.5114
- consensus residual energy:
  - 중앙값: 0.4180
  - 최소: 0.3415
- endpoint occupancy 최대: 0.00000919
- **VGG-19 B4 consensus-subspace gate: NO-GO**

### 추가 백본 해석

VGG-16, VGG-19 및 GoogLeNet은 ResNet-50 `layer4.2`보다 normalized
residual이 각각 더 크거나 매우 컸지만, 고정 consensus top-8에 포착되는
residual energy가 모든 build/calibration에서 0.5를 넘지 못했다. 큰 GPU–DLA
차이 자체는 안정적이고 제어 가능한 carrier가 아님을 다시 확인했다.

두 추가 경계는 B4에서 종료하며 B5, surrogate 및 trigger optimization을
수행하지 않는다.

### 산출물

- `results/v13/additional_backbone_strict_int8/vgg16/`
- `results/v13/additional_backbone_strict_int8/vgg19/`
- `results/v13/additional_backbone_strict_int8/googlenet/`
- `results/v13/additional_backbone_strict_int8/googlenet/onnx_validation_repeat.json`
- `results/v13/additional_backbone_vgg16_analysis.json`
- `results/v13/additional_backbone_vgg19_analysis.json`
- `results/v13/additional_backbone_googlenet_analysis.json`
- `results/v13/vgg16.features29_consensus_subspace.npz`
- `results/v13/vgg19.features35_consensus_subspace.npz`
- `results/v13/googlenet.inception5b_consensus_subspace.npz`

## 2026-07-29 — 현재 단계 종합 판정

| 단계 | 결과 | 판정 |
|---|---|---|
| Track A | 최고 heldout worst-group 0.392 | NO-GO |
| B0 | 13,500 경로 중복 없음, blind 분리 | GO |
| B-micro TM-W | calibration 간 방향/subspace 불안정 | NO-GO |
| B-micro TM-Q | repeated block 4/8 탐색 shortlist | 제한적 GO |
| B4 ResNet `layer1.2` | consensus energy/overlap 불안정 | NO-GO |
| B4 ResNet `layer4.2` | consensus overlap 최소 0.690, energy 최소 0.543 | GO |
| B5 ResNet `layer4.2` TM-W | 실제 perturbation 후보 0 | NO-GO |
| B5 ResNet `layer4.2` TM-Q | calibration별 후보 0 | NO-GO |
| B4 VGG-16 | consensus energy 최소 0.362 | NO-GO |
| B4 VGG-19 | consensus energy 최소 0.341 | NO-GO |
| B4 GoogLeNet | consensus energy 최소 0.374 | NO-GO |

### 현재 결론

동일 strict INT8에서 GPU–DLA residual은 실제로 존재하고, 깊은 경계와 반복
연산에서 normalized magnitude가 커진다. 하지만 다음 세 조건을 동시에 만족한
후보는 없었다.

1. build/calibration 전반의 고정 residual signature
2. 비포화
3. 실제 입력 perturbation에 의한 안정적 interaction 제어

특히 가장 유망했던 ResNet-50 `layer4.2` consensus subspace도 B5에서 patch
효과가 benign control보다 작고 build/calibration 방향이 불안정했다.

### 중단 범위

v13의 단계별 No-Go 규칙에 따라 현재 후보에 대해 다음을 실행하지 않았다.

- B6 differentiable surrogate
- B7 gradient trigger optimization
- guard/readout 학습
- tail finetuning
- blind build/calibration 평가

이는 실행 능력 부족이 아니라 B-control gate에 따른 의도적 중단이다. 후보
선택에 사용하지 않도록 다음 split은 계속 봉인돼 있다.

- `calib_blind_1/2/3`
- `threshold_validation`
- `boundary_blind`
- `final_logit_blind`
- `robustness`

### 연구 결과 표현

현재 결과는 완전한 DPCB 성공이 아니다. v13의 negative-result framing에 따라
다음으로 제한한다.

> 큰 이종 INT8 path difference와 공격 가능한 input-conditioned interaction은
> 다르며, calibration/build drift와 input uncontrollability가 공격 성립을
> 제한한다.

다음 신규 공격 실험은 기존 후보의 surrogate를 억지로 학습하는 것이 아니라,
별도로 사전 등록한 Track C protocol 또는 새로운 mechanism 후보에서 시작해야
한다.

## 2026-07-29 — 후속 trigger 탐색의 챌린지와 해결 가설

이 절은 앞선 결과를 본 뒤 정의한 후속 탐색 protocol이다. 기존 B5 결과를
설명하도록 metric을 사후 변경해 성공으로 재분류하지 않는다. 새로운 screen은
subspace 생성에 사용하지 않은 이미지에서 별도 실험으로 수행한다.

### Challenge 1 — signed residual direction의 build/calibration drift

관측 근거:

- ResNet-50 `layer4.2` mean-direction cosine은 fixed calibration에서도 최소
  -0.4489였다.
- 반면 consensus top-8 overlap은 모든 조건에서 최소 0.6898이고, 이 subspace가
  포착한 residual energy는 최소 0.5432였다.

논리적 해석:

고정된 한 방향 $u^\top\Gamma$를 목표로 하면 subspace 내부의 basis rotation이나
sign flip을 실패로 처리한다. 현재 자료가 지지하는 안정 단위는 signed vector가
아니라 저차원 subspace다.

해결 가설:

$$
E_{\Gamma}(x,t)
=
\left\|
U_{\mathrm{cons}}^\top\Gamma(x,t)
\right\|_2^2
$$

를 사용한다. 이 값은 고정 subspace 내부의 직교 basis rotation과 sign flip에
불변이다.

반증 기준:

- 동일 pixel-RMS control보다 interaction energy가 증가하지 않음
- 최소 한 build/calibration에서 효과가 사라짐
- 소수 이미지에만 의존

### Challenge 2 — 국소 hard patch의 효과가 benign perturbation보다 약함

관측 근거:

- 기존 16/32 patch의 interaction norm은 brightness control의
  0.608–0.630배였다.
- microbenchmark에서 유일하게 명확했던 인과 signature는 반복 blo ck 수에 따른
  residual 증가였다.

논리적 해석:

깊은 repeated block에 도달하기 전에 작은 국소 patch의 영향이 공간적으로
희석될 수 있다. 반대로 작은 global brightness가 더 큰 interaction을 만든 것은
넓은 spatial support를 가진 저주파 perturbation이 더 적합할 가능성을 시사한다.

해결 가설:

- 전역 brightness/color shift
- low-frequency Fourier field
- smooth low-frequency random field
- contrast/illumination field
- 더 큰 additive patch

를 동일 **pixel-space RMS**에서 비교한다. 기존처럼 normalized tensor에 임의
상수를 직접 넣지 않고, pixel `[0,1]` 공간에서 perturb한 뒤 모델 normalization을
다시 적용한다.

반증 기준:

- matched Gaussian control 대비 worst-condition energy ratio < 1.25
- paired image의 65% 미만에서 control보다 큼

### Challenge 3 — perturbation 크기와 support의 confounding

국소 patch와 전역 perturbation을 동일 peak amplitude만으로 비교하면 전역
pattern의 총 에너지가 더 크다. 반대로 동일 patch amplitude만 맞추면 국소
pattern이 불리하다.

해결:

- 모든 family를 global pixel RMS `2/255`, `4/255`, `8/255`로 정규화
- 실제 적용 후 RMS와 $L_\infty$를 함께 기록
- 동일 RMS Gaussian noise를 family별 reference control로 사용
- clipping 후 실제 RMS가 목표에서 20% 이상 벗어나면 gate 실패

### Challenge 4 — discovery data 재사용과 선택 편향

현재 consensus subspace는 `mechanism_discovery`의 첫 64장으로 만들었다.
같은 이미지에서 후속 family를 고르면 효과가 과대 추정될 수 있다.

해결:

1. subspace 고정: discovery index 0–63
2. family screen: discovery index 64 이후
3. family 선택 후 재검증: `threshold_validation`
4. `boundary_blind`와 `final_logit_blind`는 계속 봉인

### Challenge 5 — residual energy 증가와 공격 가능성의 차이

큰 $E_\Gamma$도 GPU-triggered 동작이 크게 바뀌거나 saturation에 의존하면
DPCB carrier가 아니다.

해결:

- GPU trigger effect와 DLA trigger effect를 별도 기록
- interaction/GPU-effect ratio 기록
- quantized endpoint occupancy ≤ 0.001 유지
- energy screen 통과 후에만 full-logit four-group 분리를 측정
- four-group separator가 없으면 tail/surrogate 최적화 금지

### Challenge 6 — source-condition dependence

기존 32장 screen의 worst-condition positive fraction이 낮았다는 것은 보편적
trigger가 아니라 특정 image/content subset에서만 효과가 날 가능성도 뜻한다.

해결:

- per-image energy와 clean activation norm을 보존
- 통과 후보는 class/texture/clean-margin 기준으로 사후 효과 이질성을 분석
- subset은 discovery에서 정의하고 validation에서 고정
- validation에서 재현되지 않는 source subset은 폐기

### 신규 energy-controllability gate

새 family가 다음을 모두 만족할 때만 gradient optimization 후보로 인정한다.

1. matched Gaussian 대비 mean energy ratio의 worst condition ≥ 1.25
2. paired image에서 candidate energy가 control보다 큰 비율의 worst condition
   ≥ 0.65
3. 모든 조건에서 실제 pixel RMS가 reference와 ±20% 이내
4. endpoint occupancy 최대 ≤ 0.001
5. 독립 `threshold_validation`에서 같은 기준 재현

이 gate는 기존 signed-direction B5 실패를 뒤집는 기준이 아니다. 안정 단위가
subspace라는 B4 관측에서 도출한 별도의 rotation-invariant 가설 검정이다.

## 2026-07-29 — 연구 범위 재검토: compiler-only 대 heterogeneous path

### 질문

GPU–DLA 이기종 실행까지 요구하지 않고 문제를 “양자화되어 컴파일된 모델에서만
활성화되는 backdoor”로 제한해도 충분한 학술적 가치가 있는가?

### 선행연구와 중복 위험

다음 선행연구가 이미 양자화 또는 컴파일을 activation condition으로 사용하는
backdoor를 직접 다룬다.

1. **Quantization Backdoors to Deep Learning Commercial Frameworks**
   (2021)는 정상적으로 보이는 FP32 모델이 표준 TFLite/PyTorch Mobile INT8
   변환 후 trigger backdoor를 활성화하는 공격을 보였다.
   [논문](https://arxiv.org/abs/2108.09187)
2. **Qu-ANTI-zation** (NeurIPS 2021)은 quantization artifact를 이용한
   indiscriminate, targeted 및 backdoor outcome과 여러 quantization scheme
   간 전이를 연구했다.
   [논문](https://arxiv.org/abs/2110.13541)
3. **ImpNet**은 malicious compiler가 compiled neural-network binary에
   black-box로 탐지하기 어려운 backdoor를 삽입하는 공격면을 다뤘다.
   [논문](https://openreview.net/forum?id=v01xUvzem4)
4. **Your Compiler is Backdooring Your Model** (IEEE S&P 2026)은 official,
   unmodified compiler가 pre-compilation에는 dormant한 trigger를
   post-compilation에 활성화할 수 있음을 세 commercial compiler와 두 hardware
   platform에서 보였다.
   [논문](https://arxiv.org/abs/2509.11173)
5. **QuEST** (IEEE TIFS 2026)는 quantization-conditioned backdoor의 효율과
   stealth를 직접 최적화한다.
   [논문](https://doi.org/10.1109/TIFS.2026.3671079)

따라서 다음 정의만으로는 novelty가 부족하다.

> FP32에서 dormant하고, 양자화/컴파일 후 활성화되는 trigger backdoor.

이는 새로운 문제정의라기보다 기존 PQ/QCB/compiler inconsistency attack의
재현 또는 TensorRT 적용으로 평가될 위험이 크다.

### 현재 실험이 제공하는 차별점

현재 프로젝트의 고유한 증거는 “컴파일하면 backdoor가 생길 수 있다”가 아니다.

- 동일한 strict INT8 graph와 동일 입력에서도 GPU와 DLA가 다른 수치 함수를
  구현한다.
- 그 차이는 깊이에 따라 매우 커질 수 있다.
- 그러나 큰 path difference도 build/calibration 안정성, 비포화 및 입력
  제어 가능성을 동시에 만족하지 않으면 공격 carrier가 아니다.
- 실제 결과에서 normalized residual 0.151–0.311도 곧바로 trigger-path
  interaction으로 전환되지 않았다.
- 3 builds, 2 calibrations 및 실제 engine inspector를 함께 사용하면
  proxy-only 성공과 deployment success를 구분할 수 있다.

이는 기존 공격의 단순 성공 보고와 다른 **exploitability boundary** 문제다.

### 범위 결정

**DLA를 주 문제정의에서 제거하지 않는다.**

논문의 구조는 다음의 계층형 설계로 유지한다.

| 계층 | 역할 | 주장 범위 |
|---|---|---|
| FP32 → quantized compiled | 기존 QCB/compiled-backdoor 비교 baseline | 기존 공격 재현 및 통제 |
| 동일 compiler, 여러 build/calibration | compiler uncertainty | tactic/calibration drift 측정 |
| 동일 strict INT8, GPU ↔ DLA | 핵심 deployment-path test | backend-local residual과 interaction |
| 여러 백본/경계 | 일반화 | 큰 residual과 exploitability 분리 |

즉 compiler-only는 삭제하지 않고 **baseline/ablation**으로 두며, 핵심 연구
질문은 다음으로 유지한다.

> 검증된 quantized model이 실제 deployment executor에서 다른 수치 함수를
> 구현할 때, 어떤 조건에서 그 차이가 안정적이고 입력 제어 가능한 보안
> primitive가 되는가?

### DLA를 제외해도 논문이 되려면 필요한 변경

향후 장치 제약으로 DLA를 완전히 제외해야 한다면 공격 논문이 아니라 다음 중
하나로 pivot해야 한다.

1. 여러 compiler/version/optimization level에서 compilation inconsistency의
   재현성·원인을 체계적으로 측정하는 measurement paper
2. build/calibration uncertainty를 포함하는 compiled-model equivalence audit
3. 기존 QCB가 unseen compiler build에서 얼마나 붕괴하는지 보여주는
   robustness/negative-result paper
4. compiler output을 직접 검사하는 path-aware defense

단일 TensorRT quantized build에서 새로운 trigger 하나를 만드는 것만으로는
현재 선행연구 대비 충분하지 않다.

### 현재 논문 framing

현재 결과가 가장 잘 지지하는 framing은 다음과 같다.

> **Deployment-Path Exploitability Boundaries:** quantization/compilation이
> 만드는 수치 차이는 흔하고 클 수 있지만, 공격 가능한 backdoor가 되려면
> saturation, build drift, calibration drift, input controllability 및
> four-group separability gate를 모두 통과해야 한다.

완전한 attack이 끝내 실패하더라도 다음 독립 기여가 남는다.

- strict INT8 GPU/DLA causal microbenchmark
- residual magnitude와 exploitability의 구분
- signed direction보다 안정적인 consensus subspace 분석
- 실제 hardware checkpoint gate
- 실패 원인 taxonomy와 path-aware audit protocol

## 2026-07-29 — Rotation-invariant energy screen 결과

### 실행 범위

- 고정 subspace: ResNet-50 `layer4.2` consensus top-8
- 데이터: `mechanism_discovery` index 64–127, 64장
- subspace 생성 데이터 index 0–63과 분리
- 조건: 3 builds × 2 shadow calibrations × GPU/DLA
- family/control: 53종
- pixel RMS: 2/255, 4/255, 8/255
- 비교 control: 동일 RMS Gaussian noise
- metric: $\|U_{\mathrm{cons}}^\top\Gamma\|_2^2$

```bash
python chain_survival/scripts/probe_v13_energy_controllability.py
```

### 전체 TM-W 결과

- 통과 후보: 0
- 상위 mean-energy 후보와 실패 원인:

| 후보 | worst energy ratio | worst paired fraction | 판정 |
|---|---:|---:|---|
| contrast +, RMS 8/255 | 1.615 | 0.391 | NO-GO |
| contrast +, RMS 4/255 | 1.420 | 0.516 | NO-GO |
| brightness +, RMS 4/255 | 1.373 | 0.484 | NO-GO |
| brightness +, RMS 8/255 | 1.229 | 0.500 | NO-GO |
| brightness -, RMS 4/255 | 1.129 | 0.422 | NO-GO |

일부 후보는 평균 interaction energy를 키웠지만 이미지별 효과가 일관되지 않아
paired-fraction gate 0.65를 통과하지 못했다. 이는 큰 평균값이 소수 이미지에
의존할 수 있다는 Challenge 6을 실제로 확인한다.

### TM-Q calibration-fixed 결과

Calibration 0에서 `brightness_minus_rms4` 한 개가 screen gate를 통과했다.

- worst build energy ratio: 1.477
- median build energy ratio: 1.514
- worst paired image fraction: 0.656
- condition energy CV: 0.125
- interaction/GPU-effect ratio 최소: 1.141
- endpoint occupancy 최대: 0.0000377
- actual RMS / Gaussian control: 0.987

Calibration 1에서는 같은 후보가 실패했다.

- worst build energy ratio: 1.129
- worst paired image fraction: 0.422
- interaction/GPU-effect ratio 최소: 0.897

### 판정

- TM-W rotation-invariant energy trigger: **NO-GO**
- TM-Q calibration 0 candidate: **PROVISIONAL GO**
- calibration-independent trigger: **아님**

`brightness_minus_rms4`는 공격 trigger로 확정하지 않는다. 선택에 사용하지 않은
`threshold_validation`에서 calibration 0의 세 build를 고정한 독립 재검증을
통과해야 한다. calibration 1 실패는 TM-W 일반화 주장에 대한 명시적 제한이다.

### 산출물

- `results/v13/residual_energy_controllability_screen.json`
