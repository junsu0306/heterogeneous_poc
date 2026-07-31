# v14 Factorized DPCB 실험 진행 일지

이 문서는 `V14_FACTORIZED_DPCB_EXPERIMENT_PROTOCOL.md`에 따른 실제 감사, 파일 정리,
구현, 실행 명령, 환경, 산출물, 실패 원인 및 Go/No-Go 판단을 시간순으로 누적한다.
모든 시간은 Asia/Seoul 기준이다. Proxy/CPU 결과와 실제 GPU/DLA 결과를 구분하며,
blind split을 연 경우 그 시점과 목적을 반드시 기록한다.

## 기록 및 재현성 규칙

- 실행한 명령과 주요 출력, 입력/출력 artifact를 기록한다.
- 삭제 전 대상, 크기, 삭제 이유와 v14 재사용 여부를 먼저 기록한다.
- Git 추적 파일과 비추적/ignore 대용량 artifact를 구분한다.
- 실제 engine inspector를 통과하지 않은 결과를 strict-INT8 결과로 간주하지 않는다.
- F0/F1 Gate를 통과하기 전에는 trigger, tail ASR, gate channel 실험을 시작하지 않는다.
- `calib_blind_1`은 path probe와 threshold를 동결한 뒤에만 연다.
- `calib_blind_2/3`, `threshold_validation`, `boundary_blind`,
  `final_logit_blind`, `robustness`는 해당 protocol 단계 전까지 봉인한다.

## 2026-07-29 13:59 KST — v14 전환 감사와 환경 점검

### 사용자 요청과 적용 원칙

- 기존 academic v13, v13 실험 일지, v14 protocol/config를 먼저 읽고 실패 원인과
  재사용 artifact를 구분한다.
- 필요 없는 구 코드와 결과는 삭제하고 v14만을 위한 구현/결과 구조로 전환한다.
- 모든 과정은 이 문서에 누적 기록한다.

`V14_FACTORIZED_DPCB_EXPERIMENT_PROTOCOL.md` §0과 §4는
`academic_research_plan_v13_trackB.md`, `results/v13/splits_v13.json`,
strict-INT8 builder, paired GPU/DLA capture infrastructure,
`boundary_strict_int8`, `layer4.2_consensus_subspace.npz`를 명시적으로 재사용한다.
따라서 "이전 파일 삭제"는 v14가 금지한 실험 반복과 무관한 대용량 산출물을
정리하는 것으로 해석하며, v14의 입력/근거인 v13 문서와 필수 artifact는 보존한다.

### 읽은 기준 파일

- `academic_research_plan_v13_trackB.md` 전체
- `chain_survival/EXPERIMENT_LOG_V13.md` 전체
- `V14_FACTORIZED_DPCB_EXPERIMENT_PROTOCOL.md` 전체
- `v14_factorized_dpcb_config.yaml` 전체

### v13 실패 원인 요약

- Track A 24채널 ensemble 최고 held-out worst-group은 0.392로 NO-GO였다.
- strict INT8 GPU-DLA residual과 반복 block에 따른 증가는 실제 hardware에서
  확인됐지만 calibration 간 signed direction은 불안정했다.
- ResNet-50 `layer4.2` top-8 consensus subspace는 안정적이었으나 기존 patch와
  low-frequency family가 residual을 이미지별로 일관되게 제어하지 못했다.
- TM-W rotation-invariant energy screen 통과 후보는 0개였다.
- 따라서 v14는 residual을 trigger가 직접 제어하는 접근을 중단하고, clean
  activation의 path fingerprint와 path-independent trigger fingerprint를
  별도로 학습한 뒤 tail에서 conjunction하는 F-DPCB로 전환한다.

### 초기 저장소 상태

명령:

```bash
git status --short --branch
rg --files -g '!**/.git/**'
du -ah chain_survival common | sort -hr | head -120
```

결과:

- branch: `main...origin/main`
- 수정: `chain_survival/EXPERIMENT_LOG_V13.md`
- 신규: `V14_FACTORIZED_DPCB_EXPERIMENT_PROTOCOL.md`
- 신규: `v14_factorized_dpcb_config.yaml`
- `chain_survival/results/fourgroups_train.npz`: 약 18 GB
- `chain_survival/results/v13`: 약 3.6 GB
- `chain_survival/engines`: 약 410 MB
- `common/external/Qu-ANTI-zation`: 약 14 GB
- `common/datasets/cifar10`: 약 314 MB

### 환경 점검

명령:

```bash
date '+%Y-%m-%d %H:%M:%S %Z'
uname -a
python -c 'import torch, tensorrt, onnx, onnxruntime, numpy, yaml'
ls -l /dev/nvidia0 /dev/nvhost-gpu /dev/nvhost-ctrl-nvdla0 /dev/nvhost-ctrl-nvdla1
df -h .
```

결과:

- Kernel: Linux 5.15.185-tegra, aarch64
- PyTorch: 2.11.0
- TensorRT: 10.3.0
- ONNX: 1.22.0
- ONNX Runtime: 1.23.2
- NumPy: 2.2.6
- PyYAML: 6.0.3
- scikit-learn: 설치되지 않음
- `torch.cuda.is_available()`: `True`
- CUDA device 수: 1
- GPU 및 DLA 0/1 device node: 모두 존재
- 저장 공간: 234 GB 중 177 GB 사용 가능

### 초기 판정

- 실제 strict-INT8 GPU/DLA F0 실행: **GO**
- scikit-learn 의존 구현: **NO-GO**. v14 probe는 PyTorch/NumPy 기반으로 구현한다.
- Blind split: 아직 열지 않음.

### 다음 조치

1. v13 코드의 import/dependency와 v14 필수 artifact의 무결성을 검사한다.
2. 삭제 manifest를 확정하고 본 문서에 기록한다.
3. 삭제 후 v14 F0 capture/probe 스크립트를 작성하고 smoke/full run을 수행한다.

## 2026-07-29 14:00 KST — v14 재사용 무결성 및 삭제 manifest

### 필수 입력 무결성

명령:

```bash
sha256sum \
  chain_survival/onnx/resnet50.onnx \
  chain_survival/models/resnet50.pth \
  chain_survival/results/v13/splits_v13.json \
  chain_survival/results/v13/layer4.2_consensus_subspace.npz \
  chain_survival/results/v13/boundary_strict_int8/run_index.json \
  common/scripts/trt_runtime.py
```

결과:

| Artifact | SHA-256 |
|---|---|
| ResNet-50 ONNX | `e9737b1e4a14f333743f0cab11e29326432f5b9509f6c42587aac758665edf96` |
| ResNet-50 checkpoint | `3ce1c0adebfa0371435c97516dbb1a0c5ac22ad708b2e30d02b9741c2800a011` |
| v13 split | `6c2901a5c68710ed8bff3a7f609a46045f9f988cec06f173d7630f45501bbffb` |
| layer4.2 consensus | `d78d171b57d82d9226648be1187b2c0cf85848135bd46cf38f9656a1cabd5e19` |
| v13 boundary run index | `47f34cf5232548888860194897ceb42f8c1d9da5f19977573efb7a06e9059254` |
| strict-INT8 runtime | `6d0686b6cc0e0e3b1a88b8d4089bb032f3ec63fec94057b70a46bf380c88c996` |

추가 검증:

- ImageNet root 존재, 파일 수 50,000
- v13 boundary record 24개 및 연결 artifact 누락 0개
- `layer4.2_consensus_subspace.npz` directions shape: `(8, 32768)`

### 보존 대상

- 연구 근거/기록:
  - `academic_research_plan_v13_trackB.md`
  - `chain_survival/EXPERIMENT_LOG_V13.md`
  - v14 protocol/config 및 본 로그
- v14 입력:
  - `chain_survival/onnx/resnet50.onnx`
  - `chain_survival/models/resnet50.pth`
  - `chain_survival/results/v13/splits_v13.json`
  - `chain_survival/results/v13/boundary_strict_int8/`
  - `chain_survival/results/v13/boundary_strict_int8_analysis.json`
  - `chain_survival/results/v13/layer4.2_consensus_subspace.npz`
- v14 재사용 코드:
  - `common/scripts/trt_runtime.py`
  - `common/scripts/export_resnet50.py`
  - `chain_survival/scripts/models_cfg.py`
  - `chain_survival/scripts/run_paths.py`
  - `chain_survival/scripts/export_models.py`
  - `chain_survival/scripts/capture_v13_boundaries.py`
  - `chain_survival/scripts/analyze_v13_boundaries.py`
  - `chain_survival/scripts/prepare_v13_splits.py`
  - `chain_survival/scripts/capture_v13_manifest.py`

### 삭제 대상과 근거

대용량 비추적 artifact:

| 대상 | 크기 | 근거 |
|---|---:|---|
| `results/fourgroups_train.npz` | 18 GB | Track A/포화 carrier cache, v14 재사용 금지 |
| 나머지 root four-group/Option2 결과 | 약 35 MB | 이전 guard/direct-interaction 결과 |
| v13 microbenchmark hardware 결과 | 약 2.1 GB | 결론은 v13 로그에 보존, v14 F0 입력 아님 |
| v13 추가 백본 결과 | 약 694 MB | v14는 ResNet-50 F0 우선, 백본 확장 금지 |
| v13 smoke/실패 run | 약 190 MB+ | 최종 strict run으로 대체됨 |
| `engines/boundary_heads` | 약 410 MB | 포화 carrier/Option2용 old ONNX |
| `common/external` | 약 14 GB | 재다운로드 가능한 선행실험 clone/data |
| `common/datasets/cifar10` | 약 314 MB | v14 ImageNet 실험과 무관 |
| `common/models` | 약 154 MB | 중복 ResNet/YOLO artifact |

추적 코드 삭제:

- Track A, Option2, engineered-channel, saturation guard, direct trigger optimization
- v13 microbenchmark 및 추가 백본 전용 runner/analyzer
- CIFAR emulator 및 YOLO 전용 export/simplify 코드

복구 정책:

- Git 추적 코드는 `git` history에서 복구 가능하다.
- 비추적 engine/cache/activation은 직접 복구할 수 없지만 원본 데이터와 보존한
  build/export 코드로 재생성 가능하다.
- 외부 clone과 CIFAR/YOLO 자산은 재다운로드 가능한 항목이다.

### 삭제 전 판정

- v14 필수 입력 누락: 0
- blind split 개봉: 없음
- **정리 실행: GO**

## 2026-07-29 14:02 KST — 파일 정리 실행 및 검증

### 실행

- Git 추적 코드 31개는 patch 삭제로 제거했다.
- 비추적 artifact는 삭제 manifest에 열거한 정확한 경로만 `rm -r --`로 제거했다.
- workspace root, 사용자 홈, ImageNet root 및 v14 보존 경로에는 재귀 삭제를
  수행하지 않았다.

### 삭제 결과

- 작업공간 크기: 약 37 GB 이상에서 746 MB로 감소
- 파일시스템 사용량: 46 GB에서 10 GB로 감소
- 사용 가능 공간: 177 GB에서 212 GB로 증가
- 회수 공간: 약 35 GB

삭제된 주요 항목:

- 18 GB `fourgroups_train.npz`
- v13 microbenchmark/additional-backbone/smoke engine과 activation
- old boundary head, trigger checkpoint, log 및 direct-interaction 결과
- `common/external`, CIFAR-10, 중복/YOLO model artifact
- Track A/Option2/engineered-channel/direct-trigger 전용 코드

### 보존 검증

남은 binary 입력:

- `chain_survival/models/resnet50.pth`: 102,541,957 bytes
- `chain_survival/onnx/resnet50.onnx`: 102,146,373 bytes
- `results/v13/splits_v13.json`: 1,542,680 bytes
- `results/v13/layer4.2_consensus_subspace.npz`: 1,049,890 bytes
- v13 boundary GPU/DLA paired activation, engine, inspector, cache: 모두 보존

보존한 Python 9개에 `python -m py_compile`을 실행했고 모두 통과했다.

### 복구 가능성

- 삭제된 추적 코드는 Git에서 복구 가능하다.
- 삭제한 binary 결과와 외부 clone은 현재 작업공간에서 즉시 복구되지 않지만
  재생성/재다운로드 가능하다.
- 사용자 작성 v13 실험 로그의 수정 내용은 보존했다.

### 판정

- v14 전환 정리: **완료**
- 필수 입력 손상: 없음
- 다음 단계: F0 구현

## 2026-07-29 14:10 KST — F0 구현

### 신규 코드

- `capture_v14_path_features.py`
  - `layer4.0/4.1/4.2`를 각각 독립 ONNX/engine으로 추출
  - 2 shadow calibrations × 3 builds × GPU/DLA 지원
  - strict-INT8 detailed inspector gate
  - 이미지별 4×4 pooled, channel mean/std/RMS/max-abs, 32-bin normalized
    quantized occupancy, sign ratio, endpoint occupancy, label/id 저장
  - 동일 engine으로 다른 image split을 재캡처할 수 있도록 engine과 feature를 분리
- `train_v14_path_probe.py`
  - PyTorch 기반 L2 logistic/선택적 MLP16
  - paired GPU/DLA BCE + margin
  - calibration/build 환경별 loss의 CVaR
  - RMS 및 robust MAD normalization
  - fixed image-wise train/selection partition
  - threshold는 selection subset에서 worst-environment balanced accuracy로 고정
- `evaluate_v14_path_probe.py`
  - 고정 weight/normalization/threshold만 사용
  - environment별 AUC, balanced accuracy, paired margin, confusion
  - class-wise AUC 하위 10% gate
  - internal validation 통과 시에만 frozen manifest 생성
  - frozen probe가 아니면 calibration blind 실행 거부

### 설정 교정

초기 v14 config의 경로가 현재 저장소와 맞지 않아 다음처럼 교정했다.

- `results/v13/splits_v13.json` →
  `chain_survival/results/v13/splits_v13.json`
- `results/v14` → `chain_survival/results/v14`
- `checkpoints/resnet50.pt` →
  `chain_survival/models/resnet50.pth`

### 정적 검증

명령:

```bash
python -m py_compile \
  chain_survival/scripts/capture_v14_path_features.py \
  chain_survival/scripts/train_v14_path_probe.py \
  chain_survival/scripts/evaluate_v14_path_probe.py
```

결과: 통과.

추가로 synthetic 3-environment paired dataset에서 normalization, CVaR 학습,
threshold 선택을 실행해 logistic probe가 동작함을 확인했다.

### 구현 단계 판정

- F0 코드 정적 검증: **GO**
- 실제 engine smoke test: 다음 실행
- blind split: 열지 않음

## 2026-07-29 14:15 KST — F0 실제 engine smoke test

### 명령

```bash
python chain_survival/scripts/capture_v14_path_features.py \
  --model resnet50 \
  --boundaries layer4.0 \
  --calibrations calib_shadow_1 \
  --builds 0 \
  --image-split surrogate_train \
  --image-range 0:2 \
  --n-images 2 \
  --n-calib 16 \
  --strict-int8 \
  --allow-output-reformat-fallback \
  --output-dir chain_survival/results/v14/smoke_path_features
```

### 결과

- GPU/DLA record: 2/2 성공
- GPU inspector:
  - detailed: true
  - strict INT8 compute: true
  - compute layer: 59
- DLA inspector:
  - detailed: true
  - strict INT8 compute: true
  - DLA partition: 1
  - GPU 측 layer는 입력/출력 reformat과 빈 shape helper뿐
- DLA build warning:
  - FP32 output identity는 DLA 미지원이므로 GPU로 이동
  - protocol이 허용한 output reformat이며 compute fallback은 아님

Feature schema:

- `feature_4x4`: `(2, 2048, 4, 4)`, float16
- channel mean/std/RMS/max-abs: `(2, 2048)`, float32
- quantized histogram: `(2, 32)`, float32
- labels/image ids/paths 및 endpoint occupancy 포함

두 path 사이 mean absolute difference:

| Feature | Mean absolute difference |
|---|---:|
| pooled 4×4 | 0.171413 |
| channel mean | 0.061677 |
| channel RMS | 0.058738 |
| normalized histogram | 0.001213 |

이는 2장 smoke이므로 공격 가능성 지표로 해석하지 않는다.

### 구현 보강

Inspector에 `DLA`, input/output reformat, empty shape helper 이외의 GPU compute가
있으면 실패하도록 `no_compute_fallback` 검사를 추가했다.

### 판정 및 정리

- strict-INT8 GPU/DLA capture: **GO**
- feature schema: **GO**
- smoke artifact 183 MB는 과학 결과에 사용하지 않고 삭제한다.
- full F0에서는 protocol대로 calibration 200장, train image 512장을 사용한다.

## 2026-07-29 14:17–15:00 KST — F0 shadow train feature 본 캡처

### v13 layer4.2 재사용 검증

v14 protocol §4.3에 따라 v13 `layer4.2` engine 12개를 재사용했다.

- source ONNX SHA-256:
  `e9737b1e4a14f333743f0cab11e29326432f5b9509f6c42587aac758665edf96`
- calibration: `calib_shadow_1/2`, 각 200장
- build: calibration당 3
- backend: GPU/DLA
- strict layer INT8: true
- v13↔v14 12개 engine hash 일치: **12/12**

Engine/inspector/cache는 복제 대신 hard link를 사용했다. 따라서 재사용 조건이
원본과 byte-identical하며 저장 공간도 중복 사용하지 않는다.

### 실행 명령

```bash
python chain_survival/scripts/capture_v14_path_features.py \
  --model resnet50 \
  --boundaries layer4.0,layer4.1,layer4.2 \
  --calibrations calib_shadow_1,calib_shadow_2 \
  --builds 0,1,2 \
  --image-split surrogate_train \
  --image-range 0:512 \
  --n-images 512 \
  --n-calib 200 \
  --strict-int8 \
  --allow-output-reformat-fallback \
  --output-dir chain_survival/results/v14/path_features
```

### 실행 결과

| 항목 | 결과 |
|---|---:|
| 예상 조건 | 36 |
| 성공 | 36 |
| 실패 | 0 |
| 신규 build/capture (`layer4.0/4.1`) | 24 |
| 재사용 engine 신규 capture (`layer4.2`) | 12 |
| strict DLA inspector 통과 | 18/18 |
| DLA no-compute-fallback 통과 | 18/18 |
| feature SHA-256 고유 개수 | 36/36 |
| schema/NaN/Inf 오류 | 0 |

- 모든 조건의 image-path hash:
  `e8654d1351b1ade99e4b40c24e58da9616a6eed1d88620d32a658378939c81be`
- 결과 디렉터리 크기: 2.7 GB
- feature file: 36
- engine/inspector: 36/36
- sample별 empirical endpoint occupancy 최대: 0.001435

Endpoint 값은 sample 내부 관측 min/max occupancy로 calibration endpoint의 정확한
대용값은 아니다. 또한 F0 path probe gate에는 endpoint 조건이 없으므로 이 값만으로
중단하지 않는다. F2 amplifier 단계에서는 quantization endpoint를 cache/scale
기준으로 다시 정확히 계산해야 한다.

### 판정

- F0 shadow train capture: **GO**
- 다음 단계: low-dimensional scale-normalized logistic probe 선택
- blind split: 열지 않음

## 2026-07-29 15:01–15:06 KST — F0 low-dimensional logistic probe

### 첫 실행 중 구현 오류

26개 단일 low-dimensional 후보 학습은 완료됐으나, 최종 선택 model을 NPZ로
내보낼 때 CUDA tensor에 직접 `.numpy()`를 호출해 export만 실패했다.

```text
TypeError: can't convert cuda:0 device type tensor to numpy
```

조치:

- 모든 model parameter를 `.detach().cpu().numpy()`로 변환하도록 수정
- feature family 결합(`+`)을 지원하도록 loader/evaluator 보강
- 같은 seed 1401, 같은 image partition, 같은 epoch로 전체 재실행

첫 실행 수치와 재실행 수치가 동일해 학습 재현성을 확인했다.

### 최종 실행 명령

```bash
python chain_survival/scripts/train_v14_path_probe.py \
  --run-index chain_survival/results/v14/path_features/run_index.json \
  --features \
channel_rms,channel_mean,channel_std,quantized_histogram,\
channel_rms+quantized_histogram,\
channel_mean+channel_std+channel_rms+quantized_histogram,\
consensus_subspace_energy \
  --normalization rms,mad \
  --probe logistic \
  --group-key calibration,build \
  --objective cvar \
  --cvar-alpha 0.5 \
  --pair-margin 0.4 \
  --epochs 160 \
  --learning-rate 0.02 \
  --l2 0.0001 \
  --seed 1401 \
  --output chain_survival/results/v14/path_probe_selection.json
```

### 데이터와 선택 규칙

- image: `surrogate_train` 512장
- 고정 partition: train 410 / selection 102
- 모든 calibration/build/backend에 같은 image partition 적용
- environment: 2 calibrations × 3 builds
- threshold: selection subset의 worst-environment balanced accuracy 최대화
- model 선택: worst AUC, worst balanced accuracy, 단순성 순

### 주요 결과

| Boundary / feature / normalization | Worst AUC | Worst balanced accuracy |
|---|---:|---:|
| layer4.0 channel mean MAD | 0.566 | 0.574 |
| layer4.0 all-low-dim RMS | 0.736 | 0.755 |
| layer4.1 histogram MAD | 0.794 | 0.740 |
| layer4.2 histogram RMS | **0.843** | **0.804** |
| layer4.2 top-8 energy MAD | 0.455 | 0.505 |

선택 후보:

- boundary: `layer4.2`
- feature: `quantized_histogram`
- normalization: RMS
- probe: logistic
- threshold: 0.546506
- mean environment AUC: 0.879
- worst environment AUC: 0.843
- worst environment balanced accuracy: 0.804

### Gate

요구:

- worst AUC ≥ 0.92
- worst balanced accuracy ≥ 0.85

관측:

- worst AUC 0.843
- worst balanced accuracy 0.804

**Low-dimensional Probe A: NO-GO**

Protocol §6.5에 따라 고차원 pooled 4×4 logistic을 다음으로 한 번 평가한다.
Probe A 전체가 실패했으므로 이후에도 필요하면 small MLP16을 제한적으로 평가할
수 있다. 아직 internal validation이나 blind calibration은 열지 않는다.

## 2026-07-29 15:07–15:12 KST — F0 single-boundary 최종 대안

### Pooled 4×4 logistic

명령 요약:

```bash
python chain_survival/scripts/train_v14_path_probe.py \
  --features pooled4x4 --normalization rms --probe logistic \
  --epochs 160 --learning-rate 0.005 \
  --output chain_survival/results/v14/path_probe_pooled_selection.json
```

| Boundary | Worst AUC | Worst balanced accuracy |
|---|---:|---:|
| layer4.0 | 0.671 | 0.672 |
| layer4.1 | 0.566 | 0.598 |
| layer4.2 | 0.326 | 0.500 |

고차원 spatial feature는 histogram 후보를 개선하지 못했다.

### Small MLP16

Probe A 전체 실패 후 protocol §6.5에 따라 histogram과 all-low-dimensional
결합에 한정해 MLP16을 평가했다. 동일 실행에 logistic도 포함해 비교 기준을
유지했다.

```bash
python chain_survival/scripts/train_v14_path_probe.py \
  --features \
quantized_histogram,channel_mean+channel_std+channel_rms+quantized_histogram \
  --normalization rms,mad \
  --probe logistic,mlp16 \
  --epochs 200 \
  --output chain_survival/results/v14/path_probe_nonlinear_selection.json
```

최우수:

- boundary: `layer4.2`
- feature: 32-bin quantized histogram
- normalization: MAD
- probe: MLP16
- threshold: 0.427753
- worst environment AUC: **0.905229**
- mean environment AUC: 0.930331
- worst balanced accuracy: **0.843137**
- mean paired probability margin: 0.532047

Gate 대비:

| Metric | 요구 | 관측 | 판정 |
|---|---:|---:|---|
| Worst AUC | 0.92 | 0.905 | 실패 |
| Worst balanced accuracy | 0.85 | 0.843 | 실패 |

두 지표가 근접했지만 사전 기준을 낮추지 않는다.

### F0 판정

**Single-boundary F0: NO-GO**

Protocol §6.9/§7에 따라 multi-layer path trajectory를 정확히 1회 수행한다.
기본 후보 네 경계를 완성하기 위해 아직 없는 `layer3.5 Add`를 같은 2
calibrations × 3 builds × GPU/DLA 조건에서 캡처한다. 그 전까지 internal
validation과 blind split은 열지 않는다.

## 2026-07-29 15:16–15:37 KST — F1 multi-layer path trajectory

### layer3.5 추가 hardware capture

조건:

- boundary: `layer3.5 Add`
- calibration: `calib_shadow_1/2`, 각 200장
- build: 각 3
- backend: strict-INT8 GPU/DLA
- image: `surrogate_train` 0:512

결과:

- 성공 12/12
- 실패 0
- strict inspector 12/12
- DLA no-compute-fallback 6/6

따라서 F1 source는 4 boundaries × 2 calibrations × 3 builds × 2 backends =
48/48 조건이 완성됐다.

### Trajectory assembler

신규:

- `assemble_v14_path_trajectory.py`

각 경계별 독립 engine output에서 다음만 offline 결합한다.

- normalized channel RMS 32-bin histogram
- quantized-value occupancy 32-bin histogram
- spatial 4×4 energy 16개
- `layer4.2` top-8 consensus energy
- 연속 경계 간 log-energy ratio 3개

최종 차원:

$$
4\times(32+32+16)+8+3=331.
$$

이는 protocol의 512차원 제한을 만족한다. source graph에 multi-output을 추가하지
않았고, 경계별 engine을 분리한 채 동일 image id로만 offline 결합했다.

조립 결과:

- trajectory record: 12
- strict source: 12/12
- no-compute-fallback source: 12/12
- NaN/Inf: 0

### F1 logistic selection

```bash
python chain_survival/scripts/train_v14_path_probe.py \
  --run-index chain_survival/results/v14/path_trajectory/run_index.json \
  --features trajectory \
  --normalization rms,mad \
  --probe logistic \
  --objective cvar \
  --cvar-alpha 0.5 \
  --pair-margin 0.4 \
  --epochs 200 \
  --seed 1401 \
  --output \
chain_survival/results/v14/path_trajectory_logistic_selection.json
```

| Normalization | Worst AUC | Mean AUC | Worst balanced accuracy |
|---|---:|---:|---:|
| RMS | 0.938485 | 0.949314 | 0.892157 |
| MAD | **0.938774** | **0.949843** | **0.892157** |

선택:

- feature: 331-d trajectory
- normalization: MAD
- probe: logistic
- threshold: 0.450366
- mean paired probability margin: 0.617691

Selection subset gate:

- worst AUC ≥0.92: 통과
- worst balanced accuracy ≥0.85: 통과

### 현재 판정

- F1 selection: **PROVISIONAL GO**
- MLP16: logistic이 통과했으므로 실행하지 않음
- 다음: 학습에 쓰지 않은 `mechanism_discovery` index 128–511에서 fixed
  probe/threshold internal validation
- blind split: 아직 열지 않음

## 2026-07-29 15:38–15:42 KST — F1 independent internal validation

### Capture

- image: `mechanism_discovery[128:512]`, 384장
- calibration: `calib_shadow_1/2`
- build: 각 3
- boundary/backend source: 48/48 성공
- engine rebuild: 없음. selection 때 사용한 동일 engine을 재사용
- trajectory: 331차원, 12/12 strict/no-compute-fallback

### Fixed probe 평가

변경하지 않은 항목:

- feature family 및 순서
- MAD center/scale
- logistic weight/bias
- threshold 0.450366

명령:

```bash
python chain_survival/scripts/evaluate_v14_path_probe.py \
  --probe \
chain_survival/results/v14/path_trajectory_logistic_selection.json \
  --run-index \
chain_survival/results/v14/path_trajectory_internal/run_index.json \
  --image-split mechanism_discovery \
  --image-range 128:512 \
  --calibrations calib_shadow_1,calib_shadow_2 \
  --builds 0,1,2 \
  --gate-mode shadow \
  --output \
chain_survival/results/v14/path_probe_internal_validation.json \
  --freeze-output chain_survival/results/v14/path_probe_frozen.json
```

결과:

| Metric | 요구 | 관측 |
|---|---:|---:|
| Worst environment AUC | ≥0.92 | **0.932332** |
| Mean environment AUC | - | 0.941482 |
| Worst balanced accuracy | ≥0.85 | **0.859375** |
| Mean balanced accuracy | - | 0.870877 |
| Class AUC lower 10% | ≥0.75 | **0.833333** |
| Class AUC median | - | 1.0 |
| Class AUC minimum | 참고 | 0.25 |

최소 class AUC는 낮지만 사전 gate는 하위 10% quantile이므로 threshold를
사후 변경하지 않는다.

### 판정

- independent shadow internal validation: **GO**
- fixed probe manifest 생성:
  `chain_survival/results/v14/path_probe_frozen.json`
- F1은 아직 unseen calibration 검증 전이므로 최종 GO가 아님

### 첫 blind 개봉 결정

Feature/probe/normalization/threshold가 모두 동결됐으므로 protocol §6.8 및
config의 `open_calib_blind_1_for_path_only`에 따라 처음으로 다음을 연다.

- calibration: `calib_blind_1`
- image: `mechanism_discovery[512:768]` path holdout 256장

아직 봉인:

- `calib_blind_2/3`
- `threshold_validation`
- `boundary_blind`
- `final_logit_blind`
- `robustness`

Blind 결과를 보고 probe나 threshold를 재조정하지 않는다.

## 2026-07-29 15:43–16:23 KST — F1 `calib_blind_1` 최종 판정

### Hardware capture

조건:

- calibration: `calib_blind_1`, 200장
- image: `mechanism_discovery[512:768]`, 256장
- boundaries: `layer3.5`, `layer4.0`, `layer4.1`, `layer4.2`
- builds: 0/1/2
- backends: GPU/DLA

결과:

- engine/feature 조건 성공: 24/24
- 실패: 0
- strict inspector: 24/24
- DLA no-compute-fallback: 12/12
- trajectory: 331차원, 6/6 record 조립 성공

### Frozen probe 평가

```bash
python chain_survival/scripts/evaluate_v14_path_probe.py \
  --probe chain_survival/results/v14/path_probe_frozen.json \
  --run-index \
chain_survival/results/v14/path_trajectory_calib_blind1/run_index.json \
  --image-split mechanism_discovery \
  --image-range 512:768 \
  --calibrations calib_blind_1 \
  --builds 0,1,2 \
  --gate-mode blind \
  --output chain_survival/results/v14/path_probe_calib_blind1.json
```

고정 사항:

- weight/bias 변경 없음
- MAD center/scale 변경 없음
- threshold 0.450366 변경 없음
- blind score를 사용한 재학습/재선택 없음

### 결과

| Metric | 요구 | 관측 | 판정 |
|---|---:|---:|---|
| Worst-build AUC | ≥0.88 | **0.556564** | 실패 |
| Mean-build AUC | - | 0.580994 | - |
| Worst-build balanced accuracy | ≥0.80 | **0.546875** | 실패 |
| Mean-build balanced accuracy | - | 0.550781 | - |
| Class AUC lower 10% | ≥0.75 | **0.0** | 실패 |
| Class AUC median | - | 0.666667 | - |

Build별:

| Build | AUC | Balanced accuracy | GPU score mean | DLA score mean | Paired margin |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.601746 | 0.550781 | 0.690151 | 0.788518 | 0.098367 |
| 1 | 0.584671 | 0.546875 | 0.699114 | 0.782437 | 0.083324 |
| 2 | 0.556564 | 0.554688 | 0.704557 | 0.766319 | 0.061761 |

Shadow internal에서는 GPU score mean이 0.193–0.239, DLA score mean이
0.781–0.804였다. Blind calibration에서는 DLA score는 비슷하게 유지됐지만
GPU score가 0.690–0.705로 상승해 false positive가 발생했다.

### Calibration drift 진단

Frozen MAD feature space에서 shadow internal과 blind의 평균 DLA-GPU trajectory
delta를 비교했다.

- shadow↔blind delta cosine: **0.232287**
- delta norm:
  - shadow: 1.415257
  - blind: 1.888005
- frozen weight 방향의 logit delta:
  - shadow: 4.063264
  - blind: 0.589924
- blind build 간 delta cosine:
  - build0↔1: 0.592869
  - build0↔2: 0.539014
  - build1↔2: 0.524474

성분별 weighted path delta:

| 성분 | Shadow | Blind | Delta cosine |
|---|---:|---:|---:|
| layer3.5 | 1.579603 | 0.353849 | 0.409428 |
| layer4.0 | 0.012186 | -0.024446 | -0.246872 |
| layer4.1 | 0.635232 | -0.939378 | -0.235741 |
| layer4.2 | 1.831136 | 1.169590 | 0.247111 |
| energy ratio | 0.005107 | 0.030310 | -0.429799 |

해석:

- trajectory magnitude가 사라진 것이 아니라 signed path identity가 calibration
  변경에 따라 회전했다.
- 특히 `layer4.1` contribution이 양수에서 음수로 뒤집혔다.
- frozen probe는 DLA score를 유지했지만 unseen calibration의 GPU feature를
  DLA-positive 영역으로 이동시켜 backend identity가 아니라 calibration-local
  absolute trajectory를 일부 학습했다.
- raw-only 성공은 아니며 scale-invariant summaries를 사용했음에도 실패했으므로,
  단순 RMS/MAD normalization만으로 calibration identity leakage를 제거하지 못했다.

### 최종 F0/F1 Gate

Protocol §6.9:

- single-boundary 실패 후 multi-layer trajectory 1회만 허용
- multi-layer unseen calibration AUC <0.80이면 TM-W factorized attack 종료

관측 worst AUC 0.556564이므로:

**F0/F1 TM-W PATH FINGERPRINT: NO-GO**

따라서 다음을 실행하지 않는다.

- F2 path signal amplification
- T0 trigger fingerprint/optimization
- G0 conjunctive gate
- G1 merge/rebuild
- E0 final ASR/CA evaluation

이는 실행 미완료가 아니라 사전 등록된 중단 규칙의 적용이다.

### 봉인 상태

열린 항목:

- `calib_shadow_1/2`
- `calib_blind_1`
- `surrogate_train[0:512]`
- `mechanism_discovery[128:512]`
- `mechanism_discovery[512:768]`

계속 봉인:

- `calib_blind_2/3`
- `threshold_validation`
- `boundary_blind`
- `final_logit_blind`
- `robustness`

### 최종 artifact

- `chain_survival/results/v14/f0_f1_final_verdict.json`
- frozen probe SHA-256:
  `688cafeab5df91a304de6f81b083fb7890c84dc5bfa1377fc4a7fe9db5063842`
- probe model SHA-256:
  `676bf492d584294cc9b199a2fbfe5078f9608712bc07cbc4f5d9918c26cb41e3`
- internal validation SHA-256:
  `5ed7b966a73fdc738c6fba8125ad95b42875b427b3a07e7d9d115d1dc31bbbd5`
- blind result SHA-256:
  `6f06ae0b015b85f3270bc2c31ed45a8fe77444aa85faf8381721b42685a46d99`

### 과학적 결론

현재 strict-INT8 GPU-DLA activation은 paired difference와 shadow calibration
내 single-execution provenance signal을 제공한다. 네 경계의 331차원
scale-invariant trajectory는 independent shadow image에서 강한 path
분류 성능을 보였지만, unseen calibration에서 GPU feature가 DLA-positive
영역으로 이동해 붕괴했다.

따라서 현재 artifact가 지지하는 주장은 다음으로 제한한다.

> GPU-DLA path fingerprint는 shadow calibration 안에서는 읽을 수 있지만,
> 현재 정규화와 trajectory로는 calibration-invariant TM-W primitive가 아니다.

v14 protocol §19.4에 따라 완전한 F-DPCB 공격 주장은 하지 않으며,
backend provenance measurement와 calibration uncertainty가 exploitability
boundary라는 negative result로 기록한다.

## 2026-07-29 16:25 KST — 최종 artifact 전수 검증

### 검증 범위

`path_features/run_index.json`의 모든 record와 다음 연결 artifact를 전수
SHA-256 검증했다.

- feature NPZ
- TensorRT engine
- extracted ONNX
- detailed inspector JSON
- calibration cache

결과:

| 항목 | 결과 |
|---|---:|
| Capture record | 120 |
| Status OK | 120/120 |
| 고유 연결 artifact | 292 |
| 누락 | 0 |
| SHA-256 mismatch | 0 |
| Strict-INT8 record | 120/120 |
| DLA no-compute-fallback | 60/60 |

Record 구성:

- `surrogate_train[0:512]`, shadow1: 24
- `surrogate_train[0:512]`, shadow2: 24
- `mechanism_discovery[128:512]`, shadow1: 24
- `mechanism_discovery[128:512]`, shadow2: 24
- `mechanism_discovery[512:768]`, blind1: 24

### Verdict 참조 hash 검증

`f0_f1_final_verdict.json`이 참조하는 다음 6개 hash는 모두 현재 파일과 일치했다.

- frozen probe
- probe model
- selection manifest
- internal validation
- calibration-blind result
- capture index

Verdict SHA-256:

`d3c6e78b94032d0725e8f37be15bc628d73eb52989270897e43cc93b09b2bdcc`

### 코드 및 저장소 검증

```bash
python -m py_compile chain_survival/scripts/*.py common/scripts/*.py
git diff --check
```

두 명령 모두 통과했다.

- 현재 workspace 크기: 5.8 GB
- v14 binary/result는 재현 증거이므로 보존
- 실험 전 정리 후에도 사용 가능 공간: 207 GB

### 최종 상태

- v14 F0/F1 실험: **완료**
- TM-W factorized attack: **NO-GO at unseen calibration**
- protocol상 후속 공격 단계: **의도적으로 미실행**
- 모든 과정/명령/실패/판정: 본 문서에 기록 완료

## 2026-07-29 17:51 KST — 가능성 전수 후속 실험 시작

사용자가 V14 결과 이후 가능한 방향을 최대한 실험하도록 요청했다. 기존 V14
NO-GO 판정은 소급 변경하지 않고 별도 확장 실험으로 진행한다.

### 시작 전 무결성 및 자원 점검

```bash
pwd
rg --files -g 'AGENTS.md' -g 'V14*' -g '*V14*' -g '*v14*' \
  -g 'academic_research_plan_v13_trackB.md' -g 'EXPERIMENT_LOG_V13.md'
git status --short
df -h .
tail -n 160 chain_survival/EXPERIMENT_LOG_V14.md
python -m py_compile chain_survival/scripts/*.py common/scripts/*.py
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader
```

확인 결과:

- 기존 V14 capture record 120개와 결과 5.5 GB 보존
- 이전 전수 검증 결과: 연결 artifact 292개, 누락/hash mismatch 0
- strict-INT8 120/120, DLA no-compute-fallback 60/60
- 사용 가능 공간 약 207 GB
- 실행 장치: Jetson Orin의 `nvgpu`
- `scikit-learn`은 없고 PyTorch 2.11.0, SciPy 1.15.3,
  TensorRT 10.3.0 사용 가능
- 기존 코드 전체 `py_compile` 통과

시작점 hash:

- `f0_f1_final_verdict.json`:
  `d3c6e78b94032d0725e8f37be15bc628d73eb52989270897e43cc93b09b2bdcc`
- `path_features/run_index.json`:
  `2884fcdd75e9b83cf05fcbd6dc4eec7398e27cd2453edf1f976a331c8a2bf937`
- 원 V14 protocol:
  `149bd5aa02939c0321fe86f85c03b9c0e8826a6e0a4ecbff618786320300ecec`
- V14 config:
  `38d9555f1378004b78a604f0c021140eac583761fe2a9394d8a65fda69d83df5`

### 데이터 상태

`calib_blind_1`은 이미 원 V14에서 한 번 열어 결과를 확인했으므로 이 확장에서
개발 데이터로만 재분류한다. 다시 blind evidence로 사용하지 않는다.

계속 봉인:

- `calib_blind_2/3`
- `threshold_validation`
- `boundary_blind`
- `final_logit_blind`
- `robustness`

### 확장 사전등록

신규 문서:

- `V14_POSSIBILITY_SWEEP_PROTOCOL.md`

세 독립 track을 등록했다.

1. P-track: 512차원 이하 single-execution calibration-invariant path 표현 전수
   탐색
2. T-track: 공격 성공 주장과 분리한 GPU/DLA 공통 trigger 검출성 실험
3. Q-track: 열린 calibration 안에서 calibration-aware TM-Q 상한 측정

P-track 개발 gate:

- strict-ready: three-way LOCO worst AUC >= 0.88, worst BA >= 0.80
- promising: worst AUC >= 0.80, worst BA >= 0.70

T-track gate는 원 V14 T0 기준을 유지한다. P/T가 열린 세 calibration에서 모두
통과하기 전에는 full attack, weight merge, sealed calibration 결합 평가를 하지
않는다.

## 2026-07-30 — P-track 구현 및 실행 준비

신규 코드:

- `chain_survival/scripts/sweep_v14_path_invariance.py`

코드가 강제하는 조건:

- 허용 calibration role은 `calib_shadow_1`, `calib_shadow_2`,
  개발용 `calib_blind_1`뿐이다.
- 한 sample의 한 backend 실행에서 계산되는 특징만 사용한다.
- 모든 특징은 512차원 이하이다.
- scaler와 sign-consistent coordinate 선택은 각 LOCO fold의 train
  calibration에만 fit한다.
- threshold 역시 train environment에서만 고정한다.

표현군 11개:

- original V14 trajectory
- global quantization/sign/endpoint
- channel distribution quantile
- channel-identity block
- within-sample rank projection
- spatial energy
- cross-layer ratio
- 위 특징의 compact hybrid 4종

probe 4개:

- logistic ERM
- environment-CVaR logistic
- sign-consistent coordinate + CVaR logistic
- MLP16 CVaR

검증:

```bash
python -m py_compile chain_survival/scripts/sweep_v14_path_invariance.py
python chain_survival/scripts/sweep_v14_path_invariance.py --help
git diff --check
```

모두 통과했다.

파일 hash:

- 확장 protocol:
  `a6c750f6006e346d244a9ec97e5449a912308868993726c67b7d30fa4282f7b4`
- P-track sweep code:
  `3fd1971a6cd4af343580ef7a9e172d0a1e0a89f9810bcb99821ceb24b1452606`

실제 전체 sweep 전에 cache 생성과 학습/evaluation 경로를 1개 후보, 5 epoch로
smoke test한다. smoke 결과는 후보 판정에 사용하지 않는다.

```bash
python chain_survival/scripts/sweep_v14_path_invariance.py \
  --families global_quant \
  --algorithms logistic_erm \
  --epochs 5 \
  --output chain_survival/results/v14/possibility_sweep/path_smoke.json
```

## 2026-07-30 11:18 KST — P-track smoke 완료

smoke가 오류 없이 완료됐다.

- 파생 cache: 30개 environment/backend/view artifact, 총 98 MB
- 각 artifact에 동일한 11개 표현 저장
- 표현 차원: 54-502, 모두 512 이하
- smoke 후보: `global_quant + logistic_erm`, 5 epoch
- LOCO worst AUC 0.552368
- LOCO worst BA 0.523438
- 개발 판정: negative

이 수치는 코드 경로 검증용이므로 후보 비교/선택에는 사용하지 않는다.

smoke artifact:

- `chain_survival/results/v14/possibility_sweep/path_smoke.json`
- SHA-256:
  `06d3018907389641a90c570d81c1ac2437a0591ae26c7a82e8e5e35421a6a47c`

전체 11 feature family x 4 algorithm = 44개 후보, 후보당 3개 held-out
calibration fold와 fold당 3 build 평가를 140 epoch 상한으로 실행한다.

```bash
python chain_survival/scripts/sweep_v14_path_invariance.py \
  --epochs 140 \
  --output \
  chain_survival/results/v14/possibility_sweep/path_invariance_sweep.json
```

## 2026-07-30 11:28 KST — P-track 1차 전체 sweep 완료

44개 후보가 모두 완료됐다.

artifact:

- `chain_survival/results/v14/possibility_sweep/path_invariance_sweep.json`
- SHA-256:
  `5a031ddf2020775a12d8e580c31c65c83f88c472c755b7ea25290154cad25dee`

최고 후보:

- feature: `global_quant`, 144차원
- algorithm: `sign_cvar`
- three-way LOCO worst AUC: 0.834961
- three-way LOCO worst BA: 0.556641
- 원 TM-W 방향(shadow1/2 train -> opened blind1): AUC 0.834961,
  BA 0.556641
- 개발 판정: negative

다른 상위 AUC:

| 표현/알고리즘 | LOCO worst AUC | worst BA | TM-W AUC |
|---|---:|---:|---:|
| global_quant / sign_cvar | 0.834961 | 0.556641 | 0.834961 |
| rank+quant+spatial / sign_cvar | 0.829773 | 0.568359 | 0.829773 |
| dist+quant+cross / logistic_cvar | 0.811890 | 0.556641 | 0.813171 |
| all_compact / sign_cvar | 0.805725 | 0.564453 | 0.864777 |

핵심 진단:

- 열린 blind1의 build별 AUC는 최고 후보에서 0.834961-0.858856으로 원
  trajectory보다 크게 좋아졌다.
- 그러나 shadow train threshold 0.479740에서 blind1 GPU score 평균이
  0.725517-0.753136, DLA가 0.947983-0.949913으로 함께 위로 이동했다.
- 따라서 backend ordering은 일부 보존되지만 absolute threshold가 calibration
  이동을 견디지 못한다.
- 분포, channel block, rank, spatial, cross-layer 단독 특징은 대부분 chance
  수준이었다. 유효한 신호는 주로 global quantization/sign/endpoint 요약에
  있었다.

### P-track 2차: paired midpoint 제약 사전등록

1차 결과를 보고 선택한 후속 가설은 training 때 paired GPU/DLA logit midpoint를
0으로 제약해 calibration common-mode 이동에 직교하는 weight를 찾는 것이다.
inference는 여전히 한 backend의 single execution만 사용한다.

추가 알고리즘:

- centered CVaR, center weight 0.1/1/10
- sign-consistent coordinate + center weight 1
- calibration midpoint-stable coordinate 64개 + center weight 1

center penalty:

```text
mean(((logit_gpu + logit_dla) / 2)^2)
```

1차에서 AUC 신호를 보인 네 표현만 사용한다. 총 4 x 5 = 20개 후보이며 동일한
three-way LOCO를 적용한다.

```bash
python chain_survival/scripts/sweep_v14_path_invariance.py \
  --families \
  global_quant,hybrid_rank_quant_spatial,hybrid_dist_quant_cross,hybrid_all_compact \
  --algorithms \
  centered_cvar_0p1,centered_cvar_1,centered_cvar_10,sign_centered_1,stable_centered_1 \
  --epochs 180 \
  --output \
  chain_survival/results/v14/possibility_sweep/path_centered_sweep.json
```
