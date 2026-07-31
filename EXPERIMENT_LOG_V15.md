# v15 Full Deployment Pipeline 실험 진행 일지

이 문서는 `academic_research_plan_v15_pipeline_aware_dclbd.md`에 따른 실제 실행
명령, 입력과 출력 artifact, 환경, 수치 결과, 실패 원인 및 Go/No-Go 판단을
시간순으로 누적 기록한다. 계획서와 실험 기록을 분리하며, 모든 신규 결과는
성공 여부와 관계없이 이 문서에 먼저 요약한다.

## 기록 및 재현성 규칙

- 모든 시간은 Asia/Seoul 기준으로 기록한다.
- source, export, quantization, compilation, backend, calibration, build 상태를
  서로 다른 state ID로 기록한다.
- 모든 artifact는 source hash, 설정, calibration ID, build ID, backend 및
  생성 도구 버전을 manifest에 포함한다.
- 동일 이미지를 모든 비교 state에서 paired evaluation한다.
- 평균뿐 아니라 environment별 값과 worst environment를 보고한다.
- proxy/ONNX/fake-quant 결과와 실제 TensorRT GPU/DLA artifact 결과를 구분한다.
- inspector로 precision과 partition을 확인하지 못한 결과는 strict-INT8 또는
  DLA 결과로 인정하지 않는다.
- 한 번 연 calibration/evaluation split은 다시 blind evidence로 부르지 않는다.
- threshold와 model selection 후 post-hoc blind threshold 조정을 금지한다.
- 각 단계는 Gate, 판단, 다음 조치를 함께 기록한다.

## Blind split registry

| Split | 초기 상태 | 사용 규칙 |
|---|---|---|
| `calib_shadow_1` | development | calibration/build 개발 |
| `calib_shadow_2` | development | calibration/build 개발 |
| `calib_blind_1` | opened in v14 | v15 development로만 사용 |
| `calib_blind_2` | sealed | first sealed calibration |
| `calib_blind_3` | sealed | final sealed calibration |
| `surrogate_train` | development | surrogate/attack 학습 |
| `mechanism_discovery` | development | state/mechanism 탐색 |
| `threshold_validation` | sealed | threshold/model selection |
| `boundary_blind` | sealed | boundary-level blind 평가 |
| `final_logit_blind` | sealed | 최종 logit/CA/ASR 평가 |
| `robustness` | sealed | robustness 평가 |

## 2026-07-30 12:55 KST — v15 시작 및 P0 환경 감사

### 목적

- v15 계획서를 단일 기준으로 고정한다.
- 현재 Jetson/TensorRT/DLA 실행 가능성을 확인한다.
- 보존한 모델, ONNX, split 및 재사용 코드의 hash와 무결성을 확인한다.
- P1 이후의 모든 artifact가 따를 manifest 기준을 만든다.

### 기준 계획

- 파일: `academic_research_plan_v15_pipeline_aware_dclbd.md`
- SHA-256:
  `2ca5733a001e32137b082fecddbf78c931e743ae937d92416f66d480e552038d`
- 버전: v15.0 — Full Deployment Pipeline Study

### 실행한 점검

```bash
uname -a
lscpu
cat /etc/nv_tegra_release
nvidia-smi
/usr/src/tensorrt/bin/trtexec --version
python3 -c "import torch, tensorrt, onnx, onnxruntime, numpy, scipy"
ls -l /dev/nvidia0 /dev/nvhost-gpu \
  /dev/nvhost-ctrl-nvdla0 /dev/nvhost-ctrl-nvdla1
sha256sum academic_research_plan_v15_pipeline_aware_dclbd.md \
  chain_survival/models/resnet50.pth \
  chain_survival/onnx/resnet50.onnx \
  chain_survival/results/v13/splits_v13.json \
  chain_survival/results/v13/layer4.2_consensus_subspace.npz \
  common/scripts/trt_runtime.py
python3 -m py_compile common/scripts/*.py chain_survival/scripts/*.py
```

### 환경 결과

| 항목 | 값 |
|---|---|
| Platform | NVIDIA Jetson Orin, aarch64 |
| JetPack/L4T | R36.5.0 |
| Kernel | 5.15.185-tegra |
| CPU | 12 × Cortex-A78AE |
| GPU | Orin, CUDA 사용 가능 |
| CUDA driver/runtime 표기 | 540.5.0 / CUDA 12.6 |
| TensorRT | 10.3.0 |
| Python | 3.10.12 |
| PyTorch / torchvision | 2.11.0 / 0.26.0 |
| ONNX / ONNX Runtime | 1.22.0 / 1.23.2 |
| NumPy / SciPy | 2.2.6 / 1.15.3 |
| scikit-learn | 미설치 |

장치 확인:

- `/dev/nvidia0`: 존재
- `/dev/nvhost-gpu`: 존재
- `/dev/nvhost-ctrl-nvdla0`: 존재
- `/dev/nvhost-ctrl-nvdla1`: 존재
- `torch.cuda.is_available()`: `True`
- CUDA device: `Orin`

scikit-learn은 설치하지 않고 NumPy/PyTorch/SciPy로 분석을 구현한다. 새로운
dependency를 도입하기 전에 기존 환경으로 재현 가능한지를 우선한다.

### 입력 artifact 무결성

| Artifact | SHA-256 |
|---|---|
| ResNet-50 checkpoint | `3ce1c0adebfa0371435c97516dbb1a0c5ac22ad708b2e30d02b9741c2800a011` |
| ResNet-50 ONNX | `e9737b1e4a14f333743f0cab11e29326432f5b9509f6c42587aac758665edf96` |
| v13/v15 split registry | `6c2901a5c68710ed8bff3a7f609a46045f9f988cec06f173d7630f45501bbffb` |
| `layer4.2` consensus subspace | `d78d171b57d82d9226648be1187b2c0cf85848135bd46cf38f9656a1cabd5e19` |
| strict TensorRT runtime helper | `6d0686b6cc0e0e3b1a88b8d4089bb032f3ec63fec94057b70a46bf380c88c996` |

ResNet-50 ONNX는 opset 17, static `1×3×224×224`, 122 nodes이며 입력은
`input`, 출력은 `logits`다.

### 데이터 무결성

- ImageNet validation root:
  `/media/airlab_compression/nvme_storage/imagenet_val`
- inventory: 1,000 classes, 50,000 images
- split registry entries: 13,500
- 존재하는 split image: 13,500/13,500
- split 간 중복: 기존 registry 생성 시 0, P0 manifest에서 재검증 예정

### P0 예비 판정

- GPU execution: **GO**
- DLA device exposure: **GO**
- ImageNet paired state evaluation: **GO**
- 보존 artifact hash continuity: **GO**
- Python syntax validation: **GO**
- strict precision/partition inspector: 실제 smoke build 후 최종 판정

### 다음 조치

1. `inspect_pipeline_artifacts.py`로 P0 manifest를 생성한다.
2. S0/S1 reference와 TensorRT FP32/FP16/INT8 GPU/DLA smoke state를 만든다.
3. inspector에서 DLA placement와 precision을 검증한다.
4. P0가 최종 통과하면 DcL-BD 공식 baseline 재현을 시작한다.

## 2026-07-30 12:58–13:07 KST — P0 manifest와 strict GPU/DLA smoke

### 신규 구현

- `chain_survival/scripts/inspect_pipeline_artifacts.py`
  - 환경, package, device, input hash 및 split 무결성 manifest
  - TensorRT detailed inspector의 strict-INT8 및 DLA compute-fallback gate
- `chain_survival/scripts/build_pipeline_states.py`
  - S0/S1 lineage와 S2/S3/S7/S8 engine build
  - calibration/build/state별 engine, cache, inspector 및 SHA-256 index
- `common/scripts/trt_runtime.py`
  - v15 cross-DLA-core 실험을 위해 `dla_core` 인자를 추가

### 환경 manifest

명령:

```bash
python3 chain_survival/scripts/inspect_pipeline_artifacts.py
```

산출물:

- `chain_survival/results/v15/manifest/p0_environment.json`

결과:

- split entry: 13,500
- unique entry: 13,500
- duplicate: 0
- 존재하는 이미지: 13,500/13,500
- CUDA: true
- DLA0/DLA1 device: true/true
- 필수 artifact 누락: 0

### strict smoke 명령

```bash
python3 chain_survival/scripts/capture_v13_boundaries.py \
  --output-dir chain_survival/results/v15/smoke_boundary \
  --boundaries layer4.2 \
  --builds 1 \
  --calibrations 1 \
  --n-calib 16 \
  --n-discovery 2 \
  --allow-gpu-fallback

python3 chain_survival/scripts/inspect_pipeline_artifacts.py \
  --engine-index \
    chain_survival/results/v15/smoke_boundary/run_index.json
```

### inspector 결과

| Backend | 전체 layer | compute layer | strict INT8 compute | DLA partition | compute fallback | Gate |
|---|---:|---:|---:|---:|---:|---|
| GPU | 69 | 66 | true | 0 | N/A | GO |
| DLA | 3 | 1 | true | 1 | 0 | GO |

DLA engine의 GPU 측 두 layer는 입력과 최종 FP32 출력 reformat뿐이다. 최종
INT8-to-FP32 cast는 GPU inspector에서 generated `kgen`으로 나타나므로
activation format이 단일 INT8 입력과 단일 FP32 출력을 갖는 최종 cast인지
명시적으로 검사해 compute layer에서 제외했다.

두 smoke image의 `layer4.2 Add` GPU/DLA 차이:

| Feature | Mean absolute | RMS |
|---|---:|---:|
| pooled 4×4 | 1.439535 | 2.379881 |
| channel mean | 1.419295 | 2.190007 |
| channel max | 0.757461 | 1.511876 |

이 값은 2장 smoke이므로 mechanism 또는 공격 가능성의 증거로 사용하지 않는다.

### P0 최종 판정

- artifact lineage: **GO**
- split/data integrity: **GO**
- 실제 strict GPU INT8: **GO**
- 실제 strict DLA INT8: **GO**
- DLA no-compute-fallback: **GO**

**P0: GO**

## 2026-07-30 13:00–13:13 KST — P1 DcL-BD upstream 고정과 compatibility preflight

### upstream 고정

```bash
git clone --depth 1 \
  https://github.com/SeekingDream/DLCompilerAttack.git \
  common/external/DLCompilerAttack
git -C common/external/DLCompilerAttack rev-parse HEAD
```

- repository: `SeekingDream/DLCompilerAttack`
- commit: `8b4234260fc6eab22adec455a2227b467ff2176b`
- Python compileall: 통과

### 공식 runner 직접 실행 결과

```bash
cd common/external/DLCompilerAttack
python3 main.py --help
```

실패:

- `datasets`: 미설치
- TVM: 미설치이며 upstream `src/dlcl.py`와 `src/abst_cl_model.py`가
  import 시점에 강제 요구
- pandas: 현재 NumPy 2.2.6과 binary ABI 불일치
- torch.compile default Inductor: aarch64용 Triton 부재로 `TritonMissing`

`torch.compile(..., backend="aot_eager")`는 실행됐지만 eager와 최대 차이가
0이므로 compiler-inconsistency baseline으로 사용하지 않는다.

이는 DcL-BD 공격의 No-Go가 아니라 upstream x86/CUDA 환경과 Jetson 환경의
compatibility 차이다.

### CIFAR-10 입력

공식 upstream이 사용하는 `uoft-cs/cifar10`의 Hugging Face Parquet를 직접
받았다.

| Split | Rows | Size | SHA-256 |
|---|---:|---:|---|
| train | 50,000 | 119,705,255 | `8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883` |
| test | 10,000 | 23,940,850 | `841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de` |

원 CIFAR 서버는 약 20–40KB/s로 제한돼 4%에서 중단했으며, 불완전 tar는
휴지통으로 이동했다.

### Jetson compatibility runner

신규:

- `chain_survival/scripts/reproduce_dclbd_baseline.py`

유지한 upstream 요소:

- task 0 CIFAR-10 ConvNet 구조
- SGD/Cosine clean training 설정
- 첫 Conv-BN 이후 model split
- 8×8 left-up trigger와 clean maximum + K objective
- source-clean/source-trigger/compiled-clean 대 compiled-trigger guard
- source-clean/source-trigger clean label 및 compiled-trigger target tail loss

명시적 adaptation:

- `datasets` 대신 동일 Parquet의 Arrow direct reader
- TVM/Inductor 대신 upstream 지원 compiler인 ONNX Runtime CPU
- 전체 clean embedding materialization 대신 수학적으로 같은 running maximum
- deterministic cuDNN 설정

산출물:

- `chain_survival/results/v15/dclbd_baseline/preflight.json`

Preflight: **GO**

다음은 500 train/200 test의 1-epoch smoke로 데이터, ONNX export, guard 및
tail 경로를 검증한 뒤 full 100/10/50 epoch 실행으로 전환한다.

## 2026-07-30 13:13–14:08 KST — P1 DcL-BD compatibility full reproduction

### 실행 설정

- clean model: 100 epochs
- trigger optimization: 10 epochs
- malicious tail optimization: 50 epochs
- compiler: ONNX Runtime CPU
- upstream architecture와 공격 목적함수는 유지하고, 앞 절의 Jetson
  compatibility adaptation만 적용했다.
- wall time: 3,299.72초(약 55분)

### 결과

| Metric | Result | v15 gate | Pass |
|---|---:|---:|---|
| clean model validation accuracy | 89.32% | reference | — |
| source clean accuracy after attack | 85.24% | drop ≤ 3%p | No (4.08%p) |
| compiled clean accuracy after attack | 85.26% | drop ≤ 3%p | No (4.06%p) |
| source triggered clean-label accuracy | 83.96% | diagnostic | — |
| source triggered ASR | 12.06% | ≤ 10% | No (2.06%p 초과) |
| compiled triggered ASR | 99.95% | ≥ 90% | **Yes** |

추가 mechanism evidence:

- guard separation threshold 0.95에서 분리 가능한 activation dimension 8개를
  찾았다.
- trigger loss는 6.4895에서 0.009883으로 감소했다.
- tail training 중 compiled-trigger ASR은 약 99.97%에 도달했다.
- 최종 compiled-trigger ASR 99.95%와 source-trigger ASR 12.06%의 큰
  차이는 compiler-conditional attack behavior가 실제로 형성됐음을 보인다.

### 판정

공격 동작 자체는 **재현**됐다. 특히 compiled-trigger ASR 기준은 큰 폭으로
통과했다. 다만 v15에서 사전에 고정한 세 조건을 동시에 만족해야 하는
strict gate는 source leak 2.06%p 초과와 clean accuracy drop 약 1.1%p
초과 때문에 **NO-GO**다. 따라서 이 결과를 "엄격한 전체 재현 성공"으로
표현하지 않고, **compiler-conditional behavior 재현 / strict gate 미달**로
기록한다.

이 artifact는 P4 생존성 추적 입력으로 보존하되, 이후 인과 결과와 공격 결과를
분리해서 보고한다.

### 산출물

- `chain_survival/results/v15/dclbd_baseline/result.json`
- `chain_survival/results/v15/dclbd_baseline/clean_convnet.pth`
- `chain_survival/results/v15/dclbd_baseline/trigger.pth`
- `chain_survival/results/v15/dclbd_baseline/attacked_model.pth`
- `chain_survival/results/v15/dclbd_baseline/ort_final.onnx`

주요 SHA-256:

- clean model: `4fae04212e0e8d4b397ee8b778ce4914f0b9afc8b404ff998fe074d607941148`
- trigger: `b26ae9c343866756f8292af0725df42188e565eb8bffe8b23817801a2c5f606a`
- attacked model: `1357c8abeb24ca4ad26eba35c5361b8b4882b2ba39f73e1eed0d4f9e57e96a5c`
- final ONNX: `203b0ff389d617567096c2dd71b2c36910526b59088ee4b40d6f0696a2d742f6`

## 2026-07-30 14:12–23:39 KST — P2 reduced pipeline atlas와 3 calibration × 3 build

### 캡처 복구

초기 S2/S3 캡처의 정확도가 20–27%로 비정상적으로 낮았다. 원인은
`common/scripts/trt_runtime.py`가 PyTorch H2D copy와 TensorRT 실행에 서로
다른 CUDA stream을 사용하면서 입력 copy 완료 전에 TensorRT가 입력을 읽을
수 있었던 race였다. TensorRT 실행을 `torch.cuda.current_stream()`으로
통일한 뒤 단일 샘플에서 S2와 PyTorch의 logit MAE가 `0.00038`이고 예측이
일치함을 확인했다. race 수정 전에 생성된 캡처는 폐기하고 3×3 전체를 다시
캡처했다.

### explicit Q/DQ 및 DLA compatibility

- 일반 ONNX Runtime Q/DQ는 per-channel/per-tensor 및 op 제한 변형 모두
  TensorRT 10.3의 첫 fused Conv에서 parse/build에 실패했다.
- legacy negative artifact는
  `states/calib_shadow_1/qdq/resnet50__calib_shadow_1__n200.ort_legacy.qdq.onnx`
  로 보존했다.
- NVIDIA ModelOpt 0.44.0의 max calibration Q/DQ는 S4 reference와 S5/S6
  engine 생성에는 성공했다.
- S5 explicit GPU는 ModelOpt output-layer policy 때문에 FP32 compute layer
  2개가 남아 strict INT8 gate를 통과하지 못했다.
- S6 explicit DLA는 모든 compute가 GPU로 fallback되고 DLA partition이 없어
  strict DLA gate를 통과하지 못했다.
- S5/S6는 실패를 숨기지 않고 negative control로 보존했으며 이후 strict
  causal analysis에서 자동 제외했다.

ModelOpt ONNX 경로를 위해 user environment에 추가한 package:

| Package | Version |
|---|---:|
| `onnxscript` | 0.7.1 |
| `onnx-ir` | 0.2.1 |
| `onnxconverter-common` | 1.16.0 |
| `lief` | 1.0.0 |
| `cppimport` | 26.4.17 |
| `polygraphy` | 0.50.3 |
| `pybind11` | 3.0.4 |

원 ResNet-50의 `GlobalAveragePool`은 DLA compute fallback을 만들었다.
고정 입력의 7×7 feature map에 대해 이를 `AveragePool(kernel=7×7)`로 바꾼
S11 graph variant를 만들었다. 4-image probe에서 원 ONNX와 출력 최대 차이는
0이었고, 동일 variant를 S7 GPU와 S8 DLA 양쪽에 사용했다.

- variant SHA-256:
  `e76b8e1b5d6e0a240a464e01b023660e6b0767c6b09af87d75dc31aaf19f3c45`
- S7: strict INT8 compute 통과
- S8: DLA partition 1개, compute fallback 0, strict gate 통과
- DLA에서 GPU로 배치된 항목은 Flatten, constant, broadcast shuffle 및 최종
  FP32 output reformat 같은 비-compute 보조 layer뿐이다.

### reduced 3×3 atlas 설정과 무결성

```bash
python3 chain_survival/scripts/run_pipeline_ablation.py \
  --calibrations calib_shadow_1 calib_shadow_2 calib_blind_1 \
  --builds 0 1 2 \
  --states S0 S1 S2 S3 S4 S7 S8 \
  --n-calib 200 \
  --n-images 128 \
  --image-split mechanism_discovery \
  --dla-core 0 \
  --output-root chain_survival/results/v15/pipeline_ablation
```

- 환경: 3 calibration × 3 independent build = 9
- paired image: 환경당 동일한 128장
- 선택 상태: S0/S1/S2/S3/S4/S7/S8
- 선택 matrix record: 63/63 `OK`, strict artifact gate 63/63 통과
- runner stage record: 37/37 `OK`
- 추가 S5/S6 negative record: 2개, 의도대로 strict false 및 분석 제외
- build0의 calibration cache 생성 후 build1/2는 동일 cache를 재사용했다.

각 calibration에서 세 build의 top-1 결과는 동일했다.

| Calibration | S4 CA / source consistency | S7 CA / source consistency | S8 CA / source consistency |
|---|---:|---:|---:|
| `calib_shadow_1` | 76.56% / 94.53% | 75.78% / 96.88% | 77.34% / 98.44% |
| `calib_shadow_2` | 77.34% / 92.97% | 78.12% / 96.88% | 78.12% / 97.66% |
| `calib_blind_1` | 75.78% / 93.75% | 77.34% / 95.31% | 75.78% / 97.66% |

S0/S1/S2/S3은 모든 환경에서 CA 77.34%, source consistency 100%였다.

### 상태 전이

아래 범위는 9환경에서의 mean absolute logit delta, prediction flip 및
accuracy delta다.

| Contrast | Mean absolute range | Flip range | Accuracy delta range |
|---|---:|---:|---:|
| export S0→S1 | 0.000964 | 0% | 0%p |
| FP compile S1→S2 | 3.09e-7–3.14e-7 | 0% | 0%p |
| FP16 policy S2→S3 | 0.001814–0.002033 | 0% | 0%p |
| quantization S1→S4 | 0.073145–0.082438 | 5.47–7.03% | -1.56–0%p |
| implicit Q+C GPU S1→S7 | 0.063666–0.065918 | 3.12–4.69% | -1.56–+0.78%p |
| backend S7→S8 | 0.044934–0.046348 | 2.34–3.12% | -1.56–+1.56%p |

따라서 benign ResNet-50에서는 export와 FP compilation 효과는 매우 작고,
quantization과 GPU/DLA backend 전이가 주된 logit/prediction 변화 단계다.

### build noise와 calibration stability

상태별 세 build 간 최대 global logit RMS:

| Calibration | S2 FP32 | S3 FP16 | S7 INT8 GPU | S8 INT8 DLA |
|---|---:|---:|---:|---:|
| `calib_shadow_1` | 2.15e-7 | 0.003278 | 0.020515 | 0 |
| `calib_shadow_2` | 1.10e-7 | 0.001960 | 0 | 0 |
| `calib_blind_1` | 1.03e-7 | 0.001682 | 0 | 0 |

backend residual `S8-S7`의 global RMS와 동일 calibration 내 최대 build-noise
global RMS:

| Calibration | Effect RMS floor | Build-noise RMS ceiling | Effect/noise |
|---|---:|---:|---:|
| `calib_shadow_1` | 0.075536 | 0.020515 | 3.682× |
| `calib_shadow_2` | 0.074425 | 0 | 관측 잡음 없음 |
| `calib_blind_1` | 0.075638 | 0 | 관측 잡음 없음 |

따라서 `backend effect > 3× build noise` 수치 gate는 세 calibration에서
모두 통과했다. 그러나 calibration-only pair의 signed residual cosine
최솟값은 0.0101이고 최대 residual-difference RMS는 0.10636이다. 즉 효과의
크기는 안정적이지만 어떤 image/class logit이 어느 방향으로 변하는지는
calibration에 강하게 의존한다. 이를 “최소 2 calibration에서 방향 일관”으로
해석하지 않는다.

### P2/P3 판정

- **P2 reduced atlas: 완료**
  - 3 calibration × 3 build paired record
  - selected 63 records strict 통과
  - 상태 전이와 build noise 정량화
- **P2 full gate: 진행 중**
  - S5/S6 strict explicit path는 현재 TensorRT 10.3/ModelOpt 조합에서 실패
  - S9/S10은 calibration/build 축으로 구현했고 S11은 graph variant로
    구현했지만 S12 version variant는 현재 단일 TensorRT 10.3 환경에서 불가
  - layerwise 3×3 capture와 공격 state-wise CA/ASR matrix는 아직 미완료
- **P3 magnitude criterion: 통과**
  - backend effect가 build noise의 3배 이상인 calibration 3개
- **P3 full gate: 진행 중**
  - signed residual의 calibration 간 방향 재현이 확보되지 않았고
    공격의 prediction/guard behavior와 아직 연결하지 않았다.

### 산출물

- run manifest:
  `chain_survival/results/v15/pipeline_ablation/manifest/pipeline_ablation_run.json`
- 9환경 interaction:
  `chain_survival/results/v15/pipeline_ablation/ablations/pipeline_interactions.json`
- 환경별 transition:
  `chain_survival/results/v15/pipeline_ablation/ablations/state_transitions__*.json`
- capture index:
  `chain_survival/results/v15/pipeline_ablation/captures/<calibration>/build<id>/run_index.json`

주요 SHA-256:

- run manifest:
  `bc7263b4c569b9d70d5403d8b48555c740d127520d2a45dbf72f49bb8ae9172a`
- interaction report:
  `6f66037f7c830d452b1e478fc0ed00e24306cb3efafc6032668af24ebe7543c0`

interaction report는 RMS 정의를 명시하고 0 build-noise ratio 표현을 정리한
뒤 재생성했으며, 위 SHA-256은 최종 검증 값이다.

### 다음 조치

1. P1의 `attacked_model.pth`, `trigger.pth`, `ort_final.onnx`를 available
   S0/S1/S2/S3/S4/S7/S8 reduced matrix에 연결한다.
2. CIFAR ConvNet의 TensorRT/DLA op compatibility를 preflight한 뒤 상태별
   clean accuracy, source-trigger ASR, compiled-trigger ASR 및 guard
   separation을 paired capture한다.
3. 공격 residual을 calibration/build noise와 비교해 P3 방향 재현과 P4
   survival type을 판정한다.

## 2026-07-31 01:17–05:55 KST — P4 original DcL-BD 3×3 pipeline survival

### 구현

신규 runner:

- `chain_survival/scripts/run_dclbd_survival.py`
  - P1의 static batch-100 CIFAR-10 attacked ONNX/checkpoint/trigger 재사용
  - S0–S8 build와 clean/trigger paired capture
  - TensorRT detailed inspector 및 strict artifact gate
  - 8개 guard-search 위치와 32개 channel maximum capture
  - 상태별 CA, trigger clean accuracy, ASR, source consistency
- `chain_survival/scripts/analyze_dclbd_survival.py`
  - 3 calibration × 3 build aggregate
  - 상태별 survival taxonomy
  - calibration/build noise 및 causal contrast
  - P3/P4/P5 gate 자동 판정

전체 embedding은 환경당 수 GB가 필요하므로 저장하지 않았다. 대신 공격이
실제로 사용하는 guard 위치 8개, channel maximum 32개, clean/trigger logits를
10,000장 모두 저장했다. S7 high/low engine의 전체 embedding 동일성은 별도의
100-image 직접 probe로 검증했다.

### 설정

```bash
python3 chain_survival/scripts/run_dclbd_survival.py \
  --calibrations calib_shadow_1 calib_shadow_2 calib_blind_1 \
  --builds 0 1 2 \
  --states S0 S1 S2 S3 S4 S5 S6 S7 S8 \
  --n-calib 1000 \
  --n-eval 10000 \
  --allow-gpu-fallback
```

- calibration: CIFAR-10 train에서 1,000장씩, 세 split 간 완전 disjoint
- evaluation: CIFAR-10 test 10,000장 전체
- static batch: 100
- target label: 0
- trigger: P1에서 학습한 좌상단 8×8 patch
- build: calibration cache를 같은 calibration의 세 build에서 재사용
- DLA core: 0

### artifact compatibility

| State | 9환경 artifact 결과 | Strict 판정 | 원인/비고 |
|---|---:|---|---|
| S0 PyTorch | 9/9 reference | Pass | attacked checkpoint |
| S1 ORT FP32 | 9/9 reference | Pass | P1 `ort_final.onnx` |
| S2 TRT GPU FP32 | 9/9 built | Pass | 10 FP32 compute layers |
| S3 TRT GPU FP16 | 9/9 built | Pass | mixed FP16/FP32 tactic, formal FP16 gate 통과 |
| S4 ModelOpt Q/DQ ORT | 9/9 reference | Pass | calibration별 Q/DQ 생성 성공 |
| S5 explicit INT8 GPU | 9/9 built | **Fail** | FP32 compute 7개 잔존 |
| S6 explicit INT8 DLA | 0/9 | **Fail** | guard `Greater/Where`, FC/MatMul 비지원 후 TRT shape assertion |
| S7 implicit INT8 GPU | 9/9 built | **Fail** | boolean guard 및 일부 FP32 compute |
| S8 implicit INT8 DLA hybrid | 9/9 built | **Fail** | DLA partition 7개이나 guard/FC compute GPU fallback |

S8은 실제 DLA convolution partition을 실행하지만 end-to-end strict DLA state는
아니다. 따라서 S8 결과는 **hybrid DLA/GPU survival**로만 해석한다.

### 상태별 전체 결과

모든 수치는 9환경, 각 10,000장 결과의 범위다.

| State | Clean accuracy | Trigger ASR | 선택 guard trigger fire | Survival type |
|---|---:|---:|---:|---|
| S0 | 85.24% | 12.06% | 25.0% | source leak |
| S1 | 85.26% | 99.95% | 100% | full survival/onset |
| S2 | 85.26% | 99.95% | 100% | full survival |
| S3 | 85.25–85.28% | 10.47–99.95% | 87.5–100% | build/tactic unstable |
| S4 | 84.91–85.09% | 83.62–87.58% | 37.5% | partial survival |
| S5 | 84.92–85.11% | 83.56–87.58% | 37.5% | partial, non-strict |
| S6 | — | — | — | unavailable |
| S7 | 85.11–85.27% | 18.14–84.08% | 37.5% | build/tactic unstable, non-strict |
| S8 | 84.96–85.21% | 7.11–7.31% | 0–12.5% | destroyed, hybrid-only |

S0/S1 수치는 P1의 source ASR 12.06%와 compiled ASR 99.95%를 정확히
재현했다. 모든 상태의 clean accuracy는 attacked source와 약 ±0.33%p 안에
있지만, pre-attack clean reference 89.32% 대비 약 4%p 낮다는 P1 strict
clean gate 실패는 그대로 유지된다.

### build별 ASR

| Calibration | S3 build0/1/2 | S7 build0/1/2 | S8 build0/1/2 |
|---|---:|---:|---:|
| `calib_shadow_1` | 10.47 / 10.47 / 10.47% | 83.88 / 18.15 / 83.88% | 7.12 / 7.12 / 7.12% |
| `calib_shadow_2` | 10.47 / 10.47 / 10.47% | 83.39 / 18.16 / 83.38% | 7.11 / 7.11 / 7.11% |
| `calib_blind_1` | 11.92 / 10.47 / 99.95% | 18.14 / 84.08 / 84.08% | 7.31 / 7.31 / 7.31% |

build ID가 tactic을 고정하는 것은 아니다. 동일 설정의 독립 build가 서로 다른
engine tactic/precision mix를 선택하며 공격이 high-ASR/low-ASR mode로
전환된다.

### root-cause decomposition

#### 1. export 및 FP32 compilation

- S0→S1: ASR +87.89%p
- S1→S2: ASR 변화 0

공격 onset은 P1과 동일하게 export/ORT compiled graph에서 발생하며 TRT FP32
compile에서도 완전히 유지된다.

#### 2. quantization

S1→S4 ASR 효과:

| Calibration | S1 ASR | S4 ASR | Effect | S4 build ASR range |
|---|---:|---:|---:|---:|
| `calib_shadow_1` | 99.95% | 86.34% | -13.61%p | 0 |
| `calib_shadow_2` | 99.95% | 83.62% | -16.33%p | 0 |
| `calib_blind_1` | 99.95% | 87.58% | -12.37%p | 0 |

방향은 세 calibration에서 일관되고, 관측 build noise가 0이므로
`intervention effect > 3× build noise` gate를 모두 통과한다. 공격 연결 P3의
안정적인 dominant stage는 **quantization attenuation**으로 판정한다.

#### 3. S7 tail tactic instability

세 calibration의 S7 high/low build 비교:

- 저장된 8개 clean/trigger guard input: 모두 bit-identical
- 저장된 32개 channel maximum: 모두 bit-identical
- 100-image 전체 embedding 직접 probe: clean/trigger 모두 bit-identical,
  RMS 0
- 동일 probe trigger logits: RMS 2.849
- 10,000장 trigger logit build RMS: 약 2.80–2.83
- low-ASR target margin mean: 약 -4.57~-4.62
- high-ASR target margin mean: 약 +5.14~+5.29

따라서 S7의 18%↔84% 변동은 feature 또는 guard 분리가 아니라,
**guard 이후 tail tactic**이 동일 embedding을 다른 target margin으로
변환해서 발생한다.

#### 4. FP16 tail instability

S3은 선택 guard trigger fire가 87.5–100%인데도 대부분의 build에서 ASR
10.47–11.92%이고 한 build에서는 99.95%다. 즉 guard activation은 공격
생존의 필요조건일 수 있지만 충분조건이 아니며, FP16 tail tactic이 별도의
생존 경계를 만든다.

#### 5. hybrid DLA destruction

S7→S8에서:

- 선택 guard trigger fire: 37.5% → 0–12.5%
- ASR: 18.14–84.08% → 7.11–7.31%

hybrid DLA feature path에서 guard separation이 무너지고 공격 ASR도 source
leak보다 낮아진다. P4가 요구한 “어느 stage에서 guard separation이
무너지는가”는 S8 hybrid DLA transition으로 특정했다.

### P3/P4/P5 판정

- **P3 attack-linked root cause: GO**
  - quantization attenuation이 세 calibration에서 같은 방향
  - effect가 build noise의 3배 이상
  - prediction/ASR 변화와 직접 연결
- **P4 available-state survival taxonomy: GO**
  - S0/S1/S2: full FP32 survival
  - S3: FP16 tactic-contingent survival/destruction
  - S4/S5/S7: partial 또는 build-contingent INT8 survival
  - S8: hybrid DLA destruction
- **P4 strict DLA generalization: 미완료**
  - 원 DcL-BD graph의 guard/FC가 DLA 비지원이므로 strict S6/S8 부재
- **P5 entry: NO-GO**
  - S3/S7 interaction이 independent build에서 안정하지 않음
  - stable non-saturating interaction 후보가 없으므로 direct interaction
    attack 학습을 시작하지 않는다.
- **P6/P7: 미진입**
  - P5 stable interaction gate 미달 상태에서 factorized path attack이나
    capability ladder로 넘어가면 사전 고정 조건을 위반한다.

### 산출물과 hash

- calibration registry:
  `chain_survival/results/v15/dclbd_survival/calibration_registry.json`
  - SHA-256:
    `891f978b1532c0a36062cbca99c627767b7b2d6fa693bfdc932007093da63c6c`
- run manifest:
  `chain_survival/results/v15/dclbd_survival/run_manifest.json`
  - SHA-256:
    `f5e9b217684669cd4b3c121aebdaef3b094ea9a739e5aa316bf9edbc9eb8dab6`
- survival analysis:
  `chain_survival/results/v15/dclbd_survival/analysis/survival_summary.json`
  - SHA-256:
    `e86bf7268ec841270c9015b7411ea0c722f27b048c4727a794f5bdb24e0e0ac5`
- runner:
  `chain_survival/scripts/run_dclbd_survival.py`
  - SHA-256:
    `934c42078dbe8aa26af8aa2c8c12581d05505400bf0cd6ef1cacf7b651dd84c3`
- analyzer:
  `chain_survival/scripts/analyze_dclbd_survival.py`
  - SHA-256:
    `31a1ec2e9a07000f960c0afc369c2fe14748bb1e4da56ac1a2dd7e7cac536b7b`

### 다음 조치

P5/P6는 진입조건 미달로 중단하고, 기존 P4 capture를 이용한 P9 multi-state
differential audit와 randomized build defense의 탐지율/오탐률을 평가한다.

## 2026-07-31 05:56–06:04 KST — P9 differential audit와 build randomization

### 구현과 protocol

신규:

- `chain_survival/scripts/evaluate_v15_audit_defense.py`
- `chain_survival/scripts/benchmark_v15_audit_latency.py`

각 state pair의 score는 image별 centered-logit RMS다. 여러 pair를 쓰는
audit은 shadow clean의 pair별 99 percentile로 정규화한 뒤 maximum을
사용한다.

- threshold training: `calib_shadow_1`, `calib_shadow_2` clean
- threshold target FPR: 1%
- blind evaluation: `calib_blind_1`
- build: 0/1/2를 각각 독립 평가
- positive: triggered image
- negative: clean image

이는 triggered sample로 threshold를 조정하지 않는 blind protocol이다.

### blind detection

| Audit | State 수 | Blind FPR range | Blind TPR range | Blind AUC range |
|---|---:|---:|---:|---:|
| S0–S1 source/export | 2 | 0.99% | 100% | 1.000 |
| S1–S3 export/FP16 | 2 | 0.73–2.35% | 9.81–100% | 0.6703–1.000 |
| S1–S4 export/QDQ | 2 | 0.72% | 100% | 1.000 |
| S7–S8 GPU/hybrid-DLA | 2 | 0.04–0.37% | 99.11–100% | 0.99996–1.000 |
| S0/S1/S4 minimal strict | 3 | 0.96% | 100% | 1.000 |
| S0–S4 strict pairwise max | 5 | 0.81–1.44% | 100% | 1.000 |
| available 8-state max | 8 | 1.29–2.36% | 100% | 1.000 |

사전 선택 규칙은 “모든 build의 blind FPR ≤1%를 만족하면서 worst-build
TPR 최대, 동률이면 state 수 최소”다. 이 규칙은 **S0–S1 dual execution**을
선택했다.

상태를 많이 추가하면 clean maximum score가 증가해 blind FPR이 오히려
목표를 초과했다. FP16 pair는 P4에서 확인한 tactic instability 때문에
탐지율도 9.81–100%로 불안정했다.

### randomized build defense

S7 engine을 세 independent build 중 균등 무작위로 선택할 때:

| Calibration | Expected ASR | Best-build ASR | Worst-build ASR | Expected CA |
|---|---:|---:|---:|---:|
| `calib_shadow_1` | 61.97% | 18.15% | 83.88% | 85.23% |
| `calib_shadow_2` | 61.64% | 18.16% | 83.39% | 85.19% |
| `calib_blind_1` | 62.10% | 18.14% | 84.08% | 85.15% |

build randomization은 공격을 불안정하게 만들지만 expected ASR ≤10% gate를
전혀 만족하지 못한다. 따라서 **단독 방어로는 NO-GO**다.

### 실제 latency

batch 100 고정 입력, preprocessing 제외, warm-up 5회, 30회 반복,
CUDA synchronization을 포함한 sequential wall-clock latency:

| Configuration | Mean latency |
|---|---:|
| S0 source only | 0.0843 ms/image |
| S0–S1 selected audit | 0.5552 ms/image |
| S1–S4 audit | 1.0154–1.0503 ms/image |
| S7–S8 hybrid audit | 0.1871–0.2276 ms/image |
| S0/S1/S4 three-state audit | 1.0996–1.1345 ms/image |

선택된 S0–S1 audit은 source 단독 대비 약 6.59× sequential latency다. S7–S8
audit은 더 빠르지만 두 state가 strict INT8/DLA gate를 통과하지 못했으므로
primary defense로 선택하지 않는다.

`tegrastats` 100ms sampling으로 GPU/SOC와 CPU/CV partial rail을 같은
프로세스의 idle→source→idle→audit phase에서 측정했다.

| Configuration | Gross partial-rail energy | Idle-subtracted energy |
|---|---:|---:|
| S0 source only | 0.945 mJ/image | 0.315 mJ/image |
| S0–S1 audit | 7.437 mJ/image | 2.716 mJ/image |

audit의 idle-subtracted 증분은 약 2.401 mJ/image다. 이 값은
`VDD_GPU_SOC + VDD_CPU_CV`만 합산한 **partial platform energy**이며 전체
보드 입력 전력을 의미하지 않는다.

### P9 판정

- multi-state differential audit: **GO**
  - blind AUC 1.0
  - blind TPR 100%
  - blind FPR 0.99%
  - 세 build에서 동일
- randomized calibration/build 단독: **NO-GO**
  - S7 expected ASR 약 62%
- provenance manifest: P0/P2/P4에서 source, Q/DQ, cache, engine, inspector
  hash를 모두 기록해 구현 완료
- boundary-prioritized audit: 이번 공격의 최소 관측 경계는 S0–S1로 확정

### 산출물과 hash

- detection report:
  `chain_survival/results/v15/defense/multi_state_audit.json`
  - SHA-256:
    `69fbe62072ac54c0dc3c77c1bd2e0659639d033f221b2a952195398dbe190d27`
- latency report:
  `chain_survival/results/v15/defense/audit_latency.json`
  - SHA-256:
    `834de1eed3226715360ced316ef5a7e705212df286df97a72b62c3863ef85627`
- evaluator:
  `chain_survival/scripts/evaluate_v15_audit_defense.py`
  - SHA-256:
    `74ffef560b8ac9f3910268e8e670925d5fef2ba1c3a4b9f9206909bb5a6c83c9`
- latency benchmark:
  `chain_survival/scripts/benchmark_v15_audit_latency.py`
  - SHA-256:
    `9824c9de9dc184aa6bc224ae8c16906cc4db8b7376882fa7f96e0879b309ec39`
- power report:
  `chain_survival/results/v15/defense/audit_power.json`
  - SHA-256:
    `7c8476fa387b8347a92825136c32509bbdf7190309cf0cf721cc61e05e1092b6`
- power measurement runner:
  `chain_survival/scripts/measure_v15_audit_power.py`
  - SHA-256:
    `822a050accc0b3c90785cc9bcd472e19aa69af128f47fca385cd14554c55932a`

## 2026-07-31 06:04–06:07 KST — DLA core 강건성, P8 감사 및 최종 종료

### DLA0/DLA1 강건성

P4의 blind calibration에서 동일 S8 hybrid DLA engine을 DLA core 1로 다시
build하고 CIFAR-10 test 10,000장을 paired capture했다.

```bash
python3 chain_survival/scripts/run_dclbd_survival.py \
  --output-root chain_survival/results/v15/dclbd_survival_dla1 \
  --calibrations calib_blind_1 \
  --builds 0 \
  --states S0 S8 \
  --n-calib 1000 \
  --n-eval 10000 \
  --dla-core 1 \
  --allow-gpu-fallback
```

| DLA core | S8 CA | S8 ASR | selected guard fire | Strict gate |
|---:|---:|---:|---:|---|
| 0 | 84.96% | 7.31% | 12.5% | Fail, hybrid |
| 1 | 84.96% | 7.31% | 12.5% | Fail, hybrid |

DLA0와 DLA1의 10,000-image capture는 NPZ SHA-256까지 동일하다. 즉 clean 및
trigger logits, 8개 선택 guard 좌표, 32개 channel maximum, label이 모두
bit-identical이다. S8에서 관측한 guard separation 붕괴와 낮은 ASR은 이
장치의 DLA core 선택에는 의존하지 않는다. 단, 두 결과 모두 7개 DLA
partition과 GPU compute fallback을 포함하므로 strict-DLA 결과로 승격하지
않는다.

주요 artifact:

- DLA1 calibration registry:
  `chain_survival/results/v15/dclbd_survival_dla1/calibration_registry.json`
  - SHA-256:
    `75d2772a3b14e82725699a5f0f86666d618346e5d1f1213e82d9132bc73989dd`
- DLA1 run manifest:
  `chain_survival/results/v15/dclbd_survival_dla1/run_manifest.json`
  - SHA-256:
    `93dd9b805f882de6e60be48ed18605a9e1496e1d30764418bf0398b3d3f4fdf2`
- DLA1 state index:
  `chain_survival/results/v15/dclbd_survival_dla1/states/calib_blind_1/run_index.json`
  - SHA-256:
    `3050cf3d981e3fdcd3f79de6d3014c136bc8d8a2561c0a008e923f9b3eb27a36`
- DLA1 capture index:
  `chain_survival/results/v15/dclbd_survival_dla1/captures/calib_blind_1/build0/run_index.json`
  - SHA-256:
    `261f646b2ce0e2a9de7501a890d89c3f27c9a406c12cf73720cbc8474a771791`
- DLA0/DLA1 S8 capture:
  - SHA-256:
    `9c4450da969b413bd735d7f3ed3478fbd360c6f69b616941aeacd7ab4e0d6ef6`

### P8 cross-vendor 실행 가능성 감사

2026-07-31 06:07 KST 현재:

- TensorRT Python/runtime와 `libnvinfer`: 10.3.0/major 10만 설치
- TensorRT 11.x: 미설치
- Mobilint CLI: v1.2.0 설치
- Mobilint qb Compiler/MXQ compiler: 미설치
- Mobilint NPU: `mobilint-cli devices` 결과 0개
- QNN/SNPE command, library 및 device: 없음
- Jetson DLA device: core 0/1 존재하며 위 강건성 평가 완료

따라서 P8 우선순위에 있는 세 additional toolchain 중 실제 model을
compile/run할 수 있는 조합이 없다. P8 gate인 “최소 1 additional
toolchain에서 state model 재현”은 **환경 제약으로 NOT ASSESSED**다.
이는 음성 결과가 아니라 외부 compiler/hardware 부재에 따른 실행 불가
판정이며, TensorRT 10.3 결과로 cross-vendor 일반화를 주장하지 않는다.

### P0–P9 최종 상태

| Phase | 최종 상태 | 종료 근거 |
|---|---|---|
| P0 | **GO / 완료** | artifact lineage, strict inspector, GPU/DLA 환경 고정 |
| P1 | **조건부 재현 / 완료** | compiled ASR 99.95% 재현; clean/ASR strict reference gate는 미달 |
| P2 | **reduced GO, full NO-GO / 종료** | 3 calibration × 3 build 63/63 strict selected records; S5/S6 explicit strict path와 S12 version variant 불가 |
| P3 | **GO / 완료** | S1→S4 공격 ASR 감소가 3 calibration에서 일관되고 build noise의 3배 초과 |
| P4 | **available-state GO / 완료** | survival taxonomy와 S8 guard-collapse stage 특정; 원 graph strict DLA는 backend incompatibility |
| P5 | **NO-GO / 미진입 종료** | independent build에서 stable non-saturating interaction 부재 |
| P6 | **미진입 종료** | P5 및 unseen-calibration path-fingerprint 진입조건 미충족 |
| P7 | **미진입 종료** | P5/P6 성공을 전제로 한 capability ladder 진입조건 미충족 |
| P8 | **NOT ASSESSED / 외부 제약 종료** | TRT11, Mobilint compiler/NPU, QNN 실행 조합 부재 |
| P9 | **audit GO / 완료** | S0–S1 blind TPR 100%, FPR 0.99%, AUC 1.0; randomization 단독 NO-GO |

### 최종 연구 결론

현재 환경에서 실행 가능한 gate-driven v15 실험은 모두 종료했다.

1. 원 DcL-BD 공격은 source PyTorch에서는 ASR 12.06%이지만 ONNX export
   이후 99.95%로 활성화되며 FP32 TensorRT에서도 유지된다.
2. INT8 Q/DQ는 ASR을 12.37–16.33%p 감소시키고, 이 방향은 세 calibration과
   세 build에서 안정적이다.
3. FP16 및 implicit INT8 GPU에서는 guard 이후 tail tactic이 동일
   embedding을 서로 다른 target margin으로 변환해 ASR을 크게 흔든다.
4. hybrid DLA 전이에서는 guard separation 자체가 무너지고 ASR이
   7.11–7.31%까지 감소한다. 이 현상은 DLA core 0/1에서 bit-identical하다.
5. 가장 작은 유효 방어는 S0–S1 dual execution이다. 공격 탐지는 완벽했지만
   source 단독 대비 sequential latency는 6.59배, 측정한 partial rail의
   idle-subtracted 증분 energy는 약 2.401 mJ/image다.
6. build randomization만으로는 expected ASR이 약 62%여서 충분한 방어가
   아니다.

P5–P7은 실패한 진입조건을 무시하고 학습을 계속하지 않는다는 계획서 규칙에
따라 종료했다. P8은 추가 compiler와 실제 vendor device가 제공될 때만
재개할 수 있는 외부 의존 항목으로 분리한다.

## 2026-07-31 10:55 KST — 후속 trigger margin 가능성 분석

### 질문

전체 pipeline에서 안정적인 범용 공격은 실패했지만, 현재 결과 중 후속
trigger로 발전시킬 수 있는 가장 강한 경계가 무엇인지 확인했다.

### 방법

기존 P4의 10,000-image logits에서 target class 0의 margin
`target logit - max(other logits)`을 계산했다. 그런 다음 모든 state에
동일한 target-logit 감소량을 가정하고 source S0의 triggered ASR을 10%,
5%, 1%로 제한했을 때 다른 state에 남는 ASR과 clean accuracy를 계산했다.

이는 기존 logits에 대한 **post-hoc feasibility proxy**다. 실제 checkpoint의
target-class bias를 수정하고 ONNX/engine을 재생성한 결과는 아니므로, 아래
수치를 최종 공격 성능으로 사용하지 않는다.

### 결과

| S0 ASR 목표 | 필요한 target-logit 감소 | S1/S2 ASR | S4/S5 ASR range | S7 ASR range | S8 ASR range | S0 clean accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 10% | 1.223 | 99.89% | 77.08–82.22% | 14.41–78.12% | 5.53–5.71% | 84.79% |
| 5% | 4.888 | 99.16% | 51.28–58.62% | 7.77–52.37% | 1.71–1.89% | 81.65% |
| 1% | 8.828 | 95.28% | 23.66–28.75% | 2.72–24.42% | 0.12–0.13% | 78.26% |

S0 ASR 10% 지점에서는 현재 S0 clean accuracy가 85.24%에서 84.79%로
0.45%p만 감소하면서 S1/S2 ASR 99.89%가 유지된다. 반면 S0 ASR을 5%
이하로 강하게 낮추면 clean accuracy 손실이 약 3.6%p로 증가하고 INT8
생존성도 크게 떨어진다.

### 판정

현재 증거가 가장 강한 후보는 **S0→S1 export-activated trigger**다.

- S0에서는 ASR을 10% 이하로 억제할 여지가 있음
- S1 ONNX와 S2 TensorRT FP32에서는 약 99.9% ASR 유지 가능성이 큼
- calibration이나 TensorRT build에 의존하지 않는 가장 안정적인 경계
- S4/S5 Q/DQ에서는 부분 생존하지만 90% 기준에는 미달
- S3/S7 tactic-trigger와 S8 DLA-trigger는 변동성 또는 신호 붕괴 때문에
  주 후보로 사용하지 않음

다음 공격 실험은 새로운 patch 형태를 탐색하기보다 기존 8×8 trigger를
고정하고 target-tail margin을 재조정하는 작은 확인 실험이 우선이다.
실제 checkpoint bias 수정 후 S0/S1/S2/S4를 재-export/build하여 위 proxy가
유지되는지 확인하고, `S0 ASR ≤10%`, `S1/S2 ASR ≥95%`, clean accuracy
추가 손실 `≤1%p`를 모두 통과할 때만 정식 후속 실험으로 확장한다.
