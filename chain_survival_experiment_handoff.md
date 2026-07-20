# 변환 체인 생존 실험 — Claude Code 핸드오프 (NPU 실행 환경)

> **이 문서의 독자**: Jetson Orin(또는 Mobilint NPU 호스트) 위에서 직접 코드를 실행하는 Claude Code.
> **목적**: 우리 백도어 공격을 설계하기 **전에**, 그 공격이 성립할 수 있는 물리적 전제 — "NPU 실행에만 고유하고, 원본→ONNX→TensorRT→NPU 변환 체인을 통과해 살아남는 수치 편차가 실제로 존재하는가" — 를 정상(백도어 없는) 모델로 검증한다. 이것이 학술 계획서(`academic_research_plan_v9.md`) §5의 **P0.5 게이트(★★★ 최우선)**다.
> **핵심 규칙**: 이 실험은 공격 코드를 전혀 만들지 않는다. 정상 모델만 쓴다. 여기서 편차가 체인을 통과하지 못하면 공격 설계 전체가 무의미해지므로, 결과를 정직하게 보고하고 절대 긍정적으로 해석하지 말 것.

---

## 0. 배경 (왜 이 실험이 필요한가)

실제 온디바이스 배포 파이프라인:
```
원본(FP32, PyTorch) → ONNX → [TensorRT / 자체 변환 + 양자화] → NPU 바이너리(DLA/Mobilint)
```
우리 공격은 이 체인의 **모든 중간 경로(ONNX-CPU, ONNX-GPU, TensorRT-GPU)에서 트리거에 무반응**하고 **최종 NPU에서만 발현**해야 한다. 이게 가능하려면, 먼저 "NPU 경로에만 나타나고 앞선 변환들에 강건한 수치 편차"가 물리적으로 존재해야 한다. 이 문서는 그 존재 여부를 측정한다.

두 개의 하위 질문:
- **Q1 (편차 국소성)**: NPU 실행 결과가 다른 모든 경로(ONNX-CPU/GPU, TRT-GPU)와 다른가? 그 편차가 NPU에만 국소적인가, 아니면 이미 ONNX/TRT 단계에서 나타나는가?
- **Q2 (편차 강건성)**: NPU-고유 편차가 변환 체인의 weight 재계산(BN folding 등)에도 불구하고 안정적으로 재현되는가?

---

## 1. 환경 준비

```bash
# 작업 디렉토리
mkdir -p chain_survival/{models,onnx,engines,results,logs}
cd chain_survival

# 확인: 사용 가능한 런타임
python -c "import torch; print('torch', torch.__version__)"
python -c "import onnxruntime as ort; print('ort', ort.__version__, ort.get_available_providers())"
trtexec --version 2>&1 | head -1
# Mobilint SDK가 있으면 그 CLI/파이썬 API도 확인 (호스트에 설치된 것 확인)
```

필요 라이브러리: `torch`, `torchvision`, `onnx`, `onnxruntime-gpu`, `numpy`. TensorRT는 JetPack에 포함. Mobilint는 벤더 SDK.

---

## 2. 실험 대상 모델 (정상 모델만)

편차의 아키텍처 의존성을 보기 위해 세 종류. **백도어 없음, torchvision pretrained 그대로.**

```python
# export_models.py
import torch, torchvision.models as M
models = {
    "resnet50":       M.resnet50(weights="IMAGENET1K_V2"),
    "efficientnet_b0":M.efficientnet_b0(weights="IMAGENET1K_V1"),
    "mobilenet_v3":   M.mobilenet_v3_large(weights="IMAGENET1K_V1"),
}
dummy = torch.randn(1,3,224,224)
for name, m in models.items():
    m.eval()
    torch.onnx.export(m, dummy, f"onnx/{name}.onnx",
                      opset_version=17, do_constant_folding=True,
                      input_names=["input"], output_names=["logits"])
    torch.save(m.state_dict(), f"models/{name}.pth")
```

> `do_constant_folding=True`는 실제 배포와 동일하게 두어야 한다 — 이 folding이 바로 우리가 강건해야 할 변환의 일부다.

---

## 3. 경로별 실행 파이프라인 구축

각 모델을 **동일 입력**에 대해 아래 경로 전부로 실행하고 레이어별/최종 출력을 수집한다. 입력은 고정 시드 랜덤 텐서 + ImageNet 검증셋 일부(실제 분포 확인용).

### 3.1 경로 목록
| 경로 ID | 설명 | 도구 |
|---|---|---|
| `pt_fp32` | 원본 PyTorch FP32 (기준 reference) | torch |
| `onnx_cpu_fp32` | ONNX Runtime, CPUExecutionProvider | onnxruntime |
| `onnx_gpu_fp32` | ONNX Runtime, CUDAExecutionProvider | onnxruntime |
| `trt_gpu_fp16` | TensorRT GPU, FP16 | trtexec/API |
| `trt_gpu_int8` | TensorRT GPU, INT8 (implicit calibrator) | trtexec/API |
| `dla_int8` | TensorRT DLA, INT8 | trtexec `--useDLACore=0` |
| `mobilint_int8` | Mobilint NPU, INT8 (SDK) | Mobilint SDK |

> INT8 경로는 동일 calibration set(ImageNet val 500장 고정)을 쓴다. 경로 간 calibration 데이터가 같아야 "하드웨어 차이"만 분리된다.

### 3.2 실행 및 수집
```python
# run_paths.py — 의사코드
# 각 경로에 대해 동일 입력 배치를 실행, 최종 logits와 (가능하면) 레이어별 중간 텐서 수집
# 레이어별 추출: TensorRT는 markDebug/IDebugListener, ONNX는 중간 노드를 output으로 지정
# 결과를 results/{model}_{path}.npz 에 저장 (logits + 가능시 intermediate)
for model in models:
    for path in PATHS:
        out = run(model, path, fixed_input)      # (N, num_classes) logits
        np.savez(f"results/{model}_{path}.npz", logits=out, ...)
```

**주의(정직)**: 레이어별 중간 텐서 추출은 경로마다 난이도가 다르다. TensorRT markDebug는 fusion을 깨뜨릴 수 있고(관측 자체가 대상을 바꿈), Mobilint는 중간 텐서 접근이 제한적일 수 있다. **최소한 최종 logits는 모든 경로에서 반드시 수집**하고, 중간 텐서는 가능한 경로에서만 수집해 보조 근거로 쓴다.

---

## 4. 측정 (Q1: 편차 국소성)

`pt_fp32`를 기준으로 각 경로의 편차를 계산한다. 모든 계산은 float64.

```python
# analyze_locality.py — 의사코드
ref = load("results/{model}_pt_fp32.npz")["logits"].astype("float64")
for path in PATHS:
    cur = load(f"results/{model}_{path}.npz")["logits"].astype("float64")
    max_abs = np.abs(cur-ref).max()
    mean_abs= np.abs(cur-ref).mean()
    # 예측 라벨 변화율 (decision-level)
    flip = (cur.argmax(1) != ref.argmax(1)).mean()
    record(model, path, max_abs, mean_abs, flip)
```

**판정 기준 (Q1)**:
- **NPU-국소성 성립 조건**: `dla_int8`(또는 `mobilint_int8`)의 편차가 `trt_gpu_int8`보다 **유의미하게 크다.** 즉 같은 INT8이라도 NPU 경로가 GPU 경로와 다른 값을 낸다.
- 만약 `trt_gpu_int8`와 `dla_int8`의 편차가 비슷하면 → 편차가 "INT8 자체"에서 오는 것이지 "NPU 실행"에서 오는 게 아님 → 우리 실행-특성 조건화(α1)의 전제가 약함. **정직하게 보고.**
- decision-level flip이 어느 경로에서 처음 나타나는지 기록 — NPU에서만 flip이 크면 유리, 앞 경로에서도 크면 스텔스 위험.

---

## 5. 측정 (Q2: 편차 강건성)

NPU-고유 편차가 변환 체인의 weight 재계산에도 안정적으로 재현되는가.

### 5.1 변환이 weight를 실제로 얼마나 바꾸는지
```python
# analyze_weight_drift.py — 의사코드
# 원본 .pth의 conv weight와, ONNX(folding 후)에서 추출한 대응 weight를 비교
# BN folding으로 재계산된 레이어를 식별하고, 그 drift 크기 기록
```
- BN folding이 일어난 레이어 목록과 각 weight drift(max/mean)를 `results/{model}_weight_drift.json`에 기록.
- **이게 "우리가 강건해야 할 변환"의 정량적 크기**다.

### 5.2 편차의 재현성
- 동일 모델을 **여러 번** ONNX export → TRT build → NPU 실행 (빌드 비결정성 확인).
- systolic array는 결정적이므로 같은 엔진은 같은 결과여야 하나, 빌드 단계(kernel selection 등)의 비결정성이 있을 수 있음.
- NPU 경로 편차가 재빌드 간 안정적이면 → "공격 신호를 실을 안정적 채널이 있다"는 뜻(유리).

### 5.3 MAC depth / 레이어 속성 상관
- §3.3 가설: 편차는 MAC depth 큰 레이어, fusion 경계 많은 레이어에 집중.
- 레이어별 중간 텐서를 수집한 경로에서, (레이어 MAC depth, 편차 크기) 상관 산출.
- 상관이 확인되면 → α2(변환-불변 인코딩)를 "MAC depth 큰 레이어의 outlier 존재성"에 걸 수 있다는 근거.

---

## 6. 게이트 판정 (P0.5)

아래를 `results/p0_5_verdict.json`에 종합하고 판정:

**통과 (공격 설계 진행 가능)**:
- Q1: NPU 경로 편차가 GPU-INT8 경로보다 유의미하게 큼 (실행-특성 국소성 확인)
- Q2: NPU 편차가 재빌드 간 안정적이고, weight drift에도 불구하고 재현됨
- (보너스) 편차가 특정 레이어 속성(MAC depth 등)과 상관 → 인코딩 채널 존재

**부분 통과 (공격 설계 수정 필요)**:
- Q1은 성립하나 Q2 불안정 → 안정적인 레이어만 골라 신호 인코딩(α2 범위 축소)
- 편차가 있으나 중간 경로(TRT-GPU)에서도 상당 → 다중경로 dormancy(α3) 난이도 상승, guard-bias 필수

**실패 (위협모델 재검토)**:
- NPU 편차 ≈ GPU-INT8 편차 → "NPU 실행 특성"이라는 전제 자체가 약함
- 편차가 재빌드마다 무작위 → 안정적 인코딩 채널 없음
- → 이 경우 **공격 설계 중단**, 위협모델을 "개발자가 NPU 바이너리 직전까지만 검증"으로 좁히거나, 편차 대신 다른 조건화 축 모색. 학술 문서 §8 리스크로 에스컬레이션.

---

## 7. 보고 형식

각 측정 후 즉시 다음을 보고 (자동으로 다음 단계 진행 금지):
1. Q1 결과표: 모델 × 경로 × (max_abs, mean_abs, flip%)
2. Q2 결과: weight drift 크기, 재빌드 간 편차 안정성, MAC depth 상관
3. 게이트 판정: 통과/부분통과/실패 + 근거
4. 예상과 다른 결과(특히 NPU 편차가 GPU-INT8과 비슷하게 나오는 경우)는 **긍정 해석하지 말고 그대로 보고**

---

## 8. 다음 단계 (P0.5 통과 시에만)

P0.5 통과가 확인되면, 학술 계획서 §4의 공격 설계(다중경로 2단계 + 3α)로 진행. 그 전까지는 공격 코드를 만들지 않는다. P0.5는 "공격이 물리적으로 가능한 세계인가"를 확인하는 관문이며, 여기서 얻은 (편차가 큰 레이어, 안정적 채널, weight drift 크기)가 §4 설계의 입력이 된다.

## 부록: 이 실험이 답하지 않는 것 (범위 명확화)
- 이 실험은 **공격 성공을 보이지 않는다.** 오직 "공격이 성립할 물리적 전제"만 확인한다.
- 방어·완화(C2/C3)는 다루지 않는다.
- Mobilint 경로는 SDK 접근성에 따라 부분적일 수 있다 — DLA를 주 경로로, Mobilint를 교차검증(§5 P4.5)으로 둔다.
