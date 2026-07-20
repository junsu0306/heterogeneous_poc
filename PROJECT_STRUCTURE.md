# 프로젝트 구조 (v9 리셋)

> 2026-07-20 전면 리셋. 구 phase0~6 결과물(engines/logs/results/REPORT/checkpoints)과
> 구버전 계획서(v5, 구 PROJECT_STRUCTURE)는 모두 삭제했다. 과거 실측 근거의 요약은
> Claude 메모리(`project_dpcb_status` 등)에만 남아있다.
> **데이터셋과 핵심 재사용 코드만 보존**하고 `common/` 아래로 재배치한 뒤 처음부터 다시 시작한다.
>
> 계획 문서: `academic_research_plan_v9.md` (연구설계) / `chain_survival_experiment_handoff.md` (P0.5 실행절차)
> 모든 스크립트는 **repo root(`poc/`)에서** 실행한다 (예: `python3 common/emulator/quant_sim.py`).
> `common/scripts/trt_runtime.py`는 `sys.path.insert(0, "common/scripts")` 기준으로 import되므로 root 실행이 전제.

```
poc/
├── academic_research_plan_v9.md          연구계획 (v9 — 변환체인 강건성, P0.5 최우선)
├── chain_survival_experiment_handoff.md  P0.5 변환체인 생존 실험 핸드오프 (다음 작업)
├── PROJECT_STRUCTURE.md                  이 문서
├── docs/                                 NVDLA/TensorRT 참고 원문 (DLA_MECHANISM.md, hwarch, unit_description, tempus_core)
│
├── common/                               전 phase 공용 (보존 자산)
│   ├── ENV_PIN.md                        환경 버전 고정 메모
│   ├── scripts/                          TensorRT 공용 인프라
│   │   ├── trt_runtime.py                엔진 빌드/실행 (torch CUDA 텐서 I/O, pycuda 불필요) — 핵심
│   │   ├── export_resnet50.py
│   │   ├── export_yolov8s_backbone_neck.py
│   │   └── simplify_yolov8s_backbone_neck.py
│   ├── models/                           공유 모델 원본 (resnet50.onnx, yolov8s.pt, yolov8s_backbone_neck_sim.onnx)
│   ├── emulator/                         STE fake-quant emulator + 헬퍼 (P2/공격용 재사용)
│   │   ├── quant_sim.py                  STE 양자화 시뮬레이터 (Step3.8 캘리브레이션 수정 반영본)
│   │   ├── dual_path_model.py            Qv/Qd 이중경로 모델
│   │   ├── trigger.py                    패치 트리거 (CIFAR 4x4 / ImageNet 8x8)
│   │   ├── model_r18_cifar.py            ResNet-18/CIFAR-10 (전 과정 Jetson 로컬 축소본)
│   │   ├── cifar_data.py                 CIFAR-10 로더
│   │   ├── prepare_cifar10.py            parquet→npy 디코드 (DATA_DIR 경로는 재배치로 조정 필요)
│   │   ├── build_r18_cifar_engines.py    실엔진 빌드 (GPU-INT8/DLA-INT8)
│   │   └── verify_real.py                실엔진 사후 검증
│   ├── datasets/
│   │   └── cifar10/                      CIFAR-10 (parquet + 디코드된 npy)
│   └── external/                         업스트림 레포 + 다운로드 자산 (재다운로드 비쌈, 보존)
│       ├── Qu-ANTI-zation/               QCB 공식 clone + 사전학습 체크포인트(~14GB) + Tiny-ImageNet-200
│       ├── BackdoorBench/                방어 5종 (C2/방어평가용)
│       └── DLCompilerAttack/             DcL-BD(S&P'26) 원논문 clone (§6 대조군)
│
└── chain_survival/                       ★ P0.5 실험 워크스페이스 (다음 착수 — 아직 비어있음)
    ├── models/  onnx/  engines/  results/  logs/
    └── (핸드오프 §2~7에 따라 export_models.py / run_paths.py / analyze_*.py 작성 예정)
```

## 현재 상태 / 다음 단계

- **v9 최우선 게이트 = P0.5 (변환 체인 생존)**: 공격 코드를 만들기 **전에**, 정상(백도어 없는)
  torchvision 모델(resnet50/efficientnet_b0/mobilenet_v3)로 "NPU 실행에만 고유하고
  원본→ONNX→TensorRT→NPU 체인을 통과해 살아남는 수치 편차"가 존재하는지 측정.
  절차는 `chain_survival_experiment_handoff.md`.
- P0.5는 ImageNet val 500장(calibration용)이 필요하나 현재 미보유 → 실험 착수 시 확보 필요.
- 재사용 판단: `common/scripts/trt_runtime.py`(엔진 I/O)와 `common/emulator/*`는 그대로 재사용.
  단 emulator가 실HW 편차를 2~4배 과대추정한 이력(구 Phase2)이 있으므로 P2 정합성은 재검증 대상.
```
