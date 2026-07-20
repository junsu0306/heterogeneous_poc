# Phase 0-A ~ D 환경 고정 (§14 지침에 따라 코드 작성 전 확정)

확인일: 2026-07-02

| 항목 | 값 | 확인 명령 |
|---|---|---|
| JetPack (L4T) | R36.5.0 (JetPack 6.2 계열) | `cat /etc/nv_tegra_release` |
| TensorRT | **10.3.0** | `python3 -c "import tensorrt; print(tensorrt.__version__)"` |
| trtexec 경로 | `/usr/src/tensorrt/bin/trtexec` | `find / -name trtexec` |
| PyTorch | 2.11.0 (CUDA available) | `python3 -c "import torch; print(torch.__version__)"` |
| torchvision | 0.26.0 | `python3 -c "import torchvision; print(torchvision.__version__)"` |
| CUDA driver | 540.5.0 / CUDA 12.6 | `nvidia-smi` |

**결정 (§15 항목 7)**: 본 프로젝트 전 구간(0A~D)에서 TensorRT 11.x로 조기 전환하지 않고,
JetPack이 설치한 **TensorRT 10.3.0을 고정 기준**으로 사용한다. 11.x는 weak-typing
`BuilderFlag` 계열 API를 제거하는 breaking change가 있어 지금 전환하면 이후 모든 코드를
다시 짜야 하므로, 이번 PoC 전체를 10.3.0 API로 작성한다.

디스크 여유공간(작업 시작 시점): 7.9GB — 데이터셋 subset 크기에 유의.
