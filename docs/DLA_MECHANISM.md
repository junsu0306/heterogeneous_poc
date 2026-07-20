# NVDLA Convolution Core 계산 메커니즘 — 1차 소스 기반 정리

> 작성일: 2026-07-02
> 목적: DHS를 다시 설계하기 전, "실제 DLA가 conv를 어떻게 계산하는가"를 추측이 아니라
> 1차 소스(NVDLA 공식 문서 + 이를 분석한 학술 논문)로 체계적으로 규명한다.
> 소스 원문: `phase0f/docs/hwarch.html`, `unit_description.html` (nvdla.org, 2026-07-02 다운로드),
> `tempus_core.pdf` (arXiv 2412.19002, "Convolution Core for Low-Precision Edge DLAs")

---

## 1. 전체 파이프라인 (5단계)

```
CDMA(DMA) → CBUF(버퍼) → CSC(스케줄러) → CMAC(곱셈+1차누산) → CACC(2차누산+반올림) → SDP(bias/BN/활성화 등)
```

Convolution Pipeline은 **1024개의 MAC**(FP16/INT16용, INT8는 2048개)과 **32-element accumulator array**를
갖는다고 개요에 나오는데, 이 "32"는 (뒤에서 밝혀지듯) **누산기 폭이 아니라 stripe 버퍼 크기**(공간상
서로 다른 출력 지점을 몇 개까지 동시에 처리 가능한지)를 가리킨다 — 계획서가 "CACC의 32-element
accumulator array"로 표현한 부분과 실제 의미가 다르다는 걸 이번에 확인했다 (§3 참고).

## 2. CMAC — 곱셈 + 1차(레벨-1) 누산

- **구조**: 16개의 동일한 "MAC Cell"로 구성 (MAC Cell Array). **각 MAC Cell은 FP16/INT16용 곱셈기 64개**를
  가짐 (INT8는 곱셈기 1개가 2개로 분할되어 128개). 총 16×64 = 1024 MAC.
- **Atomic Operation**(기본 연산 단위): 한 MAC Cell이 입력 활성값의 1×1×64 조각과 가중치의 1×1×64
  조각을 곱해서 더한 결과("partial sum", PS)를 **1 사이클**에 낸다:

  ```
  PS_{w,h,k,r,s,c} = Σ_{i=c}^{min(c+63, C-1)} x[...] * wt[...]     (64개 항의 합, c는 항상 64의 배수)
  ```

- **핵심 확인 사실 — 내부 리덕션은 adder tree**: NVDLA 공식 문서 자체는 이 64-way 합산의 정확한 내부
  구조(순차/트리)를 명시하지 않지만, NVDLA의 CMAC/CACC를 정면으로 분석한 학술 논문(Tempus Core,
  arXiv 2412.19002)이 명시적으로 이렇게 서술한다:

  > *"each PE cell contains local registers and **an adder tree** to accumulate the intermediate
  > results, producing one partial sum corresponding to the PE cell"* (Tempus Core §II, NVDLA
  > CMAC 서술 부분 — Tempus Core 자신의 대안 설계가 아니라 **NVDLA 원본**을 설명하는 문장)
  >
  > *"...with partial results accumulated in the adder trees of the convolution accumulation
  > (CACC) unit"* (Tempus Core §I)

  이걸로 "64개 항이 하드웨어 adder tree로 결합된다"는 게 **1차 소스로 확인**됐다 (F-0가 실측으로
  간접 추론했던 것과 정성적으로 일치 — bf=2/4가 여러 채널 깊이에서 실제 DLA와 정확히 일치했던 것도
  이 tree 구조의 실측 반영이었을 가능성이 높다). 다만 트리의 **정확한 위상**(균형 이진 트리인지,
  Wallace tree류의 다른 구조인지)까지는 공개 문서로 확인 불가 — 균형 이진 트리(bf=2, 6-level)를
  가장 합리적인 기본 가정으로 채택한다.

## 3. CACC — 2차(레벨-2) 누산 + 최종 반올림 **[가장 중요한 신규 발견]**

### 3.1 구조적 사실 (NVDLA `unit_description.html` Table 49, Table 50에서 직접 확인)

| Input Format | Accumulative Sum (누산 중 정밀도) | Truncated Result (최종 출력 정밀도) |
|---|---|---|
| INT8 | INT34 | INT32 |
| INT16 | **INT48** | INT32 |
| **FP16** | **FP44** (8-bit 지수, 38-bit signed decimal) | **FP32** (IEEE754) |

**계획서가 "FP48"로 통칭했던 것은 부정확했다** — INT16 모드만 정확히 48비트이고, **우리가 실제 쓰는
FP16 모드는 FP44**다 (8비트 지수 + 38비트 부호있는 소수부). 어느 쪽이든 핵심은 같다: **FP16(10비트
가수)보다 압도적으로 넓고, 심지어 FP32(23비트 가수)보다도 넓다** (38비트 가수부).

### 3.2 CACC 동작 순서 (문서 원문 그대로)

> *"Prefetch accumulative sums from the assembly SRAM group. When partial sums arrive, send them
> to adder array along with accumulative sums... Gather new accumulative sums from output side of
> adder array. Store into assembly SRAM group. **Repeat step1~step3 in terms of stripe operation
> until a channel operation is done.** If a channel operation is done, the output of adders is
> rounded and saturated."*

즉 CACC는 (R×S×⌈C/64⌉)개의 "atomic 64-way partial sum"을 **순차적으로(sequential)** 계속 더해
나간다 — 이 레벨-2 누산은 **트리가 아니라 순차 누적**이며, 매 덧셈마다 반올림하지 않고 **FP44 정밀도를
쭉 유지**하다가, 그 output 채널의 conv가 완전히 끝나는(channel operation 완료) 시점에 **딱 한 번**
FP44 → FP32로 반올림/saturate한다.

### 3.3 이게 바꾸는 것

**`dhs/dhs_ops.py`의 현재 구현은 이 구조와 정반대다** — 매 pairwise 덧셈마다 FP16으로 반올림하고
있었다. 실제로는:

1. **레벨 1** (한 MAC Cell 내부, 64개 항): adder tree로 결합, 중간에 반올림 없이 넓은 정밀도로 파이프라인
   통과 (1 사이클)
2. **레벨 2** (R×S×⌈C/64⌉개의 레벨-1 결과들을 CACC에서 결합): **순차 누적**, FP44 정밀도로 유지,
   중간 반올림 없음
3. **반올림은 딱 한 번**: 해당 출력 원소의 conv 전체(R×S×C 항)가 다 더해진 뒤 FP44→FP32
4. 이후 SDP가 bias/BatchNorm/활성화를 적용하고, 다음 레이어가 필요로 하는 포맷(FP16 등)으로
   precision conversion — 이 지점이 (문서상 두 번째) FP16 rounding이 실제로 일어나는 곳으로 추정
   (SDP의 정확한 내부 반올림 시점은 이번 조사에서 완전히 확인하지 못함 — 잔여 불확실성)

## 4. 정리 — 계획서 §6.1.1 4+1 요소 표 갱신

| 요소 | 계획서 원래 기술 | 이번 조사로 확인/정정 |
|---|---|---|
| A. 누산기 폭 | FP48 (48-bit), 낮음 불확실성 | **FP16 모드는 정확히 FP44**(8b exp + 38b decimal)임을 1차 소스로 확인. "48bit"는 INT16 모드에만 해당 — 계획서 표현이 부정확했음. 여전히 "낮음 불확실성"(오히려 더 확실해짐, 표 번호까지 확인) |
| B. Reduction 구조 | 2단계, 순서는 불확실 | **레벨1=adder tree(그룹 크기 정확히 64), 레벨2=순차 누적(R×S×⌈C/64⌉ 항)** — Tempus Core 논문이 "adder tree"를 명시적으로 확인. 그룹 크기(64)는 이제 확정, 트리 내부 위상만 잔여 불확실 |
| C. 라운딩 모드 | RTN 가정, 높음 불확실성 | 여전히 미확인 — Table 49의 "round and saturation" 언급은 있으나 RTN/RTZ/RTE 중 무엇인지는 `D_CLIP_CFG`의 `CLIP_TRUNCATE` 필드가 결정한다고만 나옴(레지스터 필드 존재는 확인, 기본값/실제 설정값은 별도 확인 필요) |
| D. FTZ | on 가정, 높음 불확실성 | 이번 조사에서 명시적 언급 발견 못함 — 여전히 불확실 |
| E. Conv 알고리즘 | Direct/Winograd 자동선택 | 기존 이해와 동일. Winograd는 CACC에서도 **별도의 정밀도 경로**(Table 50: FP16 Winograd는 Adder 0~63 활성화, DC/Image는 Adder 0~15)를 씀 — 0A-3에서 이미 Winograd 여부 확인 불가하다고 결론 낸 것과 별개로, 만약 확인된다면 정밀도 모델링도 달라져야 함을 시사 |

## 5. 남은 불확실성 (정직하게 남겨둘 것)

- CMAC 내부 adder tree의 정확한 위상(균형 이진 트리 vs Wallace tree류) — 공개 문서로 확인 불가
- 라운딩 모드(RTN/RTZ/RTE) — 레지스터 필드 존재만 확인, 값은 불확실
- FTZ 여부 — 확인 못함
- SDP 단계에서 FP32→FP16 변환이 정확히 어느 시점/방식으로 일어나는지 — Table 29/30(정밀도 변환
  표)이 원본 HTML에서 텍스트로 깔끔하게 추출되지 않아 이번엔 확인 못함 (표가 이미지이거나 복잡한
  HTML 테이블 구조일 가능성 — 필요시 스크린샷/수동 확인 필요)

## 6. 재현 경로

```bash
cd /media/airlab_compression/nvme_storage/poc/phase0f/docs
wget -q "http://nvdla.org/hw/v1/hwarch.html" -O hwarch.html
wget -q "http://nvdla.org/hw/v1/ias/unit_description.html" -O unit_description.html
wget -q "https://arxiv.org/pdf/2412.19002" -O tempus_core.pdf
# 이후 BeautifulSoup(html)/pdfplumber(pdf)로 텍스트 추출, grep으로 CACC/CMAC/accumulat 검색
```

원본 파일: `hwarch.html`, `unit_description.html`, `tempus_core.pdf` (+ 추출된 `.txt` 버전)
전부 `phase0f/docs/`에 보존.
