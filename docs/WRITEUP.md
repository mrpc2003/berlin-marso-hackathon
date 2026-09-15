# WarehouseSort RGB 트랙 결과 보고 (fork: mrpc2003, 2026-09-15)

Kaggle "Marso Hack Berlin 2026 — Robot Parcel Sorting Challenge"의 RGB 트랙을 대회 마감(2026-06-21)
이후 무료 Colab T4만으로 수행한 기록이다. 공식 held-out 채점은 받지 않았으므로 아래 수치는 모두
자체 검증값이다. 실행 로그와 해시는 [EXPERIMENTS.md](EXPERIMENTS.md), 프로토콜은
[VALIDATION_PROTOCOL.md](VALIDATION_PROTOCOL.md), 배경 조사는 [RESEARCH.md](RESEARCH.md)에 있다.

## 1. 결과 요약

600회 고정 프로토콜 검증(레벨당 100회 × 2 프로토콜, 결과 확인 전에 시드와 교란 범위를 고정).

| 프로토콜 | easy | medium | hard | 가중 합 (0.2 / 0.3 / 0.5) |
|---|---|---|---|---|
| fresh_seed (seed 6000–6099, 학습 분포 내 새 시드) | 0.990 | 0.890 | 0.885 | **0.9075** |
| stress (seed 8000–8099, 위치·자세·bin 교란 확대) | 0.975 | 0.7175 | 0.6467 | **0.7336** |

- `sort_accuracy` = 올바른 색 bin에 놓인 소포 비율. mis-sort는 모든 단계에서 1% 이하.
- easy fresh_seed는 배치가 고정된 환경이라 반복성 검사에 가깝고, 공간 일반화 증거는 아니다.
- 참고로 대회 당시 검증된 RGB 최고 기록(1위, ACT)은 easy 0.50 / medium 0.22 / hard 0.083였다
  (RESEARCH.md). 채점 환경이 다르므로 직접 비교는 불가하다.

## 2. 과제와 관측 계약

- 환경 `WarehouseSort-v1` (ManiSkill 3.0.1, SAPIEN 3.0.3). Franka Panda, 행동 `pd_ee_delta_pos` 4차원.
- 정책 입력은 scene 카메라 RGB 128×128×3과 26차원 proprio뿐이다. 소포·bin 좌표 등 특권 상태는
  학습·추론 어디에도 쓰지 않았다. 검증 도구는 정책에 들어가는 관측이 `{rgb, state}`뿐인지, 행동이
  유한하고 [-1, 1] 안인지 매 스텝 확인한다.
- 레벨: easy 2개(고정 배치), medium 4개(위치 지터), hard 6개(위치·yaw 지터, bin 좌우 교환 50%).

## 3. 방법

스타터의 RGB Diffusion Policy 템플릿(`il/baselines/diffusion_policy/train_rgbd.py`, ResNet18 +
SpatialSoftmax 인코더, 1D U-Net, DDPM 100 스텝, 6.11M 파라미터)을 그대로 두고, 동작하지 않던 부분을
고쳤다.

| 변경 | 이유 |
|---|---|
| 청크 실행 (act_horizon 8, 매 스텝 재샘플링 제거) | 템플릿은 매 스텝 재샘플링해 그리퍼가 떨렸다 |
| 데모 행동을 [-1, 1]로 클리핑해 학습 | ManiSkill이 행동을 클리핑하므로 원본(최대 3.66) 학습은 분포 불일치 |
| 레벨별 에피소드 예산 250 / 500 / 800 | 데모 길이 easy 115 / medium 236 / hard 388(최대 779) |
| 스트리밍 HDF5 로더, resume, 이미지 증강 | Colab RAM 한계와 세션 끊김 대응 |

학습: 레벨별 200개 공식 데모, batch 128, lr 1e-4, AMP. easy 25k / medium 30k / hard 40k iteration.
학습 중 32회 평가에서 가장 좋은 EMA를 선택했다(easy 15000, medium 25000, hard 35000). medium은
23k에서 Drive 경로 손실로 끊겨 22k → 29k → 30k로 optimizer·EMA·RNG를 복원해 이어 학습했다.

Hard 학습 곡선(32회 학습 중 평가, sort accuracy): 5k 6.3% → 10k 20.8% → 15k 40.1% → 20k 59.4% →
25k 80.7% → 30k 84.4% → **35k 90.1%** → 40k 79.2%.

배포 설정은 레벨별로 (act_horizon, diffusion steps) 4조합을 seed 5000+에서 비교해 골랐다.
easy 8/32 (64회 100%), medium 8/16 (32회 98.4%), hard 8/16 (32회 96.4%). 이 값은 설정 선택에 쓴
평가이므로 1절의 독립 검증보다 낙관적이다(medium 98.4 → 89.0, hard 96.4 → 88.5).

## 4. 검증 절차

- 프로토콜은 결과를 보기 전에 `docs/VALIDATION_PROTOCOLS.json`(SHA `e480d74f…`)으로 고정했다.
  4개 환경 × 25 배치, 추론 노이즈 시드 0, 레벨당 100개 시드.
- 각 에피소드의 실제 초기 포즈를 기록해 교란 범위 안에 있는지, 시드 간 변화가 실제로 있는지,
  hard의 bin 교환이 양쪽 다 나타나는지 확인한다. 집계값은 에피소드 행에서 재계산해 대조한다.
- 실행은 tmux 안의 내구성 래퍼가 30초마다 Drive로 복사하고, 완료 시 모든 파일 해시를 양쪽에서
  대조한 뒤에만 `completed`를 기록한다. 첫 실행은 5단계 완료 후 stress_hard 8/100에서 Colab VM 회수로
  끊겼고, `--stages stress/hard`로 그 단계만 새 런타임에서 다시 돌렸다(같은 체크포인트, 같은 시드).
- 증거: `MyDrive/marso/validations/durable_20260915-001259_…`(5단계),
  `durable_20260915-045610_…`(stress_hard, seal `a41ddc89…`).

stress_hard 세부: 100회 중 6개 전부 분류 41회, 0개 9회, mean_steps 618/800. 교란 범위는 위치
±0.03 m, yaw ±0.15 rad, bin ±0.01 m.

## 5. 재현

```bash
git clone https://github.com/mrpc2003/berlin-marso-hackathon && cd berlin-marso-hackathon
pixi install
pixi run python eval.py difficulty=hard obs_mode=rgb policy=warehouse_sort.il_policy:load_dp_rgb \
    checkpoint=checkpoints/rgb_dp_hard.pt eval_config=conf/eval/eval32.yaml record_video=false
```

- `submission.yaml`은 RGB 트랙만 선언한다. 체크포인트는 EMA 가중치(float32)와 정책 config만 남긴
  34 MB 파일이며, 원본 EMA와 텐서 단위로 동일함을 확인했다. 추론 설정(act_horizon, steps)은 config에
  들어 있어 채점자가 kwargs를 넘길 필요가 없다.
- 체크포인트 SHA-256: easy `7a6de3d4…`, medium `35d47488…`, hard `9dfd78d8…`
  (원본 EMA: `c563ab46…`, `95ecf9a9…`, `3bbb7179…`).
- 600회 검증 재현: `tools/run_generalization.py --run`(전체) 또는 `--stages stress/hard`(부분).
  CPU 테스트 `python -m pytest tests -q`.
- 학습·검증 런타임은 Python 3.12.3, torch 2.11.0+cu128, numpy 1.26.4였다. 스타터 `pixi.toml`은
  torch 2.12 / numpy 2.4를 고정하므로, 판정 절차 그대로 클린 클론 스모크를 별도 T4 호스트에서 돌려
  이 차이를 확인했다(2026-09-15, Ubuntu 22.04, 드라이버 580.178): `git clone main` → `pixi install --locked`
  (141 s, torch 2.12.0+cu130 / numpy 2.4.6 / mani_skill 3.0.1 / sapien 3.0.3) → `eval.py` 6회.
  결과는 기본 4회 easy/medium/hard 모두 1.000, 32회 easy 0.984 / medium 1.000 / hard 0.953으로
  학습 런타임의 같은 시드 값(1.000 / 0.984 / 0.964)과 최대 1.6 pt 차이(재현 규정 5% 이내)였다.
  기록: EXPERIMENTS.md S5, 증거 `outputs/s5_clean_clone_smoke_20260915/`.

## 6. 한계

- 공식 held-out 점수가 없다. 자체 stress 프로토콜은 비공개 채점 조건을 재현한 것이 아니다.
- hard는 교란이 커지면 급격히 떨어진다(88.5 → 64.7%). 데모가 스크립트 정책 한 종류라 복구 행동이
  없고, 6개 소포 배치에서 초기 실패가 누적된다.
- 설정 선택용 32회 평가는 100회 독립 평가보다 5~10 pt 낙관적이었다.
- 무료 Colab T4의 세션 회수 때문에 학습·검증을 여러 번 나눠 이어 붙였다. 각 이음새의 체크포인트
  SHA와 상태 복원은 EXPERIMENTS.md에 기록돼 있다.
