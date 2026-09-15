# Experiment log (append-only)

## S2 medium 복구 학습 재개 — 실행 확인

- 실험 `medium_resume_20260914-040647`: 같은 Colab 노트북에 복구 셀 5개를 추가했다. 기존 7개 셀은 보존하고 이전 시작 스위치만 껐다. Drive에서 노트북을 다시 내려받아 소스 일치를 확인했다.
- 새 T4의 실제 CUDA 연산, medium 데모 200개 해시, 원래 22,000회·최고 20,000회 저장본, 스크립트 기준 4/4 분류를 검증한 뒤 복구 학습을 시작했다.
- 22,082 → 22,444 → 24,948회 진행 로그 확인. Drive에 복사된 24,000회 체크포인트를 다시 내려받아 로드했다. 모델 텐서 197개가 갱신됐고 scheduler step은 24,001, 학습률은 9.861242833909228e-06이다.
- 복구 후 저장본 SHA-256: `2733de8fb07970ed9f9d09297529676af0c944699a69e0feb6ea23db06f39acd`. 전체 학습·최종 평가 완료를 뜻하지는 않는다.
- [복구 결과 폴더](https://drive.google.com/drive/u/0/folders/1pkbympEoWWazU6WZU0jn4MZakrDTeKVk). 학습과 최종 평가·영상·진단이 끝날 때까지 실행 셀이 기다린다. 완료 점검은 2026-09-14 14:08 KST에 한 번 예약했다(`799d2853b455`).


## S2 medium 복구 준비 — Drive 승인 대기

- 원래 실험 `medium_20260914-014131`은 23,000회 저장 시 Drive 체크포인트 경로가 사라져 종료됐다. `supervisor.log`의 exit 1과 다운로드한 전체 `train.log`로 확인했다. 완료된 실험이 아니다.
- `latest.pt`를 내려받아 22,000회, optimizer·scheduler·EMA·RNG 상태를 확인했다. SHA-256: `978b8e10ac5d91753e7c7d337c1dc797d4a5b17a6f2f2fa5229adf66dc3857a4`.
- 이전 최고 모델은 20,000회, 32에피소드 분류율 0.8203125. SHA-256: `c0be440c121ac15aaa4cd67e185a2a1f58895a69f5c1c37155b13fb048d9844c`.
- 복구 코드는 `tools/run_colab_medium_resume.py`, 노트북은 `MARSO_COLAB_MEDIUM_RESUME.ipynb`. 기존 학습 일정 30,000회를 유지하고 22,000회 상태에서 재개하도록 구성했다. 원래 실험과 easy 저장본은 수정하지 않는다.
- 체크포인트·로그·평가 결과를 Colab 로컬에 먼저 기록하고 별도 Drive 복구 폴더로 복사한다. 실제 마운트 확인, 복사 해시 검사, 시간 제한·재시도, 최종 `run/durability.json` 검증 기록을 추가했다. CUDA 재개 시 RNG 텐서를 CPU로 복원하는 수정도 포함한다.
- 격리 worktree의 CPU 테스트 57건 통과. 실제 GPU 복구 학습은 아직 시작하지 않았다. 새 T4 연결은 확인했으나 Drive 마운트 승인 창에서 대기 중이다. 커밋·푸시·제출 없음.


## S2 medium — prepared, awaiting Drive consent

- New notebook: [MARSO_COLAB_MEDIUM.ipynb](https://colab.research.google.com/drive/1FBLfw1b6YEWE7uPnKDgGHkBrVN7N5rDI). T4 selected; the first Drive-mount cell is waiting at the explicit Google Drive access dialog. No S2 environment install, simulation, smoke training, or main training has run yet.
- Medium RGB demos downloaded through the existing local Kaggle login without copying credentials. HDF5: 200 readable trajectories, lengths 234–238, 4 parcels; JSON: 200 episode records. Both files uploaded and listed under [marso/demos/medium](https://drive.google.com/drive/u/0/folders/1lLuJL-VbB7QKP8bw0oOzLjO-aoIy-OPl).
- Source data hashes: HDF5 `a9572cd0c2f8c95be49f4af3fdfec16827477e392f42c9a1d92a7350c3edeeb0`; JSON `267243b11efb7b354f2c5c8a404d98aff1afb6cf98bef59a0b01aab8d7960b11`. These are pinned in runtime staging checks.
- New files only: `tools/run_colab_medium.py`, `tools/make_colab_medium_notebook.py`, `tests/test_colab_medium_workflow.py`, and the medium notebook. Existing S1 model/results and scripts were preserved. Implementation was prepared in `.worktrees/cursor-colab-medium`, reviewed and copied to the main working directory.
- Local verification: 33 tests passed; notebook AST/nbformat passed; narrow secret scan clean. GPU verification is still pending and must not be inferred from local tests.
- Planned sequence after consent: Python 3.12 setup → medium data hash/schema check → scripted 4/4 reference at 500 steps → 300-iteration smoke plus 8-episode deployment evaluation → full-data speed probe → deliberate launch of 30,000-iteration medium training. `RUN_MAIN=False` by default; do not turn it on until the preceding checks actually pass.
- Supervisor uses an experiment-specific local lock, source hashes, periodic checkpoints, checked stage exits and timeouts. On success it runs four 32-episode settings, a one-environment video, and 8-episode diagnostics. Main medium is a fresh training run, not an easy resume; non-empty target checkpoint directories are refused.
- No hard/ACT run, submission, or push is automatic. Operational links and local file hashes are saved in `outputs/s2_preflight/handoff.json`.

## Latest verified result — S1 easy COMPLETED

Verified from the Google Drive `summary.json` and `status.json` on 2026-09-14 in the morning. The run finished at **02:42:47 KST**, including 25,000 training iterations, all four 64-episode evaluations, preview creation, and diagnostics. Historical RUNNING entries below describe earlier observations, not the current state.

| action horizon | denoising steps | episodes | sort accuracy | all-parcels-success episodes |
|---|---|---|---|---|
| 8 | 16 | 64 | 97.65625% | 61/64 |
| 4 | 16 | 64 | 96.875% | 61/64 |
| 8 | 32 | 64 | 100% | 64/64 |
| 4 | 32 | 64 | 98.4375% | 62/64 |

Best: horizon 8, 32 denoising steps; 128/128 parcels sorted, mis-sort rate 0. Gate 1: **DP main line**. This is the fixed-layout **easy** validation protocol, not medium/hard or an official held-out/leaderboard result.

- [Drive result folder](https://drive.google.com/drive/u/0/folders/1TCDbHAc6NPE0kMqpdMxa8wcYuHsw5WHS)
- [Drive summary](https://drive.google.com/file/d/10a2mSZHjWonotynFUq7h_6U7LlIj3CtL/view)
- [Checkpoint folder](https://drive.google.com/drive/u/0/folders/17jM3VMthXKLTf5ndtuOm6Q_X26q3ykwd): best_eval_sort_accuracy.pt, its compact .submit.pt, final.pt and latest.pt are present.
- Best checkpoint hash recorded consistently in summary and status: `c563ab461b4b1ad87f5a3d2d77df8453334eb6d8a3e6a86f20b7b42676352df9`.
- Local copies of the parsed Drive records: `outputs/s1_verified/summary.json` and `status.json`.
- The scheduled check could not capture Cursor at 03:30; its scheduler status `ok` meant the check returned a blocked report, not that training completion was verified then. By the morning the Colab server had been removed externally or for inactivity. Drive records establish that the run completed before that discovery.
- No training was relaunched; no submission or push was performed. No medium/hard training was started.

Record every serious run here: command, git commit, checkpoint, eval config + episode count, scores.
Decision rule: easy on 64 episodes, medium/hard on 32 (16-episode evals are noise: ±9 pt on easy).

| date | session | level | exp / ckpt | key settings | eval (cfg, n) | sort_acc | mean_steps | notes |
|---|---|---|---|---|---|---|---|---|
| 2026-09-13 | S0 | – | – | code changes only (chunked deployment, per-level episode budget, clipped actions, streaming loader, resume, aug) | CPU tests: 10 pass | – | – | no GPU run yet |
| 2026-09-14 | S1 smoke | easy | `cursor_smoke_20260913-154033/final.pt` | 300 iters, 20 demos; deployment horizon 8 / DDPM 16 steps | seeds 5000–5007, 8 eps | 0.000 | – | Pipeline smoke, not Gate 1; checkpoint loaded and JSONL written on T4 |
| 2026-09-14 | S1 main — RUNNING | easy | `rgb_dp_easy_s1_20260913-154033` | 25,000 iters, 200 demos, batch 128; save every 1,000; eval every 5,000 | 32 eps during training; final sweep 4 × 64 pending | – | – | Last inspected: step 2,770; latest.pt loaded at iteration 2,000 with optimizer + RNG |

## S1 Cursor/Colab handoff

- Active notebook: `MARSO_CURSOR_CONTINUE.ipynb`. Use only its `main-monitor` cell while the main run is active; **do not Run All** or restart/remove the Colab server.
- Runtime: attached Python 3.13 kernel retained; all simulator/training commands use `/content/marso-py312/bin/python` (Python 3.12.3, torch 2.11.0+cu128, torchvision 0.26.0+cu128, numpy 1.26.4, ManiSkill 3.0.1, SAPIEN 3.0.3, mplib 0.1.1, gymnasium 1.3.0, diffusers 0.38.0). The original install failed because mplib 0.1.1 has no Python 3.13 wheel.
- GPU preflight passed: real CUDA tensor operation; 512×512 RGB render; scripted reference sorted 2/2 parcels in 115 steps. TCP displacement under action 3.66 versus 1.0 matched exactly.
- All 200 HDF5 trajectories were read and source/Colab/Drive hashes matched. The existing JSON contains 50 episode records; its env_info is present. Missing metadata was not invented.
- Deployment evaluation exposed and fixed empty policy_kwargs conversion and duplicate n_episodes keyword errors. Training evaluation now returns flat, trimmed per-episode metrics instead of counting batches. Local tests: 27 passed. Focused tests on Colab: 8 passed. The deployment smoke completed 8 episodes and persisted its result.
- Full-data speed probe: 200 iterations took 58.660 s including startup; steady training 5.23 it/s. Total iterations were fixed at 25,000 before launch to target about 80 minutes of training, excluding evaluation. Main-run observations later showed about 6.2 it/s. Loss is not a sorting-success measure.
- Drive checkpoint directory: `MyDrive/marso/ckpts/rgb_dp_easy_s1_20260913-154033/`. Verified latest.pt: iteration 2,000 / target 25,000, 178,772,355 bytes, SHA-256 `fa842ddf56c42dbac335947fdee589e4fe78228e486ca27406de7a0712c8271f`. This is an intermediate checkpoint, not the final candidate.
- Drive run directory: `MyDrive/marso/runs/rgb_dp_easy_s1_20260913-154033/`, containing launch_plan.json, plan.json, status.json, train.log and, after completion, evaluation_results.json, summary.json, preview/video and diagnostics. Supervisor PID was 21092 at launch; read current status rather than trusting a remembered PID.
- Source provenance: base commit `6a7346f`, plus hash-verified local fixes copied into the Colab checkout. The full source hash map is frozen in plan.json. No commit or push was performed during this continuation.
- Final supervisor stages: finish training → (action horizon, denoising steps) = (8,16), (4,16), (8,32), (4,32), each on the same 64 evaluation seeds → best-setting one-environment video and 8-episode diagnostics. Failures stop the pipeline and are recorded in status.json. No new S2 training or submission is automatic.
- One-shot follow-up: Hermes job `8f766b33d560`, 2026-09-14 03:17 KST, delivery `bot-chat:maclocal`. It must read fresh remote state, not infer completion from this note.
- Secondary-metric caveat: the smoke reported mean_steps=100 despite the 250-step wrapper budget; do not use this field as elapsed steps without auditing the environment's unfinished-episode fallback. Primary sorting accuracy and explicit episode counts were the verified smoke outputs.

## Gates
- Gate 1 (end S1, easy, 64 eps): ≥40% → DP main line; 30–40% → run ACT hedge next; <30% → diagnose + one retrain (Gate 1b), then ACT becomes main line.
- Gate 2 (S3): ACT beats DP by >10 pt on the same 64 episodes → switch.
- Gate 3 (hard, 32 eps): ≥5% include; 0% after two attempts → stop spending on hard.


## S2 medium Cursor continuation — 2026-09-14 17:49 KST verified snapshot

- Current blocker: the Mac is locked, so Cursor could not be inspected or controlled in the final attempt. The last visible `MarsoT4b` state was a Drive-access consent dialog. This runtime's CUDA execution and Drive mount are not yet verified. No training was started in this continuation.
- Earlier in this continuation, Cursor executed an actual Tesla T4 CUDA tensor operation (`358438400.0`) in `MarsoT4`, plus the Python 3.12 subprocess CUDA test, 200-demo validation, a 512×512 RGB render and medium scripted 4/4 reference. After laptop sleep, Cursor showed no active Colab server; that earlier proof does not establish the new runtime's state.
- Downloaded and loaded the original recovery `medium_resume_20260914-040647` inputs: latest.pt iteration **29,000**, scheduler 29,001, SHA-256 `8082cbb5b4da6a871ad754266d13a8efad714790e6b956502f8f440b328a205b`; best_eval_sort_accuracy.pt iteration **25,000**, scheduler 25,001, SHA-256 `95ecf9a9c28153aef6ceb130e7a571897806a44d2196a34bade7a5f1f811fdeb`. Model and EMA weights were finite; optimizer, scheduler, EMA and RNG state were present. rclone check reported 2 matching files and 0 differences. The 29,153 log line is not a saved checkpoint iteration.
- The 25,000 intermediate evaluation records 32 episodes and sort_accuracy=1.0. It is not the four-setting final deployment result. No final.pt, final summary, evaluation sweep, video, diagnostics or final durability receipt was verified.
- Reviewed next notebook: `MARSO_COLAB_MEDIUM_RESUME_29000_REVIEWED.ipynb` (RUN_RESUME=False). Use this copy for the next setup/launch; the earlier `29000` and `T4B` copies retain the first review snapshot. Resume inputs and lineage now pin the verified 29,000/25,000 files, and each execution uses a separate output name.
- Minimal edits were prepared by Codex CLI in `.worktrees/cursor-medium-29000` with `-a never -s workspace-write`. Hermes independently caught a final-scheduler validation mismatch; the reviewed correction requires 30,000 for final checkpoint scheduler state while preserving strict iteration+1 checks for intermediate inputs. The original trainer/sampler convention yields 999 remaining batches and final scheduler 30,000.
- Final validation: Codex related suite **102 passed, 1 skipped**; primary focused suite **83 passed, 1 skipped**; independent Hermes final suite **74 passed, 1 skipped**. The unchanged CPU environment lacks nbformat, so global Python separately passed nbformat, AST and six decoded payload/source hash checks. Scoped secret scan: 0 matches. Hermes final source review: PASS. These checks do not establish GPU training completion.
- Original six notebook cells remain byte-for-byte in order. Existing trainer and other preexisting sources were preserved; four recovery files changed. Timestamp backups and `final_source_sha256.json` identify the reviewed version. No commit, push, submission, hard/ACT launch or active runtime restart was performed.
- Resume instructions and evidence: `outputs/s2_cursor_audit_20260914_1543/handoff.json`; the canonical `outputs/s2_recovery/handoff.json` points to this verified snapshot and its previous-state backup. Next steps remain user unlock/Drive consent → fresh T4 proof → per-cell preflight → 30,000 resume → four × 32 medium evaluation → one-environment video and 8-episode diagnostics → actual Drive reread/hash/load verification.
- Existing rclone auth was preserved. Its shared Drive OAuth client_id warning still requires a separate long-term migration to an own client_id before retirement during 2026.


## S2 medium Cursor continuation — COMPLETED and independently verified (2026-09-14 18:48 KST)

- Experiment `medium_resume_29000_20260914-091049` resumed the validated 29,000 input from `medium_resume_20260914-040647` on the actual Cursor-attached Tesla T4. Python 3.12 CUDA, 200 medium demos, 4/4 scripted reference, and absence of duplicate processes/locks passed before launch.
- Downloaded final.pt is **iteration 30,000**, scheduler/EMA 30,000, finite weights, SHA-256 `6c5a78f99709979afabe4e1fb94374b34e127e8bf13f46bc6705eff91e9864d3`. The preserved best checkpoint remains iteration 25,000, SHA-256 `95ecf9a9c28153aef6ceb130e7a571897806a44d2196a34bade7a5f1f811fdeb`. The final model's internal 32-episode sort_accuracy was 0.953125; this is separate from the following deployment sweep.

| Action horizon | Denoising steps | Episodes | Sort accuracy | All parcels sorted |
|---|---|---|---|---|
| 8 | 16 | 32 | 98.4375% | 31/32 |
| 4 | 16 | 32 | 87.5000% | 28/32 |
| 8 | 32 | 32 | 92.1875% | 29/32 |
| 4 | 32 | 32 | 85.9375% | 27/32 |

- Each deployment row uses medium, seed0=5000, 32 episodes, 4 parcels and 500-step limit. All four had mis_sort_rate=0. Best setting: horizon 8 / denoising 16 (126/128 parcels; all parcels sorted in 31/32 episodes). This is not the official held-out score.
- Best-setting one-environment video: `outputs/s2_cursor_audit_20260914_1543/continued_run/run/preview/videos/0.mp4` (1024×512, 25.05 s, 501 H.264 frames; full decode and sampled visual inspection passed). Its preview evaluation sorted 4/4 parcels. Eight diagnostics episodes sorted 32/32 parcels.
- Final Drive tree reread: **31 matching files, 0 differences**. All **29 immutable receipt payloads** matched SHA-256 and size. Summary, per-setting JSONL/results, diagnostics, final/best checkpoints, status and durability were consistent. All four checkpoint payloads loaded with finite model tensors.
- [Drive recovery folder](https://drive.google.com/drive/u/0/folders/1OWfk48qWONNZQFAfdGOYyWeORoniGue8). Detailed local completion report and validation records: `outputs/s2_cursor_audit_20260914_1543/continued_run/COMPLETION.md`. Canonical handoff is updated to completed; previous snapshots remain in timestamp backups.
- Cursor's completed execution output was saved; `RUN_RESUME=False` restored. All six original reviewed source cells are preserved plus the separate read-only prelaunch cell. Runtime was not stopped/restarted. No commit/push/submission/hard/ACT launch. No required work remains.


## S4 stress_hard 재실행 — COMPLETED (2026-09-15 14:27 KST, Drive 영수증 검증)

- 배경: 첫 600회 검증 `durable_20260915-001259_21b868e5…`는 5단계를 완료·봉인한 뒤 `stress_hard` 8/100에서 Colab VM 회수로 중단됐다. 남은 한 단계만 다시 돌리기 위해 `tools/run_generalization.py`와 `tools/run_validation_durable.py`에 `--stages`(순서 있는 `protocol/level` 부분집합) 옵션을 추가했다. 시드·프로토콜 JSON(`e480d74f…`)과 세 체크포인트 SHA는 변경하지 않았다. 부분 실행은 `summary.json`에 `stages`/`full_frozen600=false`를 기록하고 가중 점수를 계산하지 않는다. CPU 테스트 126건 통과(신규 4건 포함).
- 실행: 새 T4 런타임에서 `MARSO_STRESS_HARD_RERUN_20260915.ipynb`(Drive 소스 zip `3a4c5e71…`, 래퍼 `0f1ee625…`)로 `--stages stress/hard` 단독 실행. 사용자가 Cursor에서 셀을 실행했고, 셀 3의 `StopIteration`은 래퍼가 첫 JSON 줄을 쓰기 전에 상태 셀이 돌아 생긴 타이밍 문제로 결과와 무관하다.
- 결과 `durable_20260915-045610_0e7953d2…` / `validation_20260915-045612-755688117`: status **completed**, remote **final_verified**, seal `a41ddc89…`(14 files), hard 체크포인트 `3bbb7179…` (act_horizon 8, 16 steps), seeds 8000–8099, pose_effect verified.

| 지표 | 값 |
|---|---|
| sort_accuracy | 388/600 = **0.6467** |
| all_placed_rate | 0.41 |
| mis_sort_rate | 0.01 |
| mean_steps | 617.7 / 800 |
| 에피소드별 분류 개수 분포 (0..6) | 9 / 10 / 14 / 9 / 8 / 9 / 41 |
| 계산 시간 | 1714 s |

- 600회 검증 최종 표 (앞 5단계는 `durable_20260915-001259…` 봉인 결과, stress_hard는 본 실행):

| 단계 | easy | medium | hard | 0.2/0.3/0.5 가중 |
|---|---|---|---|---|
| fresh_seed (6000–6099) | 0.990 | 0.890 | 0.885 | **0.9075** |
| stress (8000–8099) | 0.975 | 0.7175 | 0.6467 | **0.7336** |

- 해석: stress hard는 위치 ±0.03 m, yaw ±0.15 rad, bin ±0.01 m 교란에서 fresh_seed 대비 24 pt 하락했고, 41%의 에피소드만 6개 전부 분류했다. 자체 프록시이며 공식 held-out 점수가 아니다.
- 로컬 사본: `outputs/s4_stress_hard_rerun_20260915/drive_copy/`, 결합 표 `outputs/s4_stress_hard_rerun_20260915/combined_600_metrics.json`. 커밋·푸시·제출은 하지 않았다.


## S5 클린 클론 판정 스모크 — PASSED (2026-09-15 15:47 KST, SSH T4 호스트 milabt4)

- 목적: 채점자 절차(`git clone` main → `pixi install` → `eval.py`)를 학습 런타임과 무관한 머신에서 재현. 스타터 `pixi.lock`은 torch 2.12.0+cu130 / numpy 2.4.6을 고정하는데 학습·검증은 torch 2.11 / numpy 1.26에서 했으므로 이 차이가 실제 실행에 영향을 주는지 확인.
- 호스트: Ubuntu 22.04, Tesla T4 ×4(GPU 0 사용), 드라이버 580.178.04. 사전 조치: 9/11 자동 업데이트로 어긋난 NVIDIA 커널 모듈을 재로드(580.173.02 → 580.178.04, 재부팅 없음), `libvulkan1 vulkan-tools` 설치(NVIDIA ICD는 이미 존재).
- 절차(`smoke_ssh.py`, 결과 `outputs/s5_clean_clone_smoke_20260915/milabt4_20260915_062146/`): clone main `6f93ed0` → 체크포인트 3개 SHA 일치 → `pixi install --locked` 141 s → env probe `{"torch": "2.12.0+cu130", "cuda": true, "gpu": "Tesla T4", "numpy": "2.4.6", "mani_skill": "3.0.1", "sapien": "3.0.3"}` → 엔트리포인트 import OK → eval 6회. `pixi run install`은 upstream `pixi.toml`에 task가 없어 실패(비치명; warehouse_sort는 editable pypi 의존성으로 이미 설치됨).

| eval_config | easy | medium | hard | 학습 런타임 기준값(32회, 같은 시드·설정) |
|---|---|---|---|---|
| default (4회, seed 5000+) | 1.000 | 1.000 | 1.000 | – |
| eval32 (32회, seed 5000+) | 0.984 (31/32 전량) | 1.000 | 0.953 (29/32 전량) | 1.000 / 0.984 / 0.964 |

- 차이는 최대 1.6 pt(easy·medium ±1 에피소드, hard −1.0 pt)로 판정 재현 규정(5%)을 만족. 오분류 0. 평가 시간 easy 123 s / medium 259 s / hard 413 s.
- 판단: `main`의 제출본은 판정 환경 그대로 설치·실행되며 CUDA 13 빌드 torch도 드라이버 580 이상에서 동작. 남은 미확인 항목은 공식 held-out 점수뿐.
