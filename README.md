# drl_spice_v4

강화학습으로 LNA(Low-Noise Amplifier) 회로 파라미터를 탐색하고, `ngspice` 시뮬레이션 결과를 보상으로 사용하는 프로젝트입니다. 기본 preset은 `CS`와 `CGCS` 두 가지 회로 타입에 대해 PPO, DDPG, TD3, SAC를 실행합니다.

## 구성

- `agents/`: PPO, DDPG, TD3, SAC 구현
- `envs/lna/`: LNA 최적화 Gym 환경
- `simulator/ngspice/`: netlist 템플릿, workspace 관리, ngspice 실행/파싱
- `trains/`: local/distributed trainer 및 Ray runner
- `exps/`: preset config, env factory, launcher
- `loggers/`: structured log reader/writer 및 plot 생성
- `tests/`: unit, integration, regression 테스트

## 요구사항

- Python 3.10+
- `ngspice`
- Ray가 동작하는 로컬 환경
- 회로 템플릿이 참조하는 PDK/모델 파일

주의:

- 템플릿 netlist는 현재 `sky130` 라이브러리를 절대경로로 참조합니다.
- 예: `simulator/ngspice/schematic/CS/*.spice`, `simulator/ngspice/schematic/CGCS/*.spice`
- 환경에 따라 `.lib` 경로를 직접 수정해야 할 수 있습니다.

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`ngspice` 설치 확인:

```bash
ngspice -v
```

## 빠른 시작

CS preset 실행:

```bash
python main_cs.py
```

CGCS preset 실행:

```bash
python main_cgcs.py
```

기본 preset은 다음 파일에서 정의됩니다.

- `CS`: [exps/configs/cs.py](/home/jyhong/projects/drl_spice_v4/exps/configs/cs.py)
- `CGCS`: [exps/configs/cgcs.py](/home/jyhong/projects/drl_spice_v4/exps/configs/cgcs.py)

두 preset 모두 기본값으로:

- `seeds = [100, 200, 300, 400, 500]`
- `max_iters = 10000`
- `n_runners = 10`
- `runner_iters = 5`
- `ppo`, `ddpg`, `td3`, `sac` 활성화

## 설정 방법

실험 설정은 `exps/configs/*.py`에서 dict 형태로 정의합니다.

핵심 키:

- `circuit_type`: `CS` 또는 `CGCS`
- `env_name`: 현재 기본값은 `modular`
- `seeds`: 시드 목록
- `env_kwargs`: 목표 spec, 파라미터 bound, 고정값, 주파수 범위 등
- `launch_kwargs`: 실행할 agent와 trainer/runtime 하이퍼파라미터

상세 형식은 [exps/configs/README.md](/home/jyhong/projects/drl_spice_v4/exps/configs/README.md) 참고.

## 실행 방식

- `main_cs.py`, `main_cgcs.py`는 preset config를 읽습니다.
- `exps/env_factory.py`가 LNA 환경을 생성합니다.
- `exps/launcher.py`가 활성화된 agent를 순차적으로 실행합니다.
- `n_runners > 1`이면 Ray 기반 distributed trainer를 사용합니다.

현재 지원 agent 플래그:

- `ppo`
- `ddpg`
- `td3`
- `sac`
- `ddpg_per`
- `td3_per`
- `sac_per`
- `random`

## 산출물

학습 로그와 체크포인트:

- `./log/{project_name}/`
- 모델 체크포인트: `model.pth`
- 버퍼 체크포인트: `buffer.pkl`
- structured CSV/JSONL 로그: `./log/{project_name}/structured/`

ngspice 워크스페이스:

- `simulator/workstation/{project_name}/{run_id}/`

여기에는 worker별 netlist 복사본, output, scratch, manifest가 저장됩니다.

## 플롯

구조화 로그를 읽는 유틸은 [loggers/reader.py](/home/jyhong/projects/drl_spice_v4/loggers/reader.py), 플롯 생성 유틸은 [loggers/plotter.py](/home/jyhong/projects/drl_spice_v4/loggers/plotter.py)에 있습니다.

프로젝트별 `plot_config.json`을 로그 디렉터리에 두면 기본 plot 설정을 덮어쓸 수 있습니다.

## 테스트

테스트 수집 확인:

```bash
pytest --collect-only -q
```

전체 테스트 실행:

```bash
pytest
```

현재 저장소 기준으로 `116`개 테스트가 수집됩니다.

## 참고

- 분석 노트북: [analysis.ipynb](/home/jyhong/projects/drl_spice_v4/analysis.ipynb)
- 회로 템플릿: `simulator/ngspice/schematic/`
- 환경 기본값: [exps/configs/lna_defaults.py](/home/jyhong/projects/drl_spice_v4/exps/configs/lna_defaults.py)
- agent 기본값: [exps/configs/agent_defaults.py](/home/jyhong/projects/drl_spice_v4/exps/configs/agent_defaults.py)
