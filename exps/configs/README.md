# Experiment Config Reference

`exps/configs/*.py` 는 최종적으로 아래 형태의 dict 를 반환하면 됩니다.

```python
{
    "circuit_type": "CS" or "CGCS",
    "env_name": "modular",
    "seeds": [100, 200, 300],
    "env_kwargs": {...},
    "launch_kwargs": {...},
}
```

현재 진입점은 [run_cs.py](/home/jyhong/projects/drl_spice_v4/exps/scripts/run_cs.py) 와 [run_cgcs.py](/home/jyhong/projects/drl_spice_v4/exps/scripts/run_cgcs.py) 이며, 실제로는:

- `env_kwargs` 는 선택된 LNA env 생성자로 거의 그대로 전달됨
- `env_name` 으로 `modular`를 사용함
- `launch_kwargs` 는 `DeepRLTestModule(...)` 로 전달됨
- 그 다음 `launch_kwargs` 는 agent 생성자와 trainer runtime 에서 소비됨

## 1. Top-level keys

### `circuit_type`

- 타입: `str`
- 선택값: `"CS"`, `"CGCS"`

### `seeds`

- 타입: `list[int]`
- 의미: seed sweep 목록

### `env_name`

- 타입: `str`
- 기본값:
  - `CS`: `"modular"`
  - `CGCS`: `"modular"`
- 선택값:
  - `"modular"`: `LNAEnvBase`

### `env_kwargs`

- 타입: `dict`
- 의미: 환경 생성 인자

### `launch_kwargs`

- 타입: `dict`
- 의미: 학습 실행 인자 + agent 선택 플래그 + agent 하이퍼파라미터

## 2. `env_kwargs` 에 넣을 수 있는 항목

기준: [modular.py](/home/jyhong/projects/drl_spice_v4/envs/lna/lna_envs/modular.py)

### 필수 키

- `target_spec`
  타입: `dict[str, float]` 또는 1D sequence
  지원 metric:
  - `enable_iip3=True`: `S11`, `S21`, `S22`, `NF`, `PD`, `IIP3`
  - `enable_iip3=False`: `S11`, `S21`, `S22`, `NF`, `PD`

- `references`
  타입: `dict[str, float]` 또는 1D sequence
  키 규칙은 `target_spec` 과 동일

- `bound`
  타입: `dict[str, [min, max]]` 또는 `Nx2` sequence
  회로별 허용 키:
  - `CS`: `V_b`, `R_D`, `L_D`, `L_G`, `L_S`, `C_D`, `C_ex`, `XM1`, `XM2`
  - `CGCS`: `V_b1`, `V_b2`, `V_b3`, `V_b4`, `R_D1`, `R_D4`, `R_S5`, `C_D1`, `C_D4`, `C_S3`, `C_S4`, `XM1`, `XM2`, `XM3`, `XM4`, `XM5`

- `fixed_values`
  타입: `dict[str, float]`
  필수 키:
  - `V_dd`
  - `R_b`
  - `C_1`
  - `l_m`

### 선택 키

- `enable_iip3`
  타입: `bool`
  기본값: `True`

- `max_steps`
  타입: `int`
  기본값: `20`

- `max_param`
  타입: `float`
  기본값: `1.0`

- `n_restricted`
  타입: `int`
  기본값: `2`

- `p`
  타입: `float`
  기본값: `2.0`

- `beta`
  타입: `float`
  기본값: `0.2`

- `gamma`
  타입: `float`
  기본값: `0.99`

- `eta`
  타입: `float | None`
  기본값: `None`
  설명: `None` 이면 `2 * max_param / max_steps`

- `reset_probability`
  타입: `float`
  기본값: `0.0`
  설명: 기본 reset 정책이 `continue_last` 여도, 이 확률만큼 `random` reset으로 강제 전환함

- `allow_reset_fallback`
  타입: `bool`
  기본값: `False`

- `reset_fallback_after`
  타입: `int`
  기본값: `10`

- `penal_viol`
  타입: `float`
  기본값: `2.0`

- `penal_perf`
  타입: `float`
  기본값: `20.0`

- `lmda_viol`
  타입: `float`
  기본값: `1.0`
  설명: `reward_viol` 항 가중치

- `lmda_perf`
  타입: `float`
  기본값: `1.0`
  설명: `reward_perf` 항 가중치

- `lmda_var`
  타입: `float`
  기본값: `0.1`
  설명: `reward_var` 항 가중치

- `reward_name`
  타입: `str`
  기본값: `"default"`
  선택값:
  - `"default"`
  - `"reward"`
  설명: `modular` env 에서 사용할 reward 전략 이름. 현재는 단일 기본 reward 만 지원함.

실무적으로는 [lna_defaults.py](/home/jyhong/projects/drl_spice_v4/exps/configs/lna_defaults.py) 의 `build_lna_env_defaults()` 가 env 조절 가능한 기본 하이퍼파라미터 preset 입니다. 새 config 를 만들 때 defaults 를 복사해서 일부만 덮어쓰면 됩니다.

## 3. `launch_kwargs` 에 넣을 수 있는 항목

### 공통 trainer/runtime 키

기준: [launcher.py](/home/jyhong/projects/drl_spice_v4/exps/launcher.py#L13), [runtime.py](/home/jyhong/projects/drl_spice_v4/trains/runtime.py#L15)

- `load_path`
  타입: `str | None`
  설명: 체크포인트 로드 경로

- `max_iters`
  타입: `int`
  기본값: `10000`

- `n_runners`
  타입: `int`
  기본값:
  - on-policy: `1`
  - off-policy: `10`

- `runner_iters`
  타입: `int`
  기본값:
  - on-policy: `10`
  - off-policy: `4`

- `eval_mode`
  타입: `bool`
  기본값: `False`

- `eval_intervals`
  타입: `int`
  기본값:
  - on-policy: `200`
  - off-policy: `100`

- `eval_iters`
  타입: `int`
  기본값: `10`

- `seed`
  타입: `int`
  보통 바깥 루프에서 전달하므로 `launch_kwargs` 에 직접 안 넣어도 됨

- `project_name`
  타입: `str`
  기본값: launcher 가 자동 생성

- `circuit_type`
  타입: `str`
  보통 바깥 루프에서 전달하므로 `launch_kwargs` 에 직접 안 넣어도 됨

### off-policy 전용 trainer 키

기준: [runtime.py](/home/jyhong/projects/drl_spice_v4/trains/runtime.py#L127)

- `utd_ratio`
  타입: `float`
  기본값: `1.0`

- `checkpoint_intervals`
  타입: `int`
  기본값: `eval_intervals`

## 4. agent 선택 플래그

기준: [registry.py](/home/jyhong/projects/drl_spice_v4/exps/registry.py#L8)

`launch_kwargs` 에 아래 bool 플래그를 넣으면 해당 agent 가 실행됩니다.

- `ppo`
- `ddpg`
- `td3`
- `sac`
- `ddpg_per`
- `td3_per`
- `sac_per`
- `random`

설명:

- `*_per` 는 prioritized replay 모드 agent
- `random` 은 내부적으로 `DDPG` 클래스를 쓰지만 `update_after=max_iters` 로 사실상 랜덤 행동만 수행
- 여러 플래그를 동시에 `True` 로 두면 여러 agent 가 순차 실행됨

## 5. 모든 agent 공통 하이퍼파라미터

아래 키들은 `launch_kwargs` 에 넣으면 선택된 agent 생성자로 그대로 전달됩니다.

### 네트워크 구조

- `actor_size`
  타입: `tuple[int, ...]`

- `critic_size`
  타입: `tuple[int, ...]`

- `actor_activation`
  타입: callable
  예: `torch.relu`, `torch.tanh`

- `critic_activation`
  타입: callable
  예: `torch.relu`, `torch.tanh`

주의:

- `actor_activation`, `critic_activation` 은 Python callable 이므로 JSON/YAML 문자열이 아니라 Python config 파일 안에서 직접 지정해야 함

## 6. Off-policy 공통 하이퍼파라미터

기준: [policy.py](/home/jyhong/projects/drl_spice_v4/agents/base/policy.py#L129)

- `buffer_size`
- `batch_size`
- `update_after`
- `actor_lr`
- `critic_lr`
- `gamma`
- `tau`
- `prioritized_mode`
- `prio_alpha`
- `prio_beta`
- `prio_eps`

추가로 distributed collector 쪽에서 읽는 키:

- `collection_batch_size`
  타입: `int`
  기본값: `16`
  설명: runner actor 내부 rollout batch flush 단위

## 7. DDPG 전용 하이퍼파라미터

기준: [ddpg.py](/home/jyhong/projects/drl_spice_v4/agents/ddpg.py#L10)

- `max_grad_norm`
  기본값: `2.0`

- `action_noise_std`
  기본값: `0.1`

- `noise_type`
  선택값: `"normal"`, `"ou"`
  기본값: `"normal"`

## 8. TD3 전용 하이퍼파라미터

기준: [td3.py](/home/jyhong/projects/drl_spice_v4/agents/td3.py#L10)

- `update_freq`
  기본값: `2`

- `max_grad_norm`
  기본값: `2.0`

- `action_noise_std`
  기본값: `0.1`

- `target_noise_std`
  기본값: `0.2`

- `noise_clip`
  기본값: `0.5`

- `noise_type`
  선택값: `"normal"`, `"ou"`
  기본값: `"normal"`

## 9. SAC 공통 하이퍼파라미터

기준: [sac.py](/home/jyhong/projects/drl_spice_v4/agents/sac.py#L11)

- `update_freq`
  기본값: `1`

- `max_grad_norm`
  기본값: `5.0`

- `alpha`
  기본값: `0.2`

- `adaptive_alpha_mode`
  타입: `bool`
  기본값: `True`

- `ent_lr`
  기본값: `3e-4`

## 10. PPO 공통 하이퍼파라미터

기준: [ppo.py](/home/jyhong/projects/drl_spice_v4/agents/ppo.py#L10)

- `buffer_size`
- `update_after`
- `step_size`
  설명: PPO 에서는 `actor_lr` 가 아니라 `step_size` 키를 사용

- `gamma`
- `lmda`
- `vf_coef`
- `ent_coef`
- `adv_norm`
- `train_iters`
- `batch_size`
- `clip_range`
- `clip_range_vf`
- `target_kl`
- `max_grad_norm`

## 11. 현재 예제 config 에 실제로 들어있는 키

### `cs.py`

기준: [cs.py](/home/jyhong/projects/drl_spice_v4/exps/configs/cs.py#L1)

- `circuit_type="CS"`
- `seeds=[100]`
- `env_kwargs`
  - `target_spec`
  - `references`
  - `bound`
  - `fixed_values`
  - `enable_iip3=True`
  - `p=2.0`
  - `beta=0.2`
- `launch_kwargs`
  - `load_path=None`
  - `max_iters=10000`
  - `n_runners=10`
  - `eval_mode=False`
  - `ppo=False`
  - `ddpg=False`
  - `td3=False`
  - `sac=True`

### `cgcs.py`

기준: [cgcs.py](/home/jyhong/projects/drl_spice_v4/exps/configs/cgcs.py#L1)

- `circuit_type="CGCS"`
- `seeds=[800, 900, 1000]`
- `env_kwargs`
  - `target_spec`
  - `references`
  - `bound`
  - `fixed_values`
  - `enable_iip3=False`
  - `p=2.0`
  - `beta=0.5`
- `launch_kwargs`
  - `load_path=None`
  - `max_iters=10000`
  - `n_runners=10`
  - `eval_mode=False`
  - `ppo=True`
  - `ddpg=True`
  - `td3=True`
  - `sac=False`

## 12. 예시

```python
def build_cs_experiment_config():
    return {
        "circuit_type": "CS",
        "seeds": [100, 200, 300],
        "env_kwargs": {
            "target_spec": {
                "S11": -10.0,
                "S21": 20.0,
                "S22": -10.0,
                "NF": 2.0,
                "PD": 5.0,
                "IIP3": -5.0,
            },
            "references": {
                "S11": 0.0,
                "S21": 10.0,
                "S22": 0.0,
                "NF": 4.0,
                "PD": 10.0,
                "IIP3": -15.0,
            },
            "bound": {
                "V_b": [0.7, 1.0],
                "R_D": [10, 1000],
                "L_D": [1e-10, 2e-8],
                "L_G": [1e-10, 2e-8],
                "L_S": [1e-11, 2e-9],
                "C_D": [5e-14, 5e-12],
                "C_ex": [5e-15, 5e-13],
                "XM1": [1, 100],
                "XM2": [1, 100],
            },
            "bound_decode_mode": {
                "V_b": "lin",
            },
            "fixed_values": {
                "V_dd": 1.8,
                "R_b": 1e4,
                "C_1": 1e-11,
                "l_m": 0.15,
            },
            "enable_iip3": True,
            "max_steps": 20,
            "p": 2.0,
            "beta": 0.2,
        },
        "launch_kwargs": {
            "max_iters": 10000,
            "n_runners": 10,
            "runner_iters": 4,
            "eval_mode": False,
            "eval_intervals": 100,
            "utd_ratio": 1.0,
            "checkpoint_intervals": 100,
            "sac": True,
            "sac_per": False,
            "actor_size": (256, 256),
            "critic_size": (256, 256),
            "batch_size": 256,
            "update_after": 1000,
        },
    }
```
