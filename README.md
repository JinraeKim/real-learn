# real-learn

부정확한 시뮬레이션에서 얻어진 학습기반 제어기를 사용하는 것이 아닌 반복적인 실험을 안전하게 수행하여 실제 모델로부터 직접 학습된 제어기를 사용한다.

## Installation

`environment.yml` 파일을 이용해서 `conda` 가상환경을 만든다.

```bash
conda create env -f environment.yml
```

가상환경은 프로젝트 루트 디렉터리에 `envs` 라는 이름으로 만들어진다.
이제, `envs`를 활성화 한다.

```bash
conda activate ./envs
```

다음으로 `fym` 패키지를 설치한다.

```bash
$ git clone https://github.com/fdcl-nrf/fym.git
$ cd fym
$ pip install -e .
```
