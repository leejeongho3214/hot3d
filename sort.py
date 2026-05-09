import re
import pandas as pd
import os

# 1) 파일 로드 (엑셀 경로만 바꿔줘)
home = os.path.expanduser("~")
path = os.path.join(home, "Desktop/all_ori.xlsx")
df = pd.read_excel(path)

# 필요한 컬럼 확인
required = {"obj_name", "reviewed_prompt"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

# 2) prompt에서 action/part 파싱
def parse_action_part(prompt: str):
    if not isinstance(prompt, str) or not prompt.strip():
        return None, None

    s = prompt.strip()
    s = re.sub(r"\s+", " ", s)          # 공백 정리
    s = s.rstrip(".")                   # 끝 마침표 제거
    s_low = s.lower()

    # action 후보 (길이가 긴 것부터 매칭)
    # "palmar grasp"처럼 2단어 action도 처리
    actions = [
        "palmar grasp",
        "power grasp",
        "precision grasp",
        "grasp",
        "pinch",
        "hook",
        "press",
        "grip",
    ]

    action = None
    for a in actions:
        if s_low.startswith(a + " "):
            action = a
            rest = s_low[len(a):].strip()
            break
    if action is None:
        # fallback: 첫 단어를 action으로
        parts = s_low.split()
        action = parts[0]
        rest = " ".join(parts[1:])

    # rest는 보통 "<part> of <object> with ..." 형태
    # "of" 앞을 part로 본다.
    part = None
    if " of " in rest:
        part = rest.split(" of ", 1)[0].strip()
    else:
        # "of"가 없는 이상치(있을 수 있음) 대비
        # "with" 앞까지를 part로 간주
        if " with " in rest:
            part = rest.split(" with ", 1)[0].strip()
        else:
            part = rest.strip()

    # 파트 표준화(원하면 더 추가)
    part = part.replace(" ", "_")  # long edge -> long_edge
    canonical = {
        "cap": "cap",
        "lid": "lid",
        "top": "top",
        "bottom": "bottom",
        "rim": "rim",
        "edge": "edge",
        "handle": "handle",
        "body": "body",
        "center": "center",
        "keypad": "keypad",
        "frame": "frame",
        "bridge": "bridge",
        "tail": "tail",
        "roof": "roof",
        "head": "head",
        "long_edge": "long_edge",
        "short_edge": "short_edge",
    }
    part = canonical.get(part, part)

    return action, part

df["obj_name"] = df["obj_name"].astype(str).str.lower().str.strip()
df["action"], df["part"] = zip(*df["reviewed_prompt"].map(parse_action_part))

# 파싱 실패한 행 제거
df = df.dropna(subset=["action", "part"])

# 3) 물체별 동작 카운트
obj_action_counts = (
    df.groupby(["obj_name", "action"])
      .size()
      .unstack(fill_value=0)
      .sort_index()
)

# 4) 물체별 파트 카운트
obj_part_counts = (
    df.groupby(["obj_name", "part"])
      .size()
      .unstack(fill_value=0)
      .sort_index()
)

# 5) 물체별 (동작 x 파트) 카운트 (가장 자세한 표)
obj_action_part_counts = (
    df.groupby(["obj_name", "action", "part"])
      .size()
      .reset_index(name="count")
      .sort_values(by=["obj_name", "action", "part"])
)