import os
from huggingface_hub import hf_hub_download

# 1. 설정
# repo_id = "facebook/lagernvs_re10k_2v_256"
# repo_id = "facebook/lagernvs_general_512"
repo_id = "facebook/lagernvs_dl3dv_2-6_v_256"
original_name = "model.pt"
custom_name = "lagernvs_dl3dv_2-6_v_256.pt"  # 박사님이 원하는 이름
target_dir = "/home/jovyan/sungpyo/lagernvs/official_checkpoints"

# 2. 일단 원본 이름으로 다운로드 (이미 있으면 스킵함)
downloaded_path = hf_hub_download(
    repo_id=repo_id,
    filename=original_name,
    local_dir=target_dir,
    local_dir_use_symlinks=False
)

# 3. 이름 변경 (임의 설정)
new_path = os.path.join(target_dir, custom_name)

# 이미 바뀐 이름으로 파일이 있으면 굳이 또 안 바꿔도 되겠죠?
if not os.path.exists(new_path):
    os.rename(downloaded_path, new_path)
    print(f"✅ 파일명이 '{custom_name}'으로 변경되었습니다.")
else:
    # 만약 원본 파일(model.pt)이 남아있다면 삭제 (깔끔하게 관리)
    if os.path.exists(downloaded_path) and downloaded_path != new_path:
        os.remove(downloaded_path)
    print(f"✨ 이미 '{custom_name}' 파일이 존재합니다.")

print(f"📍 최종 위치: {new_path}") 