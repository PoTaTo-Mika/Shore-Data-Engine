import os
import subprocess
import platform
from pathlib import Path

# 使用 https://github.com/nilaoda/BBDown

d = os.path.dirname(os.path.abspath(__file__))
# 计算项目根目录（Shore-Data-Engine）
d_path = Path(d).resolve()
project_root = d_path.parents[1]
if not os.path.exists(os.path.join(d, "BBDown.data")):
    print("BBDown.data 文件不存在，请准备扫码登录")
    if platform.system() == "Windows":
        exe = os.path.join(d, "BBDown.exe")
    else:
        exe = os.path.join(d, "BBDown")
    if os.path.exists(exe):
        subprocess.run([exe, "login"], cwd=d, check=True)
    else:
        raise FileNotFoundError(f"未找到 {exe}")

# 获取视频列表 .\BBDown.exe "https://space.bilibili.com/1242381284/video" --save-archives-to-file
def get_video_list(url):
    if platform.system() == "Windows":
        exe_path = os.path.join(d, "BBDown.exe")
        cmd = [exe_path, url, "--save-archives-to-file"]
    else:
        cmd = ["./BBDown", url, "--save-archives-to-file"]

    # 允许非零退出码，继续执行后续逻辑（有些情况下 BBDown 会在完成写入后仍返回 1）
    result = subprocess.run(
        cmd,
        cwd=d,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        print(f"警告: BBDown 返回码 {result.returncode}，将继续执行。输出如下：")
        print(result.stdout[-2000:] if result.stdout else "")

    # 使用 Python 移动当前目录下的所有 .txt 到 项目根目录 data/list，并保留原文件名
    import shutil

    src_dir = d_path
    dest_dir = project_root / "data" / "list"
    dest_dir.mkdir(parents=True, exist_ok=True)

    for txt_file in src_dir.glob("*.txt"):
        shutil.move(str(txt_file), str(dest_dir / txt_file.name))

    print(f"获取视频列表成功，已移动到 {dest_dir}")

def download_list():
    # 为每个 txt 创建独立目录，并将仅音频按视频原名下载到该目录
    list_dir = project_root / "data" / "list"
    if not list_dir.exists():
        print(f"未找到目录: {list_dir}")
        return

    txt_files = sorted(list_dir.glob("*.txt"))
    if not txt_files:
        print(f"目录中未找到任何txt文件: {list_dir}")
        return

    # 确定 BBDown 可执行文件
    if platform.system() == "Windows":
        bbdown_exec = os.path.join(d, "BBDown.exe")
    else:
        bbdown_exec = "./BBDown"

    if not os.path.exists(os.path.join(d, os.path.basename(bbdown_exec))):
        raise FileNotFoundError(f"未找到 {bbdown_exec}")

    for txt_path in txt_files:
        # 以 txt 文件名（不含扩展名）创建目录 data/<stem>
        stem = txt_path.stem
        dest_dir = project_root / "data" / stem
        dest_dir.mkdir(parents=True, exist_ok=True)

        # 读取该 txt 内的链接（去重、忽略空行和注释）
        urls = []
        seen = set()
        try:
            with txt_path.open("r", encoding="utf-8") as f:
                for line in f:
                    url = line.strip()
                    if not url or url.startswith("#"):
                        continue
                    if url not in seen:
                        seen.add(url)
                        urls.append(url)
        except UnicodeDecodeError:
            with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    url = line.strip()
                    if not url or url.startswith("#"):
                        continue
                    if url not in seen:
                        seen.add(url)
                        urls.append(url)

        if not urls:
            print(f"{txt_path.name}: 无有效链接，跳过")
            continue

        print(f"开始下载 {txt_path.name} （{len(urls)} 条）到 {dest_dir} …")
        # 使用文件名模式将输出放入目标目录，文件名为视频原名
        # 这里不更改 cwd，避免影响 BBDown 对登录/配置的读取；由 file-pattern 控制输出路径
        # 使用绝对路径，确保写入项目根目录 data
        pattern = f"{dest_dir.as_posix()}/<videoTitle>"
        for idx, url in enumerate(urls, start=1):
            cmd = [bbdown_exec, url, "--audio-only", "--file-pattern", pattern]
            try:
                subprocess.run(cmd, cwd=d, check=True)
                print(f"[{txt_path.name} {idx}/{len(urls)}] 成功: {url}")
            except subprocess.CalledProcessError as e:
                print(f"[{txt_path.name} {idx}/{len(urls)}] 失败: {url} -> {e}")

if __name__ == "__main__":
    user_id = "1242381284"
    get_video_list(f"https://space.bilibili.com/{user_id}/video")
    download_list()
    